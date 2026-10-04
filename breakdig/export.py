"""Cut sections out of the original files and write tagged WAVs."""

import csv
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from mutagen.id3 import COMM, TALB, TBPM, TIT2, TPE1
from mutagen.wave import WAVE

from . import audio
from .db import model_dir
from .isolate import PAD_SECONDS, isolate, stems_name
from .query import Section
from .separate import Separator

SNAP_SECONDS = 0.002
MAX_NAME = 120
# The MKII takes 48 kHz; the SX, A and the original 404 take 44.1 kHz.
SP404_RATES = {"mk2": 48000, "sx": 44100}
SP404_CSV = "index.csv"
SP404_FIELDS = ["file", "artist", "title", "bars", "bpm", "start", "end", "source", "stems"]
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# What _origin writes: source, start and end, then the key and the kept stems if there are any.
_ORIGIN = re.compile(r"(.*) (\d+\.\d{3})-(\d+\.\d{3})s(?: ([0-9a-f]+))?(?: \[([a-z+]+)\])?")


@dataclass
class Clip:
    audio: np.ndarray  # (frames, channels) float32
    sample_rate: int
    bits: int
    start: float = 0.0  # where the first frame sits in the source, in seconds
    stems: str = ""  # the stems kept, like 'drums', or '' for the whole mix


def quietest_point(x: np.ndarray, idx: int, radius: int) -> int:
    """The frame within +-radius of idx where the loudest channel is quietest, nearest idx on a tie.

    A zero crossing of the channel average can sit well away from zero on each channel."""
    lo, hi = max(0, idx - radius), min(len(x), idx + radius + 1)
    if hi <= lo:
        return idx
    loudest = np.abs(x[lo:hi]).max(axis=1)
    frames = np.arange(lo, hi)
    return int(frames[np.lexsort((np.abs(frames - idx), loudest))[0]])


def cut(path: str, start: float, end: float, keep=(), separator=None) -> Clip:
    """Decode [start, end) at the source rate with both edges moved to the quietest point nearby.

    With keep, only those stems of it, separated from the section and a few seconds either side."""
    p = audio.probe(path)
    margin = PAD_SECONDS if keep else 0.01
    t0 = max(0.0, start - margin)
    x = audio.decode(path, start=t0, end=end + margin, info=p)
    sr = p.sample_rate
    if keep:
        x = isolate(x, sr, keep, separator)
    radius = int(round(SNAP_SECONDS * sr))
    a = quietest_point(x, int(round((start - t0) * sr)), radius)
    b = quietest_point(x, int(round((end - t0) * sr)), radius)
    return Clip(np.ascontiguousarray(x[a:b]), sr, p.bits, t0 + a / sr, stems_name(keep))


def for_sp404(clip: Clip, rate: int) -> Clip:
    x = clip.audio
    if x.shape[1] > 2:
        x = x[:, :2]
    if clip.sample_rate != rate:
        x = soxr.resample(x, clip.sample_rate, rate, quality="VHQ")
    peak = np.abs(x).max(initial=0.0)
    if peak > 1.0:
        x = x / peak  # 16-bit has no headroom; turn it down rather than clip the transients
    return Clip(x.astype(np.float32), rate, 16, clip.start, clip.stems)


def safe_name(text: str) -> str:
    return _ILLEGAL.sub("_", text).strip().rstrip(".")


def capped_name(head: str, suffix: str) -> str:
    """head + suffix, with head sanitized and shortened so the whole stays under MAX_NAME."""
    head = safe_name(head)
    return head[: MAX_NAME - len(suffix)].rstrip(" .") + suffix


def clip_name(s: Section, stems: str = "") -> str:
    """'{artist} - {title} - {bars} bars - {bpm} BPM - {mm.ss}.wav', at most 120 characters, with
    ' - {stems}' before the bars if only some stems are kept.

    Only the artist/title part is shortened, so the bars, tempo and position survive."""
    mm, ss = divmod(round(s.start), 60)
    bpm = f"{s.bpm:.0f}" if s.bpm else "unknown"
    suffix = f"{f' - {stems}' if stems else ''} - {s.bars} bars - {bpm} BPM - {mm:02d}.{ss:02d}.wav"
    return capped_name(f"{s.artist} - {s.title}", suffix)


def _origin(s: Section, stems: str) -> str:
    return f"{s.path} {s.start:.3f}-{s.end:.3f}s {s.key}".rstrip() + (f" [{stems}]" if stems else "")


def _identities(source: str, start, end, key: str | None, stems: str | None) -> set[tuple]:
    """A section counts as exported if its path or its content key matches, so an export is still
    found after the library moves, or after a retag changes the key."""
    span = (round(float(start), 3), round(float(end), 3), stems or "")
    return {("path", source, *span)} | ({("key", key, *span)} if key else set())


def _mine(s: Section, stems: str) -> set[tuple]:
    return _identities(s.path, s.start, s.end, s.key, stems)


def _our_tags(path: Path):
    """The ID3 tags of a WAV that breakdig wrote, and its source, start, end, key and stems, or None."""
    try:
        tags = WAVE(path).tags
    except Exception:
        return None  # not ours, or not a WAV breakdig can read; leave it alone
    comm = tags.getall("COMM:breakdig:eng") if tags else []
    m = _ORIGIN.fullmatch(str(comm[0])) if comm else None
    return (tags, m.groups()) if m else None


def exported_as(path: Path) -> set[tuple]:
    """The identities recorded in a WAV that breakdig wrote, or none for anything else."""
    ours = _our_tags(path)
    return _identities(*ours[1]) if ours else set()


def free_path(out: Path, s: Section, stems: str = "") -> Path:
    """Where to write s: its clip name, or 'name (2).wav' and so on if another section has it."""
    name = clip_name(s, stems)
    stem, n, mine = name[:-4], 1, _mine(s, stems)
    while True:
        dest = out / name
        if not dest.exists() or exported_as(dest) & mine:
            return dest
        n += 1
        name = f"{stem} ({n}).wav"


def beat_frames(clip: Clip, beats: list[float]) -> list[int]:
    """Where the beats fall in the clip, as frame offsets. The first can sit a few ms before the
    snapped start; it moves to frame 0."""
    frames = [max(0, round((t - clip.start) * clip.sample_rate)) for t in beats]
    return sorted({f for f in frames if f < len(clip.audio)})


def _chunk(fourcc: bytes, body: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(body)) + body + b"\0" * (len(body) % 2)


def add_cues(path: Path, frames: list[int], labels: list[str] | None = None) -> None:
    """Append a cue point per frame to a RIFF WAV, labelled 'beat 1' and on unless labels
    are given."""
    if not frames:
        return
    labels = labels or [f"beat {i}" for i in range(1, len(frames) + 1)]
    cue = struct.pack("<I", len(frames)) + b"".join(
        struct.pack("<II4sIII", i, f, b"data", 0, 0, f) for i, f in enumerate(frames, 1))
    labels = b"".join(_chunk(b"labl", struct.pack("<I", i) + label.encode() + b"\0")
                      for i, label in enumerate(labels, 1))
    with open(path, "r+b") as f:
        if f.read(4) != b"RIFF":
            return  # RF64, past 4 GB
        f.seek(0, os.SEEK_END)
        f.write(_chunk(b"cue ", cue) + _chunk(b"LIST", b"adtl" + labels))
        size = f.tell() - 8
        f.seek(4)
        f.write(struct.pack("<I", size))


def write_wav(clip: Clip, dest: Path, s: Section) -> None:
    x = clip.audio
    if 0 < clip.bits <= 16:
        subtype = "PCM_16"
    elif np.abs(x).max(initial=0.0) > 1.0:
        subtype = "FLOAT"  # MP3 and AAC often decode past full scale on loud masters
    else:
        subtype = "PCM_24"
    if subtype != "FLOAT":
        x = np.clip(x, -1.0, 1.0)
    tmp = dest.with_name(dest.name + ".part")
    try:
        sf.write(tmp, x, clip.sample_rate, subtype=subtype, format="WAV")
        add_cues(tmp, beat_frames(clip, s.beats))
        w = WAVE(tmp)
        w.add_tags()
        w.tags.add(TIT2(encoding=3, text=f"{s.title} (bars {s.first_bar}-{s.last_bar})"))
        w.tags.add(TPE1(encoding=3, text=s.artist))
        if s.album:
            w.tags.add(TALB(encoding=3, text=s.album))
        if s.bpm:
            w.tags.add(TBPM(encoding=3, text=f"{s.bpm:.0f}"))
        w.tags.add(COMM(encoding=3, lang="eng", desc="breakdig", text=_origin(s, clip.stems)))
        w.save()
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_csv(path: Path) -> tuple[list[dict], list[str], str]:
    """Rows, columns and delimiter. Excel saves plain "CSV" as ANSI, and with semicolons in
    countries that write decimals with a comma."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, newline="", encoding=encoding) as f:
                header = f.readline()
                f.seek(0)
                delimiter = ";" if header.count(";") > header.count(",") else ","
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader), list(reader.fieldnames or []), delimiter
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot read {path}: save it as CSV UTF-8")


def _write_csv(out: Path, rows: list[dict], columns: list[str], delimiter: str) -> None:
    tmp = out / (SP404_CSV + ".part")
    try:
        # The BOM makes Excel open it as UTF-8 instead of garbling accented names.
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, out / SP404_CSV)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"cannot write {out / SP404_CSV}. Is it open in Excel? "
                         "Close it and run again.") from None


def _csv_row(file: str, artist: str, title: str, bars, bpm: str, start, end, source: str, stems: str) -> dict:
    return {"file": file, "artist": artist, "title": title, "bars": bars, "bpm": bpm,
            "start": f"{float(start):.3f}", "end": f"{float(end):.3f}", "source": source, "stems": stems}


class Sp404Folder:
    """BRK_0001.WAV and on, listed in index.csv. What is done comes from the files' own tags,
    since Excel can reformat the csv's numbers."""

    def __init__(self, out: Path, rate: int):
        self.out, self.rate = out, rate
        path = out / SP404_CSV
        self.rows, self.columns, self.delimiter = _read_csv(path) if path.exists() else ([], [], ",")
        self.columns += [c for c in SP404_FIELDS if c not in self.columns]
        listed = {(r.get("file") or "").upper() for r in self.rows}
        self.done = set()
        for p in sorted(out.glob("BRK_*.WAV")):
            ours = _our_tags(p)
            if ours:
                if (had := sf.info(p).samplerate) != rate:  # one card cannot take both
                    raise ValueError(f"{out} already has {had / 1000:g} kHz SP-404 clips; "
                                     "export to another folder")
                self.done |= _identities(*ours[1])
                if p.name.upper() not in listed:  # written by an export that was killed
                    self.rows.append(_row_from_tags(p.name, *ours))
        names = {p.name.upper() for p in out.glob("BRK_*.WAV")} | listed
        self.next_no = max((int(m.group(1)) for n in names if (m := re.fullmatch(r"BRK_(\d+)\.WAV", n))),
                           default=0) + 1

    def has(self, s: Section, stems: str) -> bool:
        return bool(_mine(s, stems) & self.done)

    def next_path(self) -> Path:
        self.next_no += 1
        return self.out / f"BRK_{self.next_no - 1:04d}.WAV"

    def add(self, dest: Path, s: Section, stems: str) -> None:
        self.rows.append(_csv_row(dest.name, s.artist, s.title, s.bars, f"{s.bpm:.1f}" if s.bpm else "",
                                  s.start, s.end, s.path, stems))
        self.done |= _mine(s, stems)

    def save(self) -> None:
        _write_csv(self.out, self.rows, self.columns, self.delimiter)


def _row_from_tags(file: str, tags, origin: tuple) -> dict:
    source, start, end, _, stems = origin
    title, bars = str(tags.get("TIT2", "")), ""
    if m := re.fullmatch(r"(.*) \(bars (\d+)-(\d+)\)", title):
        title, bars = m.group(1), int(m.group(3)) - int(m.group(2)) + 1
    bpm = f"{float(str(tags['TBPM'])):.1f}" if "TBPM" in tags else ""
    return _csv_row(file, str(tags.get("TPE1", "")), title, bars, bpm, start, end, source, stems or "")


def load_separator() -> Separator:
    try:
        return Separator(model_dir())
    except Exception as e:
        raise ValueError(f"cannot load the separation model: {type(e).__name__}: {e}") from e


@dataclass
class Exported:
    section: Section
    path: str | None
    error: str | None = None
    skipped: bool = False


def export(sections: list[Section], out_dir: str, sp404: str | None = None, keep=(),
           separator: Separator | None = None) -> list[Exported]:
    """Write each section to out_dir, for the SP-404 model sp404 names if given. With keep, only
    those stems of each, separated with separator or with one loaded when the first section needs it."""
    if sp404 and sp404 not in SP404_RATES:
        raise ValueError(f"unknown SP-404 model {sp404!r}; use {' or '.join(SP404_RATES)}")
    out = Path(out_dir).resolve()
    stems = stems_name(keep)
    if out.exists() and not out.is_dir():
        raise ValueError(f"{out} is a file, not a folder")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"cannot create {out}: {e.strerror}") from None
    sections = list({s.id: s for s in sections}.values())
    results = []
    folder = Sp404Folder(out, SP404_RATES[sp404]) if sp404 else None
    if folder:
        folder.save()  # fails here, before any clip, if Excel has it open
    try:
        for s in sections:
            if not os.path.exists(s.path):
                results.append(Exported(s, None, f"source file is gone: {s.path} "
                                                 "(if it moved, run breakdig index on its new folder)"))
                continue
            if folder and folder.has(s, stems):
                results.append(Exported(s, None, skipped=True))
                continue
            if keep and separator is None:
                separator = load_separator()
            try:
                clip = cut(s.path, s.start, s.end, keep, separator)
                if folder:
                    clip, dest = for_sp404(clip, folder.rate), folder.next_path()
                else:
                    dest = free_path(out, s, stems)
                write_wav(clip, dest, s)
            except (audio.DecodeError, OSError, RuntimeError) as e:
                results.append(Exported(s, None, str(e)))
                continue
            if folder:
                folder.add(dest, s, stems)
            results.append(Exported(s, str(dest)))
    finally:
        if folder:
            folder.save()
    return results
