"""Practice drills: a tempo ladder rendered into one plain audio file per song.

Each rung of the ladder is a click count-in, the song stretched to that rung's
speed, and a bar or so of silence, so the file trains the same way Anytune's
Step-It-Up loop trainer plays: one tempo per repetition, climbing to full speed.
Stretching is Signalsmith Stretch through python-stretch, whose timeFactor is
the speed multiplier: timeFactor 0.7 makes the pass 1/0.7 times as long, which
plays at 70% speed.
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr
import soundfile as sf
from mutagen.id3 import COMM, ID3, TALB, TLEN, TIT2, TPE1
from mutagen.wave import WAVE

from . import mp3
from . import __version__
from .export import add_cues, capped_name

DEFAULT_LADDER = "70-100:5"
# At or above this a rung sits inside Signalsmith Stretch's 0.75x-1.5x sweet spot.
SWEET_SPOT_FLOOR = 65
DEFAULT_BPM = 120.0
BAR_BEATS = 4
CLICK_SECONDS = 0.030
FADE_SECONDS = 0.005
# Beat 1 of the count-in is louder and higher, the rest softer and lower.
ACCENT_CLICK = (1568.0, 0.85)
NORMAL_CLICK = (1046.5, 0.55)
SEPARATOR_BEEP = (880.0, 0.15, 0.15)  # hertz, beep length, silence between the two beeps


@dataclass(frozen=True)
class Ladder:
    label: str  # as typed, without the step, e.g. '70-100'
    percents: tuple[int, ...]


def parse_ladder(text: str) -> Ladder:
    """'70-100:5' into the rung speeds 70, 75, ..., 100. The end speed is always a rung."""
    rest, _, step_text = text.partition(":")
    try:
        start, end = (int(x) for x in rest.split("-"))
        step = int(step_text) if step_text else 5
    except ValueError:
        raise ValueError(f"--ladder takes START-END:STEP like {DEFAULT_LADDER}, got {text!r}") from None
    if not 50 <= start < end <= 100:
        raise ValueError(f"--ladder speeds must satisfy 50 <= start < end <= 100, got {text!r}")
    if not 1 <= step <= 20:
        raise ValueError(f"--ladder step must be 1 to 20, got {text!r}")
    percents = list(range(start, end, step))
    if percents[-1] != end:
        percents.append(end)
    return Ladder(f"{start}-{end}", tuple(percents))


@dataclass(frozen=True)
class Rung:
    percent: int
    time_factor: float  # percent / 100: the speed the pass plays at
    pass_seconds: float  # the drilled audio at this rung's speed
    count_in_seconds: float  # the clicks before the pass
    gap_seconds: float  # the silence after each pass but the file's very last one
    offset: float  # where this rung's count-in starts in the output


def plan(song_seconds: float, bpm: float, ladder: Ladder, passes: int = 1,
         count_in: int = 4, gap_bars: int = 1) -> list[Rung]:
    """The rungs of one drill, in playing order, pure and deterministic.

    Each rung contributes its count-in, then `passes` passes with a gap after
    every pass but the file's very last one. The count-in and the gap run at the
    rung's own tempo, so the click always matches what follows. A bar is four
    beats at the song's bpm."""
    if song_seconds <= 0:
        raise ValueError(f"nothing to drill: the song is {song_seconds:.3f}s long")
    if bpm <= 0:
        raise ValueError(f"a drill needs a tempo, got {bpm:g} BPM")
    beat = 60.0 / bpm
    rungs = []
    t = 0.0
    for i, percent in enumerate(ladder.percents):
        tf = percent / 100.0
        rung_beat = beat / tf
        gap_seconds = gap_bars * BAR_BEATS * rung_beat
        last = i == len(ladder.percents) - 1
        rung = Rung(percent, tf, song_seconds / tf, count_in * rung_beat, gap_seconds, t)
        rungs.append(rung)
        # count-in, every pass with its gap, then the final rung's trailing gap back off.
        t += rung.count_in_seconds + passes * (rung.pass_seconds + gap_seconds)
        if last:
            t -= gap_seconds
    return rungs


def total_seconds(rungs: list[Rung], passes: int = 1) -> float:
    """What render() adds up to, including every count-in and gap."""
    t = sum(r.count_in_seconds + passes * (r.pass_seconds + r.gap_seconds) for r in rungs)
    return t - rungs[-1].gap_seconds if rungs else 0.0


def click(rate: int, freq: float, amplitude: float) -> np.ndarray:
    """A short decaying burst: how a metronome sounds, rather than a square edge."""
    t = np.arange(round(CLICK_SECONDS * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.008)).astype(np.float32)


def _place(buf: np.ndarray, sound: np.ndarray, frame: int) -> None:
    n = min(len(sound), len(buf) - frame)
    if n > 0:
        buf[frame: frame + n] += sound[:n, None]


def _count_in(rung: Rung, count: int, rate: int, channels: int) -> np.ndarray:
    """Clicks on `count` beats at the rung's own beat length; the music enters one beat
    after the last click. Beat 1 is accented, and every click lands on a whole frame."""
    buf = np.zeros((round(rung.count_in_seconds * rate), channels), dtype=np.float32)
    beat = rung.count_in_seconds / count
    accent, normal = click(rate, *ACCENT_CLICK), click(rate, *NORMAL_CLICK)
    for beat_index in range(count):
        _place(buf, accent if beat_index == 0 else normal, round(beat_index * beat * rate))
    return buf


def _fade_edges(y: np.ndarray, rate: int, seconds: float = FADE_SECONDS) -> None:
    n = min(round(seconds * rate), len(y) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        y[:n] *= ramp[:, None]
        y[len(y) - n:] *= ramp[::-1][:, None]


def _stretched(mix: np.ndarray, rung: Rung, rate: int, out_rate: int) -> np.ndarray:
    """The mix at the rung's speed, faded in and out, exactly rung.pass_seconds long."""
    import python_stretch

    channels = mix.shape[1]
    stretch = python_stretch.Signalsmith.Stretch()
    stretch.preset(channels, rate)
    stretch.timeFactor = rung.time_factor
    # copy=True: a mono mix transposes to an already-contiguous view of the read-only
    # decode buffer, which the binding refuses; it needs an array of its own.
    y = stretch.process(np.array(mix.T, dtype=np.float32, order="C", copy=True)).T
    if out_rate != rate:
        y = soxr.resample(y, rate, out_rate, quality="VHQ")
    planned = round(rung.pass_seconds * out_rate)
    y = y[:planned] if len(y) >= planned else np.pad(y, ((0, planned - len(y)), (0, 0)))
    _fade_edges(y, out_rate)
    return y.astype(np.float32, copy=False)


def separator_beep(rate: int, channels: int) -> np.ndarray:
    """Two 880 Hz beeps with a beep-sized gap: one song is over, the next is coming."""
    hz, length, silence = SEPARATOR_BEEP
    n = round(length * rate)
    beep = np.zeros((n, 1), dtype=np.float32)
    beep[:, 0] = 0.6 * np.sin(2 * np.pi * hz * np.arange(n) / rate)
    _fade_edges(beep, rate)
    both = np.repeat(beep, channels, axis=1)
    return np.concatenate([both, np.zeros((round(silence * rate), channels), dtype=np.float32), both])


def render(mix: np.ndarray, rate: int, out_rate: int, rungs: list[Rung], passes: int = 1,
           count_in: int = 4) -> Iterator[np.ndarray]:
    """Yield the drill as float32 (frames, channels) blocks at out_rate, in playing order.

    mix is the audio to drill, already cut to the section if there is one. The
    passes come out of the stretcher at rate and are resampled once, so the whole
    file never sits in memory twice."""
    channels = mix.shape[1]
    for i, rung in enumerate(rungs):
        yield _count_in(rung, count_in, out_rate, channels)
        for rep in range(passes):
            yield _stretched(mix, rung, rate, out_rate)
            if not (i == len(rungs) - 1 and rep == passes - 1):
                yield np.zeros((round(rung.gap_seconds * out_rate), channels), dtype=np.float32)


def wav_subtype(bits: int, peak: float) -> str:
    """The same depth rule as every breakdig export: keep 16-bit sources at 16, go float
    when the master decodes past full scale, 24-bit for the rest."""
    if 0 < bits <= 16:
        return "PCM_16"
    return "FLOAT" if peak > 1.0 else "PCM_24"


def drill_name(artist: str, title: str, label: str, ext: str) -> str:
    """'{artist} - {title} - drill {label}.mp3', capped like every breakdig filename."""
    return capped_name(f"{artist} - {title}", f" - drill {label}.{ext}")


def origin_text(source: str, label: str, passes: int, count_in: int, gap_bars: int,
                bars: str, stems: str, key: str) -> str:
    """What the comment tag records: source, the drill's own settings, key, version."""
    parts = f"{source} drill {label} passes={passes} count_in={count_in} gap_bars={gap_bars}"
    if bars:
        parts += f" bars {bars}"
    if stems:
        parts += f" stems {stems}"
    if key:
        parts += f" {key}"
    return f"{parts} breakdig {__version__}"


# The same shape as the export tags: everything after the source path is machine-readable,
# so a rerun can tell its own files from a stranger's with the same name. The version is
# recorded but not part of the identity, so a newer breakdig overwrites its older files.
_ORIGIN = re.compile(r"(.+) drill (\d+-\d+) passes=(\d+) count_in=(\d+) gap_bars=(\d+)"
                     r"(?: bars (\d+-\d+))?(?: stems (\S+))? ([0-9a-f]{40})(?: breakdig (\S+))?")


def read_origin(path: Path) -> tuple | None:
    """(source, label, passes, count_in, gap_bars, bars, stems, key, version) from a drill
    we wrote."""
    try:
        tags = WAVE(path).tags if path.suffix.lower() == ".wav" else ID3(path)
    except Exception:
        return None
    comments = tags.getall("COMM:breakdig:eng") if tags else []
    m = _ORIGIN.fullmatch(str(comments[0])) if comments else None
    return m.groups() if m else None


def _identities(origin: tuple) -> set[tuple]:
    source, label, passes, count_in, gap_bars, bars, stems, key, _version = origin
    fixed = (label, int(passes), int(count_in), int(gap_bars), bars or "", stems or "")
    return {("path", source, *fixed), ("key", key, *fixed)}


def mine(source: str, key: str, label: str, passes: int, count_in: int, gap_bars: int,
         bars: str, stems: str) -> set[tuple]:
    """A drill counts as ours if its source path or its content key matches, so a rerun
    overwrites its own file even after the library has moved."""
    fixed = (label, int(passes), int(count_in), int(gap_bars), bars or "", stems or "")
    return {("path", source, *fixed)} | ({("key", key, *fixed)} if key else set())


def free_path(out: Path, name: str, identities: set[tuple]) -> Path:
    """Where to write the drill: its name, or 'name (2).mp3' when a file of that name
    belongs to a different drill."""
    suffix = Path(name).suffix
    stem, n = name[: -len(suffix)], 1
    while True:
        dest = out / name
        origin = read_origin(dest) if dest.exists() else None
        if origin is None and not dest.exists() or origin is not None and _identities(origin) & identities:
            return dest
        n += 1
        name = f"{stem} ({n}){suffix}"


class WavSink:
    """Writes float32 blocks to a WAV as they come, then tags it like every breakdig export."""

    def __init__(self, dest: Path, channels: int, rate: int, subtype: str):
        self.dest, self.tmp = dest, dest.with_name(dest.name + ".part")
        self.f = sf.SoundFile(self.tmp, "w", samplerate=rate, channels=channels,
                              subtype=subtype, format="WAV")

    def write(self, block: np.ndarray) -> None:
        self.f.write(block)

    def finish(self, artist: str, title: str, album: str, seconds: float, comment: str,
               cues: list[tuple[int, str]] | None = None) -> None:
        self.f.close()
        if cues:
            add_cues(self.tmp, [frame for frame, _ in cues], [label for _, label in cues])
        w = WAVE(self.tmp)
        w.add_tags()
        w.tags.add(TIT2(encoding=3, text=title))
        w.tags.add(TPE1(encoding=3, text=artist))
        if album:
            w.tags.add(TALB(encoding=3, text=album))
        w.tags.add(TLEN(encoding=3, text=str(round(seconds * 1000))))
        w.tags.add(COMM(encoding=3, lang="eng", desc="breakdig", text=comment))
        w.save()
        os.replace(self.tmp, self.dest)

    def abort(self) -> None:
        self.f.close()
        self.tmp.unlink(missing_ok=True)


class Mp3Sink:
    """Streams float32 blocks into ffmpeg's libmp3lame, then tags the file with mutagen."""

    def __init__(self, dest: Path, channels: int, rate: int):
        self.dest, self.tmp = dest, dest.with_name(dest.name + ".part")
        self.enc = mp3.Encoder(self.tmp, channels, rate)

    def write(self, block: np.ndarray) -> None:
        self.enc.write(block)

    def finish(self, artist: str, title: str, album: str, seconds: float, comment: str,
               cues: list[tuple[int, str]] | None = None) -> None:
        self.enc.close()  # cue points are a WAV feature; MP3 cue points stay out of scope
        mp3.add_id3(self.tmp, artist, title, album, seconds, comment)
        os.replace(self.tmp, self.dest)

    def abort(self) -> None:
        self.enc.proc.stdin.close()
        self.enc.proc.wait()
        self.tmp.unlink(missing_ok=True)


def open_sink(fmt: str, dest: Path, channels: int, rate: int, subtype: str = "PCM_16"):
    if fmt == "mp3":
        return Mp3Sink(dest, channels, rate)
    return WavSink(dest, channels, rate, subtype)
