"""Find audio files and compute their content keys."""

import fnmatch
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import mutagen
from mutagen.id3 import ID3

from . import audio

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".aiff", ".aif", ".aifc", ".m4a", ".ogg", ".opus", ".aac", ".wma",
              ".ape", ".wv", ".mpc", ".mp2"}
# Picked up only so they can be reported as skipped instead of silently ignored.
DRM_EXTS = {".m4p"}
SAMPLE_BYTES = 1 << 20
# Deleted files keep their extension in the recycle bin, so indexing a drive root would find them.
SYSTEM_DIRS = {"$recycle.bin", "system volume information"}
ID3_FRAMES = {"artist": "TPE1", "albumartist": "TPE2", "title": "TIT2", "album": "TALB"}


@dataclass
class Found:
    path: str
    key: str
    size: int
    duration: float  # -1 when neither mutagen nor ffprobe could read it
    mtime_ns: int = 0


def on_disk(path: str) -> str:
    """The absolute path as the disk spells it, so d:\\music and D:\\Music are one path. A mapped or
    substituted drive keeps its letter rather than turning into the path behind it."""
    path, real = os.path.abspath(path), os.path.realpath(path)
    drive, tail = os.path.splitdrive(path)
    if os.path.splitdrive(real)[0].lower() == drive.lower():
        return real
    # Take only the case of the names from the real path, and only where they are the same names.
    n = len(Path(tail).parts) - 1
    drive = drive.upper() if len(drive) == 2 else drive
    spelled = os.path.join(drive + os.sep, *Path(real).parts[-n:]) if n > 0 else drive + tail
    return spelled if os.path.normcase(spelled) == os.path.normcase(path) else drive + tail


def walk(roots: list[str], exclude: list[str] = (), skip: list[str] = ()) -> list[str]:
    """Audio files under roots. Files and folders whose name matches an exclude glob are skipped,
    and so are the folders in skip."""
    def excluded(name):
        return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in exclude)

    def skipped(folder):
        # Compared behind any mapped or substituted drive, which the index home is resolved through.
        return (os.path.normcase(os.path.basename(folder)) in skip_names
                and os.path.normcase(os.path.realpath(folder)) in skip)

    skip = {os.path.normcase(os.path.realpath(p)) for p in skip}
    skip_names = {os.path.basename(p) for p in skip}
    out = []
    for root in roots:
        root = on_disk(root)
        if os.path.isfile(root):
            out.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not excluded(d) and d.lower() not in SYSTEM_DIRS
                                 and not skipped(os.path.join(dirpath, d)))
            for name in sorted(filenames):
                # ._ files are the resource forks macOS leaves on FAT and exFAT drives.
                if (Path(name).suffix.lower() in AUDIO_EXTS | DRM_EXTS and not excluded(name)
                        and not name.startswith("._")):
                    out.append(os.path.join(dirpath, name))
    return out


def duration_of(path: str) -> float:
    try:
        f = mutagen.File(path)
        if f is not None and f.info and f.info.length > 0:
            return float(f.info.length)
    except Exception:
        pass
    try:
        return audio.probe(path).duration
    except audio.DecodeError:
        return -1.0


def content_key(path: str) -> Found:
    """sha1 over a MiB each from the start, middle and end, the byte size and the duration.

    This catches byte-identical copies cheaply. The middle and end matter for
    WAV edits of one track, which share a length and often the intro. The same
    recording in two formats has different bytes; the indexer catches that
    later by comparing decoded loudness envelopes."""
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for offset in (0, st.st_size // 2, st.st_size - SAMPLE_BYTES):
            f.seek(max(0, offset))
            h.update(f.read(SAMPLE_BYTES))
    duration = duration_of(path)
    h.update(f"|{st.st_size}|{duration:.2f}".encode())
    return Found(path, h.hexdigest(), st.st_size, duration, st.st_mtime_ns)


def full_key(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _first(tags, *names) -> str:
    for n in names:
        v = tags.get(n) if tags else None
        if v:
            v = v[0] if isinstance(v, list) else v
            v = str(v).strip()
            if v:
                return v
    return ""


def read_tags(path: str) -> dict:
    """artist/title/album from tags, falling back to an 'Artist - Title' filename."""
    tags = None
    try:
        tags = mutagen.File(path, easy=True)
        if tags is not None and isinstance(tags.tags, ID3):  # AIFF and WAV have no easy wrapper
            tags = {name: tags.tags[frame].text for name, frame in ID3_FRAMES.items() if frame in tags.tags}
    except Exception:
        pass
    artist = _first(tags, "artist", "albumartist")
    title = _first(tags, "title")
    album = _first(tags, "album")
    if not (artist and title):
        # WAV LIST/INFO and WMA tags, which mutagen's easy mode does not read.
        tags = audio.tags(path)
        artist = artist or _first(tags, "artist", "album_artist")
        title = title or _first(tags, "title")
        album = album or _first(tags, "album")
    if not (artist and title):
        stem = Path(path).stem
        parts = [p.strip() for p in stem.split(" - ")]
        # "03 - Artist - Title" or "03 - Title": a leading track number is not the artist.
        if len(parts) >= 2 and re.fullmatch(r"\d{1,3}", parts[0]) and (len(parts) >= 3 or len(parts[0]) <= 2):
            parts = parts[1:]
        if len(parts) >= 2:
            artist = artist or parts[0]
            title = title or " - ".join(parts[1:])
        else:
            title = title or parts[0]
    return {"artist": artist or "Unknown Artist", "title": title, "album": album}
