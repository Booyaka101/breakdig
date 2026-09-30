"""The index pipeline: decode, dedupe, beats, separation, profile, commit."""

import errno
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio, profile, scan
from .beats import MIN_DOWNBEATS, BeatTracker
from .db import STEMS, Index, model_dir
from .separate import CHUNK_SECONDS, LONG_FILE_SECONDS, SAMPLE_RATE, Separator

log = logging.getLogger("breakdig")

# The separator gets a copy that peaks at -6 dB or lower so no stem clips (see separate.py);
# every level is read back up by the same amount.
INPUT_GAIN = 0.5
SAME_AUDIO_FLOOR_DB = -60.0
MAX_LAG = 3
MIN_FRAMES = 100


class Abort(RuntimeError):
    """A problem with the machine rather than the file, so the run stops and the file stays to do."""


class Skip(Exception):
    """A file that cannot be indexed, with a reason worth showing the user."""

    def __init__(self, status: str, reason: str, **extra):
        super().__init__(reason)
        self.status = status
        self.extra = extra


def match_lag(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG) -> int | None:
    """The offset in frames at which two 50 ms dB envelopes are the same recording, or None.

    Decoder delay differs between formats, and lossy encoding moves levels by
    a fraction of a dB, so this compares shapes rather than bytes. Frames under
    -60 dB are clamped: a lossy codec turns digital silence into -60 dB of
    pre-echo, which says nothing about whether the music is the same."""
    n = min(len(a), len(b)) - 2 * max_lag
    if n < MIN_FRAMES:
        return None
    a, b = np.maximum(a, SAME_AUDIO_FLOOR_DB), np.maximum(b, SAME_AUDIO_FLOOR_DB)
    x = a[max_lag: max_lag + n]
    best, best_diff = None, 1.0
    for lag in range(-max_lag, max_lag + 1):
        y = b[max_lag + lag: max_lag + lag + n]
        loud = (x > SAME_AUDIO_FLOOR_DB) | (y > SAME_AUDIO_FLOOR_DB)
        if loud.sum() < 50:
            continue
        diff = np.median(np.abs(x[loud] - y[loud]))
        if diff < best_diff and np.corrcoef(x[loud], y[loud])[0, 1] > 0.99:
            best, best_diff = lag, diff
    return best


def _abort_if_every_file_would_fail(e: Exception) -> None:
    msg = str(e).lower()
    if "out of memory" in msg:
        raise Abort("the GPU ran out of memory. Close other programs that use it, "
                    "then run the same command again.") from e
    if getattr(e, "errno", None) == errno.ENOSPC or "no space left" in msg or "not enough space" in msg:
        raise Abort(f"the disk is full ({e}). Free some space, then run the same command again.") from e


def _abort_if_model_damaged(e: Exception) -> None:
    # Demucs checks a model file's hash on first use, not when it loads, and a download that was
    # cut short leaves a file that fails that check for every track.
    if type(e).__name__ != "ModelLoadingError":
        return
    m = re.search(r"for file (.+?), expected", str(e))
    path = Path(m.group(1)).resolve() if m else None
    if path is None or not path.is_relative_to(model_dir().resolve()):
        raise Abort(f"cannot load the separation model: {e}") from e
    path.unlink(missing_ok=True)
    raise Abort(f"the model file {path} was incomplete, probably from an interrupted download, "
                "and has been deleted. Run the same command again to download it again.") from e


def _gone(path: str) -> bool:
    """Whether a file was deleted, as opposed to being on a drive that is not plugged in."""
    return not os.path.exists(path) and os.path.exists(os.path.splitdrive(path)[0] + os.sep)


def _lossy(path: str) -> bool:
    try:
        return audio.probe(path).bits == 0
    except audio.DecodeError:
        return False


def _same_path(a: str, b: str) -> bool:
    """Whether two spellings name one file, as Song.mp3 and song.mp3 do on Windows."""
    return os.path.normcase(a) == os.path.normcase(b)


def _lock(path: Path):
    """Lock path until the returned file is closed, or raise Abort if another run has it."""
    f = open(path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise Abort(f"another breakdig index run is using {path.parent}. "
                    "Wait for it to finish, then run this again.") from None
    return f


def _mono(path: str) -> np.ndarray:
    """The file as mono float32, read in blocks so a long mix never sits in memory as stereo."""
    mono = np.empty(sf.info(path).frames, dtype=np.float32)
    pos = 0
    for b in sf.blocks(path, blocksize=1 << 20, dtype="float32", always_2d=True):
        mono[pos: pos + len(b)] = b.mean(axis=1)
        pos += len(b)
    return mono[:pos]


def _peak(path: str) -> float:
    return max((float(np.abs(b).max()) for b in sf.blocks(path, blocksize=1 << 20, dtype="float32")),
               default=0.0)


class Indexer:
    def __init__(self, index: Index, model: str | None = None, shifts: int = 1):
        self.index = index
        self.model = model
        self.shifts = shifts
        self._sep: Separator | None = None
        self._beats: BeatTracker | None = None
        self._lock = _lock(index.home / "index.lock")
        index.sweep_profiles()
        # Scratch lives in the index home so a killed run's leftovers get swept next time.
        root = index.home / "tmp"
        root.mkdir(exist_ok=True)
        for d in root.glob("run-*"):  # holding the lock, so no other run is using them
            shutil.rmtree(d, ignore_errors=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="run-", dir=root))

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def load_models(self) -> None:
        """Load both models now, so a bad model or setup stops the run before the first file."""
        try:
            self.separator, self.beats
        except Exception as e:
            raise Abort(f"cannot load the models: {type(e).__name__}: {e}") from e

    @property
    def separator(self) -> Separator:
        if self._sep is None:
            kw = {"model": self.model} if self.model else {}
            self._sep = Separator(model_dir(), shifts=self.shifts, **kw)
        return self._sep

    @property
    def beats(self) -> BeatTracker:
        if self._beats is None:
            self._beats = BeatTracker()
        return self._beats

    def pending(self, paths: Iterable[str], retry_failed: bool = False) -> tuple[list[scan.Found], int]:
        """Files that still need work, and how many were already indexed.

        A row follows its audio: it takes the new path of a file that moved or swapped names
        with another, and is dropped when its path holds other known audio and its own is gone."""
        found = {}
        for p in paths:
            try:
                found[p] = self._found(p)
            except OSError as e:
                log.warning("cannot read %s: %s", p, e)
        todo, known, seen, dups, settled = [], 0, {}, [], []
        for p, f in found.items():
            if f.key in seen and not self._differs(f, seen[f.key]):
                known += 1  # a copy of a file earlier in the scan
                settled.append(p)
                continue
            row = self.index.by_key(f.key)
            if row is not None and row["path"] != p:
                if _same_path(row["path"], p) or self._key_at(row["path"], found) != row["key"]:
                    self.index.update(row["id"], path=p)  # moved, renamed, or swapped names
                    row = self.index.by_key(f.key)
                elif self._differs(f, row["path"]):
                    row = self.index.by_key(f.key)
            seen[f.key] = p
            if row is None or (retry_failed and row["status"] == "failed"):
                todo.append(f)
                continue
            known += 1
            settled.append(p)
            if row["path"] == p and (row["size"], row["mtime_ns"]) != (f.size, f.mtime_ns):
                self.index.update(row["id"], size=f.size, mtime_ns=f.mtime_ns)  # so it is not read again
            if row["status"] == "duplicate":
                dups.append(f)
        keys = {f.key for f in found.values()}
        for p in settled:
            for key in self.index.keys_at(p):
                if key not in keys:
                    self.index.forget(key)  # overwritten with a copy of another file
        # After the loop, so an original that only moved has had its path updated.
        redo = {f.path for f in todo}
        for f in dups:
            orig = self.index.track(self.index.by_key(f.key)["dup_of"])
            if orig is not None and _gone(orig["path"]):
                self.index.forget(orig["key"])
                orig = None
            if orig is None or orig["path"] in redo:
                # The copy that was indexed is gone or about to change, so this one is checked again.
                todo.append(f)
                known -= 1
        return todo, known

    def _key_at(self, path: str, found: dict[str, scan.Found]) -> str | None:
        """The key of what is at path now, or None if nothing is."""
        if path in found:
            return found[path].key
        try:
            return self._found(path).key
        except OSError:
            return None

    def _differs(self, f: scan.Found, other: str) -> bool:
        """For two files with one sampled key, whether they differ somewhere else, in which case f
        gets a key over the whole file. Edits of one track can share their start, middle and end."""
        try:
            st = os.stat(other)
        except OSError:
            st = None
        copy = {"path": f.path, "key": f.key, "duration": f.duration, "size": f.size,
                "mtime_ns": f.mtime_ns, "of": other} | ({"of_size": st.st_size, "of_mtime_ns": st.st_mtime_ns}
                                                       if st else {})
        known = self.index.copy_at(f.path)
        if st and known is not None and dict(known) == copy:
            return False  # neither file has changed since they were found to be the same
        full = scan.full_key(f.path)
        try:
            if full == scan.full_key(other):
                if st:
                    self.index.save_copy(copy)
                return False
        except OSError:
            pass
        f.key = full
        return True

    def _found(self, path: str) -> scan.Found:
        """The file's content key, reusing the stored one while its size and mtime are unchanged."""
        st = os.stat(path)
        row = self.index.by_path(path) or self.index.copy_at(path)
        if row is not None and (row["size"], row["mtime_ns"]) == (st.st_size, st.st_mtime_ns):
            return scan.Found(path, row["key"], st.st_size, row["duration"], st.st_mtime_ns)
        return scan.content_key(path)

    def index_file(self, f: scan.Found) -> str:
        """Index one file and commit it. Returns the stored status, "updated" if the file only
        had its tags or bytes changed, or "skipped" if it moved or changed since the scan."""
        t0 = time.perf_counter()
        try:
            if os.path.getsize(f.path) != f.size:
                return "skipped"
        except OSError:
            return "skipped"
        base = {"key": f.key, "path": f.path, **scan.read_tags(f.path), "duration": f.duration,
                "size": f.size, "mtime_ns": f.mtime_ns}
        try:
            status, extra, bars = self._analyse(f)
        except Skip as e:
            status, extra, bars = e.status, {**e.extra, "error": str(e)}, []
        if status == "updated":
            base.pop("duration")  # the stored one is the decoded length
            self.index.rekey(extra["dup_of"], base)
            return status
        replaces = extra.pop("replaces", None)
        extra["seconds"] = time.perf_counter() - t0
        self.index.save({**base, "status": status, **extra}, bars)
        if replaces:
            self.index.forget(replaces)
        return status

    def _analyse(self, f: scan.Found) -> tuple[str, dict, list[tuple]]:
        if Path(f.path).suffix.lower() in scan.DRM_EXTS:
            raise Skip("failed", "DRM-protected iTunes file (.m4p), cannot be decoded")
        wav = self.tmp / "mix.wav"
        gain = INPUT_GAIN
        try:
            probe = audio.probe(f.path)
            audio.decode_to_wav(f.path, str(wav), SAMPLE_RATE, gain)
            peak = _peak(str(wav))
            if peak > INPUT_GAIN:  # above full scale, as float files and loud MP3s can be
                gain *= INPUT_GAIN / peak
                audio.decode_to_wav(f.path, str(wav), SAMPLE_RATE, gain)
        except audio.DecodeError as e:
            _abort_if_every_file_would_fail(e)
            raise Skip("failed", f"unreadable or corrupt: {e}") from None
        gain_db = -20.0 * np.log10(gain)
        info = sf.info(str(wav))
        duration = info.frames / info.samplerate
        extra = {"duration": duration, "sample_rate": probe.sample_rate, "channels": probe.channels}
        hop = int(round(profile.HOP_SECONDS * SAMPLE_RATE))

        mix_env = profile.file_envelope(str(wav), hop)
        mix_db = profile.to_db(mix_env, gain_db)
        dup, lag = self._find_duplicate(duration, mix_db)
        if dup is not None:
            elsewhere = not _same_path(dup["path"], f.path) and not _gone(dup["path"])
            if elsewhere and not (probe.bits and _lossy(dup["path"])):
                raise Skip("duplicate", f"same audio as {dup['path']}", **extra, dup_of=dup["id"])
            # The same file, retagged or re-saved, and maybe renamed too. Its analysis still holds
            # unless the audio moved in time.
            if lag == 0 and not elsewhere:
                return "updated", {"dup_of": dup["id"]}, []
            # A lossless copy of a lossy original is analysed on its own bars and takes its place,
            # so sections are cut from it. The lossy file becomes its duplicate on the next run.
            extra["replaces"] = dup["key"]

        try:
            mono = _mono(str(wav))
            beats, downbeats = self.beats(mono, SAMPLE_RATE)
            del mono
        except (RuntimeError, ValueError, MemoryError) as e:
            _abort_if_every_file_would_fail(e)
            raise Skip("failed", f"beat tracking failed: {type(e).__name__}: {e}", **extra) from None
        extra["beats"] = beats.astype(np.float64).tobytes()
        if len(beats) > 1:
            extra["bpm"] = 60.0 / float(np.median(np.diff(beats)))
        if len(downbeats) < MIN_DOWNBEATS:
            raise Skip("no_grid", f"only {len(downbeats)} downbeats found, need {MIN_DOWNBEATS}", **extra)

        chunk = float(CHUNK_SECONDS) if duration > LONG_FILE_SECONDS else None
        stem_dir = self.tmp / "stems"
        stem_dir.mkdir(exist_ok=True)
        try:
            stems = self.separator.separate(str(wav), str(stem_dir), chunk)
            envs = {s: profile.file_envelope(stems[s], hop) for s in STEMS}
        except Exception as e:
            _abort_if_every_file_would_fail(e)
            _abort_if_model_damaged(e)
            raise Skip("failed", f"separation failed: {type(e).__name__}: {e}", **extra) from None
        finally:
            shutil.rmtree(stem_dir, ignore_errors=True)
        envs["mix"] = mix_env
        rows = profile.profile(envs, downbeats, profile.HOP_SECONDS, gain_db)
        # Nothing reads the stem envelopes yet; they are kept so bars can be rebuilt without separating.
        np.savez_compressed(
            self.index.profile_path(f.key), hop=profile.HOP_SECONDS, beats=beats, downbeats=downbeats,
            **{k: profile.to_db(v, gain_db).astype(np.float32) for k, v in envs.items()})
        extra["chunk_seconds"] = chunk
        return "ok", extra, rows

    def _find_duplicate(self, duration: float, mix_db: np.ndarray):
        """The indexed track this is the same recording as, and the offset, or (None, None)."""
        if len(mix_db) < MIN_FRAMES + 2 * MAX_LAG:
            return None, None  # too short to tell, so skip loading every profile in a sample pack
        for row in self.index.candidates_near(duration):
            p = self.index.profile_path(row["key"])
            if not p.exists():
                continue
            with np.load(p) as z:
                lag = match_lag(mix_db, z["mix"])
            if lag is not None:
                return row, lag
        return None, None
