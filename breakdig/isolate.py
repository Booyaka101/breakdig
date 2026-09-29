"""Separate one section at export time and keep only some of its stems."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .db import STEMS
from .separate import SAMPLE_RATE, Separator

# Demucs needs context either side of the cut, or the edges come out smeared.
PAD_SECONDS = 3.0
# As when indexing, so the separator never peak-normalises a stem (see separate.py).
MAX_INPUT_PEAK = 0.5


def stems_name(keep) -> str:
    """'drums', 'bass+drums' and so on, in STEMS order."""
    return "+".join(s for s in STEMS if s in keep)


def isolate(x: np.ndarray, sample_rate: int, keep, separator: Separator) -> np.ndarray:
    """The sum of the kept stems of x, (frames, channels) at sample_rate: the same length, mono
    if x is and stereo otherwise."""
    stereo = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x[:, :2]
    y = soxr.resample(stereo, sample_rate, SAMPLE_RATE, quality="VHQ") if sample_rate != SAMPLE_RATE else stereo
    peak = float(np.abs(y).max(initial=0.0))
    gain = MAX_INPUT_PEAK / peak if peak > MAX_INPUT_PEAK else 1.0
    with tempfile.TemporaryDirectory(prefix="breakdig-") as tmp:
        wav = Path(tmp) / "section.wav"
        sf.write(wav, (y * gain).astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        files = separator.separate(str(wav), tmp)
        kept = sum(sf.read(files[s], dtype="float32", always_2d=True)[0] for s in STEMS if s in keep)
    kept = np.asarray(kept, dtype=np.float32) / gain
    if sample_rate != SAMPLE_RATE:
        kept = soxr.resample(kept, SAMPLE_RATE, sample_rate, quality="VHQ")
    kept = _fit(kept, len(x))
    return kept.mean(axis=1, keepdims=True) if x.shape[1] == 1 else kept


def _fit(y: np.ndarray, frames: int) -> np.ndarray:
    if len(y) >= frames:
        return y[:frames]
    return np.pad(y, ((0, frames - len(y)), (0, 0)))
