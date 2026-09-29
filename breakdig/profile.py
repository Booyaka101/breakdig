"""Loudness envelopes and per-bar stem levels."""

from collections.abc import Iterable

import numpy as np
import soundfile as sf

from .db import STEMS

HOP_SECONDS = 0.05
FLOOR_DB = -120.0
TAIL_DB = -60.0  # what counts as sound in the bar after the last downbeat


def power_envelope(blocks: Iterable[np.ndarray], hop: int) -> np.ndarray:
    """Mean power per hop-sized frame, over all channels. A trailing partial frame is dropped."""
    out, carry = [], np.zeros((0,), dtype=np.float64)
    for b in blocks:
        b = np.asarray(b, dtype=np.float64)
        p = (b ** 2).mean(axis=1) if b.ndim == 2 else b ** 2
        p = np.concatenate([carry, p])
        n = len(p) // hop
        if n:
            out.append(p[: n * hop].reshape(n, hop).mean(axis=1))
        carry = p[n * hop:]
    return np.concatenate(out) if out else np.zeros(0)


def file_envelope(path: str, hop: int) -> np.ndarray:
    return power_envelope(sf.blocks(path, blocksize=hop * 2000, dtype="float32", always_2d=True), hop)


def to_db(power: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    return np.maximum(10.0 * np.log10(power + 1e-12) + gain_db, FLOOR_DB)


def bars_from_downbeats(downbeats: np.ndarray, mix_db: np.ndarray, hop_seconds: float) -> np.ndarray:
    """(n, 2) start/end pairs between consecutive downbeats, plus one more bar after the last
    if the mix sounds through nearly all of it."""
    d = np.asarray(downbeats, dtype=np.float64)
    if len(d) < 2:
        return np.zeros((0, 2))
    bars = np.stack([d[:-1], d[1:]], axis=1)
    end = d[-1] + float(np.median(np.diff(d)[-4:]))
    tail = mix_db[int(round(d[-1] / hop_seconds)):int(round(end / hop_seconds))]
    # The envelope drops a trailing partial frame, so a bar ending there may be one frame short.
    if len(tail) >= (end - d[-1]) / hop_seconds - 1 and np.mean(tail > TAIL_DB) >= 0.9:
        return np.vstack([bars, [d[-1], end]])
    return bars


def bar_levels(envelope_power: np.ndarray, bars: np.ndarray, hop_seconds: float,
               gain_db: float = 0.0) -> np.ndarray:
    """RMS level in dB of each bar, from frames whose start lies inside it."""
    edges = np.round(bars / hop_seconds).astype(np.int64)
    edges = np.clip(edges, 0, len(envelope_power))
    csum = np.concatenate([[0.0], np.cumsum(envelope_power)])
    n = np.maximum(edges[:, 1] - edges[:, 0], 1)
    mean = (csum[edges[:, 1]] - csum[edges[:, 0]]) / n
    return to_db(mean, gain_db)


def profile(envelopes: dict[str, np.ndarray], downbeats: np.ndarray, hop_seconds: float,
            gain_db: float = 0.0) -> list[tuple]:
    """Rows of (idx, start, end, drums, bass, vocals, other, mix) in dB."""
    bars = bars_from_downbeats(downbeats, to_db(envelopes["mix"], gain_db), hop_seconds)
    levels = [bar_levels(envelopes[s], bars, hop_seconds, gain_db) for s in (*STEMS, "mix")]
    return [(i, float(b[0]), float(b[1]), *(float(l[i]) for l in levels))
            for i, b in enumerate(bars)]
