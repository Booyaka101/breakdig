"""Beat and downbeat tracking with beat_this."""

import warnings

import numpy as np

# beat_this pulls in rotary_embedding_torch, which warns about a torch API rename on import.
warnings.filterwarnings("ignore", category=FutureWarning, module="rotary_embedding_torch")

MIN_DOWNBEATS = 8


class BeatTracker:
    def __init__(self):
        import torch
        from beat_this.inference import Audio2Beats

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        # Audio2Beats is File2Beats minus the file loading; the mix is already decoded.
        self._model = Audio2Beats(checkpoint_path="final0", device=self.device, dbn=False)

    def __call__(self, mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        beats, downbeats = self._model(mono, sr)
        beats = refine(np.asarray(beats, dtype=np.float64), mono, sr)
        downbeats = np.asarray(downbeats, dtype=np.float64)
        # Keep downbeats locked onto the refined beat they came from.
        if len(beats) and len(downbeats):
            nearest = np.abs(downbeats[:, None] - beats[None, :]).argmin(axis=1)
            downbeats = beats[nearest]
        return beats, np.unique(downbeats)


def refine(times: np.ndarray, mono: np.ndarray, sr: int, radius: float = 0.025) -> np.ndarray:
    """Move each beat onto the sharpest energy rise within +-radius seconds.

    beat_this works on 20 ms frames, which is enough for tempo but audible as a
    flam or a clipped attack when you cut a loop on it. A beat with no clear
    onset nearby is left where it was."""
    hop = max(1, sr // 1000)
    n = len(mono) // hop
    if n == 0 or len(times) == 0:
        return times
    frames = mono[: n * hop].reshape(n, hop).astype(np.float64)
    energy = np.log10((frames ** 2).mean(axis=1) + 1e-10)
    # Rise over 3 ms rather than 1 ms, so a single noisy frame can't win. Before the
    # file counts as silence, so a hit on the very first sample is still an onset.
    padded = np.concatenate([np.full(3, -10.0), energy])
    rise = padded[3:] - padded[:-3]
    out = times.copy()
    fps = sr / hop
    r = int(round(radius * fps))
    for i, t in enumerate(times):
        c = int(round(t * fps))
        lo, hi = max(0, c - r), min(n, c + r + 1)
        if hi - lo < 3:
            continue
        j = lo + int(np.argmax(rise[lo:hi]))
        # A rise of 0.5 in log10 energy is ~5 dB inside 3 ms: a real attack.
        if rise[j] >= 0.5:
            # The rise peaks at the end of the 3 ms window; the attack starts at its beginning.
            out[i] = max(0, j - 3) * hop / sr
    return out
