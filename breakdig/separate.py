"""Four-stem separation with audio-separator (Demucs)."""

import contextlib
import io
import logging
import os
import random
from pathlib import Path

from .db import STEMS

DEFAULT_MODEL = "htdemucs_ft.yaml"
SAMPLE_RATE = 44100
# audio-separator concatenates chunks with no crossfade, so seams need flagging.
LONG_FILE_SECONDS = 20 * 60
CHUNK_SECONDS = 600


class Separator:
    def __init__(self, model_dir: str | Path, model: str = DEFAULT_MODEL, shifts: int = 1):
        from audio_separator.separator import Separator as _Sep

        self.model = model
        # onnxruntime prints a line about CUDA DLLs to stdout while this loads.
        with contextlib.redirect_stdout(io.StringIO()):
            self._sep = _Sep(
                log_level=logging.ERROR,
                model_file_dir=str(model_dir),
                use_soundfile=True,
                # Stems are peak-normalised one by one above this threshold, which
                # would skew stem-vs-mix levels. The indexer feeds a -6 dB copy so
                # nothing gets near 1.0 and this never triggers.
                normalization_threshold=1.0,
                sample_rate=SAMPLE_RATE,
                demucs_params={"segment_size": "Default", "shifts": shifts, "overlap": 0.25,
                               "segments_enabled": True},
            )
            self._sep.load_model(model)

    @property
    def device(self) -> str:
        return str(self._sep.torch_device)

    def separate(self, wav_path: str, out_dir: str, chunk_seconds: float | None = None) -> dict[str, str]:
        """Separate wav_path into out_dir and return {stem: path} for the four stems."""
        # Chunked runs return paths that already include a relative out_dir, unchunked ones do not.
        out_dir = os.path.abspath(out_dir)
        self._sep.chunk_duration = chunk_seconds
        self._sep.output_dir = out_dir
        if self._sep.model_instance is not None:
            self._sep.model_instance.output_dir = out_dir
        names = {s.capitalize(): s for s in STEMS}
        # Demucs separates at a random time offset, which moves bleed levels by up to
        # 20 dB on real records; a fixed seed makes re-indexing give the same bars.
        random.seed(0)
        # Demucs hardcodes a tqdm bar per file; keep it off the caller's console.
        with contextlib.redirect_stderr(io.StringIO()):
            files = self._sep.separate(wav_path, custom_output_names=names)
        found = {}
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(out_dir, f)
            stem = Path(p).stem.lower()
            if stem in STEMS:
                found[stem] = p
        missing = [s for s in STEMS if s not in found]
        if missing:
            raise RuntimeError(f"model {self.model} did not produce stems: {', '.join(missing)}")
        return found
