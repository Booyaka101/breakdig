import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
import make_fixture  # noqa: E402

from breakdig.db import STEMS, Index  # noqa: E402
from breakdig.indexer import INPUT_GAIN, Indexer  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixture_audio(tmp_path_factory) -> dict[str, Path]:
    """The worked-example track and its true stems."""
    return make_fixture.make(tmp_path_factory.mktemp("fixture"))


class TrueStems:
    """Stands in for Demucs by handing back the fixture's real stems, scaled like
    the separator's -6 dB input so the indexer's gain correction still applies."""

    def __init__(self, stems: dict[str, Path]):
        self.stems = stems

    def separate(self, wav_path, out_dir, chunk_seconds=None):
        out = {}
        for name in ("drums", "bass", "vocals", "other"):
            x, sr = sf.read(self.stems[name], dtype="float32")
            p = Path(out_dir) / f"{name}.wav"
            sf.write(p, np.column_stack([x, x]) * INPUT_GAIN, sr, subtype="FLOAT")
            out[name] = str(p)
        return out


class InputAsDrums:
    """Stands in for Demucs at export time: the input comes back as the drums and the rest is
    silence, so isolated drums should be the plain cut again. Keeps each input's peak."""

    def __init__(self):
        self.peaks = []

    def separate(self, wav_path, out_dir, chunk_seconds=None):
        x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        self.peaks.append(float(np.abs(x).max()))
        out = {}
        for name in STEMS:
            out[name] = str(Path(out_dir) / f"{name}.wav")
            sf.write(out[name], x if name == "drums" else np.zeros_like(x), sr, subtype="FLOAT")
        return out


class GridBeats:
    """Stands in for beat_this with the fixture's known grid: 120 BPM, bar 1 at 0 s."""

    def __init__(self, bars=make_fixture.BARS):
        self.bars = bars

    def __call__(self, mono, sr):
        beats = np.arange(self.bars * 4 + 1) * make_fixture.BEAT
        return beats, beats[::4]


@pytest.fixture
def cpu_indexer(tmp_path, fixture_audio):
    """An Indexer over a fresh index, with stubbed separation and beat tracking."""
    index = Index(tmp_path / "home")
    ix = Indexer(index)
    ix._sep = TrueStems(fixture_audio)
    ix._beats = GridBeats()
    yield ix
    ix.close()
    index.close()
