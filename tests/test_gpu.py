"""The worked example through the real models: Demucs separation and beat_this."""

import numpy as np
import pytest
import soundfile as sf
from conftest import needs_ffmpeg

from breakdig import indexer
from breakdig.db import Index, model_dir
from breakdig.export import cut
from breakdig.indexer import Indexer
from breakdig.query import PRESETS, find
from breakdig.separate import Separator

pytestmark = [pytest.mark.gpu, needs_ffmpeg]


def test_fixture_drums_only_is_bars_5_to_8(tmp_path, fixture_audio):
    with Index(tmp_path / "home") as index:
        ix = Indexer(index)
        try:
            assert ix.separator.device.startswith("cuda") or ix.separator.device == "gpu"
            todo, _ = ix.pending([str(fixture_audio["mix"])])
            assert ix.index_file(todo[0]) == "ok"
        finally:
            ix.close()
        (s,) = find(index, PRESETS["drums"])
        assert (s.first_bar, s.last_bar, s.bars) == (5, 8, 4)
        assert s.start == pytest.approx(8.0, abs=0.02)
        assert s.end == pytest.approx(16.0, abs=0.02)
        assert s.bpm == pytest.approx(120.0, abs=0.5)
        # Every downbeat, not just the section edges, should be on the grid.
        starts = index.bars_for()[s.track_id][:, 0]
        assert abs(starts - [2.0 * i for i in range(len(starts))]).max() < 0.02


def test_chunked_separation_flags_seams(tmp_path, fixture_audio, monkeypatch):
    monkeypatch.setattr(indexer, "LONG_FILE_SECONDS", 20)
    monkeypatch.setattr(indexer, "CHUNK_SECONDS", 12)
    monkeypatch.chdir(tmp_path)  # a relative home once broke chunked output paths
    with Index("home") as index:
        ix = Indexer(index)
        try:
            todo, _ = ix.pending([str(fixture_audio["mix"])])
            assert ix.index_file(todo[0]) == "ok"
        finally:
            ix.close()
        assert index.track(1)["chunk_seconds"] == 12
        (s,) = find(index, PRESETS["drums"])
        assert (s.first_bar, s.last_bar, s.seam) == (5, 8, True)  # the seam at 12 s is inside
        assert find(index, PRESETS["drums"], skip_seams=True) == []


def test_separation_is_repeatable(tmp_path, fixture_audio):
    sep = Separator(model_dir())
    runs = [sep.separate(str(fixture_audio["mix"]), str(tmp_path / str(i))) for i in range(2)]
    for stem in runs[0]:
        a, b = (sf.read(r[stem])[0] for r in runs)
        assert np.abs(a - b).max() < 1e-4, stem


def test_isolated_drums_drop_the_bass_and_pad(fixture_audio):
    # Bars 1-4 have bass and a pad over the drums.
    sep = Separator(model_dir())
    true = sf.read(fixture_audio["drums"], dtype="float32")[0]

    def off_by(clip):
        x = clip.audio[:, 0]
        a = round(clip.start * clip.sample_rate)
        return np.sum((x - true[a: a + len(x)]) ** 2) / np.sum(true[a: a + len(x)] ** 2)
    mix = str(fixture_audio["mix"])
    assert off_by(cut(mix, 0.0, 8.0, {"drums"}, sep)) < 0.05 < 1 < off_by(cut(mix, 0.0, 8.0))
