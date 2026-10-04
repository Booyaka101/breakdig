import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import needs_ffmpeg, run  # noqa: F401
from mutagen.id3 import COMM, ID3, TIT2, TPE1
from mutagen.wave import WAVE

from breakdig import __version__, drill
from breakdig.db import Index
from breakdig.drill import (Ladder, free_path, mine, origin_text, parse_ladder, plan, read_origin,
                            render, total_seconds)

pytestmark = needs_ffmpeg

LADDER = Ladder("70-100", (70, 75, 80, 85, 90, 95, 100))
KEY = "a" * 40  # the content key shape the comment tag's parser expects


def _tagged_copy(src, dest, artist, title, seconds=None):
    """A tagged WAV copy of a fixture file, so the index row has names to check."""
    data, sr = sf.read(src, dtype="float32")
    if seconds:
        data = data[: round(seconds * sr)]
    sf.write(dest, data, sr)
    w = WAVE(dest)
    w.add_tags()
    w.tags.add(TPE1(encoding=3, text=artist))
    w.tags.add(TIT2(encoding=3, text=title))
    w.save()
    return dest


def _drill_tags(path: Path, source: str) -> None:
    """Give an mp3 the comment tag a breakdig drill would carry, so the origin parser
    and free_path have something real to read."""
    tags = ID3()
    tags.add(COMM(encoding=3, lang="eng", desc="breakdig",
                  text=origin_text(source, "70-100", 1, 4, 1, "", "", KEY)))
    tags.save(path)


@pytest.fixture
def songs(tmp_path, fixture_audio):
    """A setlist folder with two tagged songs: the 32 s fixture mix and its first 16 s."""
    folder = tmp_path / "setlist"
    folder.mkdir()
    return {"dir": folder,
            "one": _tagged_copy(fixture_audio["mix"], folder / "song one.wav",
                                "Fixture Band", "Drill Me"),
            "two": _tagged_copy(fixture_audio["mix"], folder / "song two.wav",
                                "Other Artist", "Short One", seconds=16.0)}


def test_parse_ladder():
    assert parse_ladder("70-100:5") == LADDER
    assert parse_ladder("70-100") == LADDER
    assert parse_ladder("70-95:10") == Ladder("70-95", (70, 80, 90, 95))
    for bad in ("40-100", "100-70", "70-105", "70-70", "70-100:0", "70-100:21", "seventy", "70"):
        with pytest.raises(ValueError):
            parse_ladder(bad)
    with pytest.raises(ValueError, match="50 <= start < end <= 100"):
        parse_ladder("40-100")
    with pytest.raises(ValueError, match="step must be 1 to 20"):
        parse_ladder("70-100:0")


def test_planner_worked_example():
    """240 s at 120 BPM with the defaults: the brief's numbers, rung for rung."""
    rungs = plan(240.0, 120.0, LADDER)
    assert [r.percent for r in rungs] == [70, 75, 80, 85, 90, 95, 100]
    assert [round(r.pass_seconds, 1) for r in rungs] == [342.9, 320.0, 300.0, 282.4, 266.7, 252.6, 240.0]
    assert rungs[0].count_in_seconds == pytest.approx(4 * (60 / 120) / 0.7)
    assert rungs[0].gap_seconds == pytest.approx(4 * (60 / 120) / 0.7)
    assert rungs[1].offset == pytest.approx(2.857143 + 342.857143 + 2.857143, abs=1e-4)
    assert plan(240.0, 120.0, LADDER) == rungs  # pure and deterministic
    total = total_seconds(rungs)
    assert total == pytest.approx(2035.9, abs=0.1)
    assert total * 192000 / 8 / 1e6 == pytest.approx(48.9, abs=0.1)  # ~49 MB at 192 kbps


def test_planner_passes_and_layout():
    rungs = plan(60.0, 120.0, Ladder("70-100", (70, 100)), passes=2)
    beat = 60 / 120
    rung_beat = beat / 0.7
    assert rungs[1].offset == pytest.approx(
        4 * rung_beat + 2 * (60 / 0.7 + 4 * rung_beat), abs=1e-4)  # count-in, pass, gap, pass, gap
    total = total_seconds(rungs, 2)
    assert total == pytest.approx(
        rungs[1].offset + 4 * beat + 2 * (60.0 + 4 * beat) - 4 * beat, abs=1e-4)
    blocks = list(render(np.ones((60 * 44100, 2), dtype=np.float32), 44100, 44100, rungs, 2, 4))
    assert len(blocks) == 9  # count-in, pass, gap, pass, gap, count-in, pass, gap, pass
    assert sum(len(b) for b in blocks) == round(total * 44100)
    with pytest.raises(ValueError):
        plan(0.0, 120.0, LADDER)
    with pytest.raises(ValueError):
        plan(60.0, 0.0, LADDER)


def test_render_clicks_gaps_and_fades():
    """Two rungs at 44.1 kHz: the clicks land on whole frames, the gap is silence, the
    passes fade in and out, and the whole thing is exactly plan() long."""
    rate = 44100
    mix = (0.1 * np.sin(2 * np.pi * 440 * np.arange(4 * rate) / rate)).astype(np.float32)
    mix = np.column_stack([mix, mix])
    rungs = plan(4.0, 120.0, Ladder("70-100", (70, 100)), 1, 4, 1)
    blocks = list(render(mix, rate, rate, rungs, 1, 4))
    assert [b.shape for b in blocks] == [
        (round(rungs[0].count_in_seconds * rate), 2), (round(rungs[0].pass_seconds * rate), 2),
        (round(rungs[0].gap_seconds * rate), 2), (round(rungs[1].count_in_seconds * rate), 2),
        (round(rungs[1].pass_seconds * rate), 2)]
    ci = blocks[0]
    beat = round((60 / 120) / 0.7 * rate)  # 31500 frames: sample-exact at the rung's tempo
    assert beat == 31500
    for k in range(4):
        assert abs(ci[k * beat: k * beat + 44]).max() > 0.3  # a click starts on that frame
        assert abs(ci[k * beat + 400]).max() < abs(ci[k * beat: k * beat + 44]).max()
    assert abs(ci[:beat]).max() > abs(ci[beat: 2 * beat]).max() + 0.1  # beat 1 accented
    assert blocks[2].max() == 0.0 and blocks[2].min() == 0.0  # the gap is silence
    ci2 = blocks[3]
    assert abs(ci2[:44]).max() > 0.3 and abs(ci2[22050: 22094]).max() > 0.3
    fade = blocks[1][:, 0]
    assert abs(fade[0]) < 0.02
    # The 5 ms fade: the first half of the fade window is far quieter than the second half,
    # whichever way the waveform happens to swing there.
    assert np.abs(fade[:110]).max() < 0.6 * np.abs(fade[110:220]).max()
    out = np.concatenate([b[:, 0] for b in blocks])
    assert len(out) == round(total_seconds(rungs) * rate)


def test_render_is_deterministic_and_mono_stays_mono():
    rng = np.random.default_rng(0)
    mix = (rng.standard_normal((44100, 1))).astype(np.float32) * 0.2
    rungs = plan(1.0, 120.0, Ladder("70-100", (70, 100)), 1, 4, 1)
    a = np.concatenate([b[:, 0] for b in render(mix, 44100, 44100, rungs, 1, 4)])
    b = np.concatenate([b[:, 0] for b in render(mix, 44100, 44100, rungs, 1, 4)])
    assert np.array_equal(a, b)
    assert all(block.shape[1] == 1 for block in render(mix, 44100, 44100, rungs, 1, 4))


def test_stretch_round_trip_ratio():
    """python-stretch's own output length is within 1% of what plan() promised, before
    the renderer trims or pads."""
    import python_stretch

    rng = np.random.default_rng(0)
    x = np.ascontiguousarray((rng.standard_normal((2, 44100 * 10)) * 0.1).astype(np.float32))
    for tf in (0.7, 0.85, 1.0):
        stretch = python_stretch.Signalsmith.Stretch()
        stretch.preset(2, 44100)
        stretch.timeFactor = tf
        y = stretch.process(x)
        want = x.shape[1] / tf
        assert abs(y.shape[1] - want) / want < 0.01


def test_drill_name_and_paths(tmp_path):
    from breakdig.export import MAX_NAME

    assert drill.drill_name("Fixture Band", "Drill Me", "70-100", "mp3") == \
        "Fixture Band - Drill Me - drill 70-100.mp3"
    long = drill.drill_name("B" * 300, "x", "70-100", "mp3")
    assert len(long) <= MAX_NAME and long.endswith(" - drill 70-100.mp3")
    assert drill.drill_name('AC/DC: "Live"', "What? <Yes|No>*", "70-100", "wav") == \
        'AC_DC_ _Live_ - What_ _Yes_No__ - drill 70-100.wav'


def test_free_path_overwrites_ours_and_numbers_others(tmp_path):
    dest = tmp_path / "A - B - drill 70-100.mp3"
    dest.write_bytes(b"not really an mp3, but the tags are what count")
    _drill_tags(dest, "D:\\lib\\a.wav")
    ours = mine("D:\\lib\\a.wav", KEY, "70-100", 1, 4, 1, "", "")
    assert free_path(tmp_path, dest.name, ours) == dest  # ours: overwrite in place
    theirs = mine("D:\\lib\\b.wav", "b" * 40, "70-100", 1, 4, 1, "", "")
    assert free_path(tmp_path, dest.name, theirs) == tmp_path / "A - B - drill 70-100 (2).mp3"
    (tmp_path / "C - D - drill 70-100.mp3").write_bytes(b"no tags")
    assert free_path(tmp_path, "C - D - drill 70-100.mp3", ours) == \
        tmp_path / "C - D - drill 70-100 (2).mp3"


def test_origin_round_trip(tmp_path):
    path = tmp_path / "x.mp3"
    _drill_tags(path, "D:\\lib\\a.wav")
    source, label, passes, count_in, gap_bars, bars, stems, key, version = read_origin(path)
    assert (source, label, passes, count_in, gap_bars, bars, stems) == \
        ("D:\\lib\\a.wav", "70-100", "1", "4", "1", None, None)
    assert key == KEY and version == __version__


def test_cli_drill_end_to_end_mp3(capsys, tmp_path, songs, stub_models):
    home = str(tmp_path / "home")
    code, out, err = run(capsys, "--home", home, "drill", str(songs["dir"]),
                         "--out", str(tmp_path / "out"))
    assert code == 0, err
    assert "0 already indexed, 2 to do" in out  # the resumable indexer ran first
    assert "Fixture Band - Drill Me  120.0 BPM" in out
    assert "  70%" in out and "84.0 BPM" in out and "0:46" in out
    assert "100%" in out and "120.0 BPM" in out and "0:32" in out
    assert "2 file(s)" in out
    dest = tmp_path / "out" / "Fixture Band - Drill Me - drill 70-100.mp3"
    other = tmp_path / "out" / "Other Artist - Short One - drill 70-100.mp3"
    assert dest.exists() and other.exists()

    rows = list(csv.DictReader(open(tmp_path / "out" / "drills.csv", encoding="utf-8-sig")))
    assert [r["file"] for r in rows] == [dest.name, other.name]
    assert rows[0]["rungs"] == "70 75 80 85 90 95 100" and rows[0]["offset"] == "0.0"
    assert rows[0]["source"] == str(songs["one"])

    seconds = float(rows[0]["seconds"])
    data, sr = sf.read(dest, always_2d=True)
    assert abs(len(data) / sr - seconds) / seconds < 0.02  # end to end, decoded, within 2%
    assert data.shape[1] == sf.info(songs["one"]).channels  # mono stays mono, stereo stays stereo

    tags = ID3(dest)
    assert str(tags["TIT2"]) == "Drill Me - drill 70-100"
    assert str(tags["TPE1"]) == "Fixture Band"
    assert abs(int(str(tags["TLEN"])) - seconds * 1000) < 200
    comment = str(tags["COMM:breakdig:eng"])
    assert str(songs["one"]) in comment and "drill 70-100" in comment
    assert f"breakdig {__version__}" in comment

    # A rerun overwrites its own outputs deterministically: same names, no (2) copies.
    code, out, _ = run(capsys, "--home", home, "drill", str(songs["dir"]),
                       "--out", str(tmp_path / "out"))
    assert code == 0 and "2 already indexed, 0 to do" in out
    assert not list((tmp_path / "out").glob("*(2)*"))
    assert len(list((tmp_path / "out").glob("*.mp3"))) == 2


def test_cli_drill_out_folder_is_not_drilled(capsys, tmp_path, songs, stub_models):
    """A --out folder inside the library must not be indexed as songs on the next run."""
    out = songs["dir"] / "drills"
    run(capsys, "drill", str(songs["dir"]), "--home", str(tmp_path / "home"),
        "--out", str(out), "--ladder", "70-100:20")
    assert len(list(out.glob("*.mp3"))) == 2
    code, out_text, err = run(capsys, "drill", str(songs["dir"]), "--home", str(tmp_path / "home"),
                              "--out", str(out), "--ladder", "70-100:20")
    assert code == 0
    assert "2 already indexed, 0 to do" in out_text  # the drills themselves were not indexed
    assert len(list(out.glob("*.mp3"))) == 2 and not list(out.glob("*(2)*"))


def test_cli_drill_wav_cues(capsys, tmp_path, songs, stub_models):
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--format", "wav",
                         "--ladder", "70-100:20", "--passes", "2")
    assert code == 0, err
    dest = tmp_path / "out" / "Fixture Band - Drill Me - drill 70-100.wav"
    labels = cue_labels(dest)
    assert labels == ["70%", "70% x2", "90%", "90% x2", "100%", "100% x2"]
    # The 70% cue sits at 0, where its count-in starts; the 90% cue one drill-run later.
    frames = cue_points(dest)
    rate = sf.info(dest).samplerate
    rung = plan(32.0, 120.0, Ladder("70-100", (70, 90, 100)), 2, 4, 1)
    assert frames[0] == 0
    assert frames[2] == round(rung[1].offset * rate)


def cue_points(path) -> list[int]:
    data = open(path, "rb").read()
    i, cues = 12, []
    while i + 8 <= len(data):
        fourcc, size = data[i: i + 4], struct.unpack("<I", data[i + 4: i + 8])[0]
        if fourcc == b"cue ":
            n = struct.unpack("<I", data[i + 8: i + 12])[0]
            cues = [struct.unpack("<II4sIII", data[i + 12 + 24 * k: i + 36 + 24 * k])[5]
                    for k in range(n)]
        i += 8 + size + (size % 2)
    return cues


def cue_labels(path) -> list[str]:
    """The 'labl' texts of a RIFF WAV's cue chunk, in cue order."""
    data = open(path, "rb").read()
    i, labels = 12, []
    while i + 8 <= len(data):
        fourcc, size = data[i: i + 4], struct.unpack("<I", data[i + 4: i + 8])[0]
        if fourcc == b"LIST" and data[i + 8: i + 12] == b"adtl":
            j = i + 12
            while j + 8 <= i + 8 + size:
                sub, sub_size = data[j: j + 4], struct.unpack("<I", data[j + 4: j + 8])[0]
                if sub == b"labl":
                    labels.append(data[j + 12: j + 8 + sub_size].rstrip(b"\0").decode())
                j += 8 + sub_size + (sub_size % 2)
        i += 8 + size + (size % 2)
    return labels


def test_cli_drill_json_and_wav_format(capsys, tmp_path, songs, stub_models):
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--format", "wav", "--json")
    assert code == 0, err
    (manifest,) = json.loads(out)
    assert manifest["artist"] == "Fixture Band" and manifest["ladder"] == "70-100"
    assert manifest["source"] == str(songs["one"])
    assert [r["percent"] for r in manifest["rungs"]] == [70, 75, 80, 85, 90, 95, 100]
    assert manifest["rungs"][0]["bpm"] == 84.0
    dest = tmp_path / "out" / "Fixture Band - Drill Me - drill 70-100.wav"
    assert manifest["path"] == str(dest)
    info = sf.info(dest)
    assert info.samplerate == 44100 and info.subtype == "PCM_16"  # 16-bit source stays 16-bit
    assert abs(info.frames / info.samplerate - manifest["seconds"]) / manifest["seconds"] < 0.02
    wav_tags = WAVE(dest).tags
    assert str(wav_tags["TIT2"]) == "Drill Me - drill 70-100"
    # A cue on each rung's count-in, labelled with its speed, so editors can jump between them.
    labels = cue_labels(dest)
    assert labels == ["70%", "75%", "80%", "85%", "90%", "95%", "100%"]


def test_cli_drill_bars(capsys, tmp_path, songs, stub_models):
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--bars", "9-12", "--json")
    assert code == 0, err
    (manifest,) = json.loads(out)
    assert manifest["bars"] == "9-12"
    expected = total_seconds(plan(8.0, 120.0, LADDER))  # bars 9-12 are 8.000s to 16.000s
    assert manifest["seconds"] == pytest.approx(expected, abs=0.1)
    data, sr = sf.read(manifest["path"])
    assert abs(len(data) / sr - expected) / expected < 0.02

    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--bars", "9-20", "--json")
    assert code == 1 and "has bars 1-16, not 9-20" in err and json.loads(out) == []


def test_cli_drill_no_grid(capsys, tmp_path, songs, stub_models):
    home = str(tmp_path / "home")
    run(capsys, "--home", home, "index", str(songs["dir"]))
    with Index(home) as index:
        index.update(1, status="no_grid")
        index.update(2, status="no_grid")
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", home,
                         "--out", str(tmp_path / "out"), "--json")
    assert code == 0, err  # a track without a grid still drills the whole song
    (manifest,) = json.loads(out)
    assert manifest["seconds"] == pytest.approx(
        total_seconds(plan(32.0, 120.0, LADDER)), abs=0.1)
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", home,
                         "--out", str(tmp_path / "out"), "--bars", "1-4", "--json")
    assert code == 1 and "no beat grid" in err and "--bars does not apply" in err


def test_cli_drill_combined(capsys, tmp_path, songs, stub_models):
    code, out, err = run(capsys, "drill", str(songs["dir"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--combined", "--json")
    assert code == 0, err
    manifest = json.loads(out)
    assert len(manifest) == 2 and manifest[0]["path"] == manifest[1]["path"]
    dest = tmp_path / "out" / "drill 70-100.mp3"
    assert manifest[0]["path"] == str(dest)
    assert manifest[1]["offset"] == pytest.approx(manifest[0]["seconds"] + 0.45, abs=0.01)
    planned = manifest[1]["offset"] + manifest[1]["seconds"]
    data, sr = sf.read(dest, always_2d=True)
    assert abs(len(data) / sr - planned) / planned < 0.02
    # The two beeps sit between the songs: 880 Hz dominates there and nowhere before.
    first_beep = data[int((manifest[0]["seconds"] + 0.05) * sr): int((manifest[0]["seconds"] + 0.13) * sr)]
    spectrum = np.abs(np.fft.rfft(first_beep[:, 0]))
    freqs = np.fft.rfftfreq(len(first_beep), 1 / sr)
    assert abs(freqs[np.argmax(spectrum)] - 880.0) < 20
    before = data[: int(manifest[0]["seconds"] * sr) - sr]
    assert np.abs(before).max() > 0.05  # the songs themselves are there
    tags = ID3(dest)
    assert str(tags["TIT2"]) == "drill 70-100"
    assert "combined: 2 song(s)" in str(tags["COMM:breakdig:eng"])
    # A combined WAV carries a cue where each song starts, labelled with its title.
    code, out, err = run(capsys, "drill", str(songs["dir"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out-wav"), "--combined", "--format", "wav",
                         "--ladder", "70-100:20")
    assert code == 0, err
    assert cue_labels(tmp_path / "out-wav" / "drill 70-100.wav") == ["Drill Me", "Short One"]


def test_cli_drill_playlist_missing_entry(capsys, tmp_path, songs, stub_models):
    playlist = tmp_path / "gig.m3u"
    playlist.write_text("# Nov 14 setlist\n"
                        f"{songs['one']}\n"
                        "\n"
                        f"{songs['two'].relative_to(tmp_path)}\n"  # relative to the list's folder
                        f"{tmp_path / 'gone.mp3'}\n", encoding="utf-8")
    code, out, err = run(capsys, "drill", str(playlist), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"))
    assert code == 0, err
    assert "in the setlist but not on disk" in err
    assert "2 file(s)" in out
    assert (tmp_path / "out" / "Fixture Band - Drill Me - drill 70-100.mp3").exists()
    assert (tmp_path / "out" / "Other Artist - Short One - drill 70-100.mp3").exists()


def test_cli_drill_empty_folder(capsys, tmp_path):
    (tmp_path / "empty").mkdir()
    code, out, err = run(capsys, "drill", str(tmp_path / "empty"), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"))
    assert code == 1 and "no audio files found" in err


def test_cli_drill_keep_and_drop_flags(capsys, tmp_path, songs, stub_models, monkeypatch):
    from conftest import InputAsDrums

    from breakdig import export as export_module
    monkeypatch.setattr(export_module, "load_separator", InputAsDrums)
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--drop", "vocals", "--ladder", "70-100:20")
    assert code == 0, err  # the fixture has no vocals, so no-vocals equals the mix
    dest = tmp_path / "out" / "Fixture Band - Drill Me - drill 70-100.mp3"
    assert dest.exists()
    # A rerun with the same --drop overwrites its own file (the tag stores the stems in a
    # space-free form, so the identity still matches).
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--drop", "vocals", "--ladder", "70-100:20")
    assert code == 0, err
    assert not list((tmp_path / "out").glob("*(2)*"))
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out2"), "--keep", "drums", "--ladder", "70-100:20")
    assert code == 0, err
    assert (tmp_path / "out2" / "Fixture Band - Drill Me - drill 70-100.mp3").exists()
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out3"), "--keep", "drums", "--drop", "vocals")
    assert code != 0 and "use --keep or --drop, not both" in err


@pytest.mark.parametrize("argv, message", [
    (["--ladder", "40-100"], "50 <= start < end <= 100"),
    (["--ladder", "100-70"], "50 <= start < end <= 100"),
    (["--ladder", "70-100:0"], "step must be 1 to 20"),
    (["--ladder", "70-100:21"], "step must be 1 to 20"),
    (["--ladder", "seventy"], "START-END:STEP"),
    (["--bars", "9"], "--bars takes A-B like 9-16"),
    (["--bars", "5-2"], "1 <= A <= B"),
    (["--passes", "0"], "must be 1 or more"),
    (["--count-in", "0"], "must be 1 or more"),
    (["--gap-bars", "0"], "must be 1 or more"),
    (["--keep", "kazoo"], "--keep takes one to three of"),
    (["--drop", "kazoo"], "--drop takes some of"),
    (["--drop", "drums,bass,vocals,other"], "would leave no stems"),
    (["--format", "flac"], "invalid choice"),
])
def test_cli_drill_bad_args(capsys, tmp_path, songs, argv, message):
    code, out, err = run(capsys, "drill", str(songs["dir"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), *argv)
    assert code != 0 and message in err


def test_cli_drill_below_sweet_spot_warns(capsys, tmp_path, songs, stub_models):
    code, out, err = run(capsys, "drill", str(songs["one"]), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--ladder", "60-100:10")
    assert code == 0
    assert "60% is outside the 0.75x-1.5x stretch sweet spot" in err


def test_cli_drill_loud_master_keeps_headroom(capsys, tmp_path, songs, stub_models, fixture_audio):
    loud = tmp_path / "loud"
    loud.mkdir()
    data, sr = sf.read(fixture_audio["mix"], dtype="float32")
    sf.write(loud / "loud.wav", data * 1.4, sr, subtype="FLOAT")
    code, out, err = run(capsys, "drill", str(loud / "loud.wav"), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out-wav"), "--format", "wav")
    assert code == 0, err
    dest = tmp_path / "out-wav" / "Unknown Artist - loud - drill 70-100.wav"
    assert dest.exists()
    assert sf.info(dest).subtype == "FLOAT"  # the float-master treatment, as in every export
    peak = float(np.abs(sf.read(dest, dtype="float32")[0]).max())
    assert peak == pytest.approx(1.4, rel=0.02)
    code, out, err = run(capsys, "drill", str(loud / "loud.wav"), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out-mp3"))
    assert code == 0, err
    mp3_peak = float(np.abs(sf.read(str(tmp_path / "out-mp3" / "Unknown Artist - loud - drill 70-100.mp3"),
                                    dtype="float32")[0]).max())
    # Turned down to 1.0 before the encode; the decode overshoots by about 9%, the way every
    # MP3 does (the codebase's "lossy decodes past full scale"), so 1.2 and not 1.001.
    assert 0.9 < mp3_peak < 1.2


def test_cli_drill_channels(capsys, tmp_path, songs, stub_models, fixture_audio):
    four = tmp_path / "quad"
    four.mkdir()
    data, sr = sf.read(fixture_audio["mix"], dtype="float32", always_2d=True)
    sf.write(four / "quad.wav", np.concatenate([data] * 4, axis=1), sr)
    code, out, err = run(capsys, "drill", str(four / "quad.wav"), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--ladder", "70-100:20")
    assert code == 0, err
    assert sf.info(str(tmp_path / "out" / "Unknown Artist - quad - drill 70-100.mp3")).channels == 2
    mono = tmp_path / "mono"
    mono.mkdir()
    sf.write(mono / "mono.wav", data.mean(axis=1), sr)
    code, out, err = run(capsys, "drill", str(mono / "mono.wav"), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out2"), "--ladder", "70-100:20")
    assert code == 0, err
    assert sf.info(str(tmp_path / "out2" / "Unknown Artist - mono - drill 70-100.mp3")).channels == 1


def test_cli_drill_sample_rates(capsys, tmp_path, songs, stub_models, fixture_audio):
    """A 48 kHz source: wav drills stay at 48 kHz, mp3 drills come out at 44.1 kHz."""
    src = tmp_path / "48k"
    src.mkdir()
    data, _sr = sf.read(fixture_audio["mix"], dtype="float32", always_2d=True)
    sf.write(src / "fortyeight.wav", data, 48000)
    seconds = data.shape[0] / 48000
    code, out, err = run(capsys, "drill", str(src / "fortyeight.wav"),
                         "--home", str(tmp_path / "home"), "--out", str(tmp_path / "out-wav"),
                         "--format", "wav", "--ladder", "70-100:20")
    assert code == 0, err
    wav = tmp_path / "out-wav" / "Unknown Artist - fortyeight - drill 70-100.wav"
    assert sf.info(wav).samplerate == 48000
    code, out, err = run(capsys, "drill", str(src / "fortyeight.wav"),
                         "--home", str(tmp_path / "home"), "--out", str(tmp_path / "out-mp3"),
                         "--ladder", "70-100:20")
    assert code == 0, err
    mp3 = tmp_path / "out-mp3" / "Unknown Artist - fortyeight - drill 70-100.mp3"
    info = sf.info(mp3)
    assert info.samplerate == 44100  # resampled for the lossy encode, as the docs say
    planned = total_seconds(plan(seconds, 120.0, Ladder("70-100", (70, 90, 100))))
    assert abs(info.frames / info.samplerate - planned) / planned < 0.02


def test_cli_drill_unreadable_source(capsys, tmp_path, songs, stub_models):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not an mp3")
    code, out, err = run(capsys, "drill", str(bad), "--home", str(tmp_path / "home"),
                         "--out", str(tmp_path / "out"), "--json")
    assert code == 1 and ("Invalid data" in err or "failed" in err)
