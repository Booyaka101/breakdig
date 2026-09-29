import csv
import os
import struct
import subprocess

import numpy as np
import pytest
import soundfile as sf
import soxr
from conftest import InputAsDrums, needs_ffmpeg
from mutagen.wave import WAVE

from breakdig import audio
from breakdig import export as export_module

from breakdig.export import (MAX_NAME, SP404_CSV, clip_name, cut, export, quietest_point, safe_name)
from breakdig.query import Section

pytestmark = needs_ffmpeg


def section(path, first=5, last=8, start=8.0, end=16.0, artist="Fixture Band", title="Test Track", bpm=120.0,
            key="", beats=()):
    return Section(1, first, last, start, end, bpm, -34.0, False, artist, title, "Album", str(path), key,
                   list(beats))


def test_quietest_point():
    x = np.sin(2 * np.pi * np.arange(1000) / 100)[:, None]  # zero every 50 samples
    assert quietest_point(x, 148, 5) == 150
    assert quietest_point(x, 120, 5) == 115  # no zero within reach, so the quietest there is
    assert quietest_point(x, 0, 5) == 0
    # The left channel crosses zero at 150, but the right is loud there.
    stereo = np.column_stack([x[:, 0], np.roll(x[:, 0], 25)])
    assert abs(stereo[quietest_point(stereo, 150, 20)]).max() < 0.75


def test_clip_name_format():
    s = section("x.wav", start=67.6)
    assert clip_name(s) == "Fixture Band - Test Track - 4 bars - 120 BPM - 01.08.wav"
    s.bpm = None
    assert clip_name(s).endswith(" - 4 bars - unknown BPM - 01.08.wav")


def test_clip_name_illegal_characters_and_length():
    s = section("x.wav", artist='AC/DC: "Live"', title="What? <Yes|No>*" + "x" * 300)
    name = clip_name(s)
    assert len(name) <= MAX_NAME
    assert not set('<>:"/\\|?*') & set(name)
    assert name.startswith("AC_DC_ _Live_ - What_ _Yes_No__")
    assert name.endswith(" - 4 bars - 120 BPM - 00.08.wav")
    assert safe_name("trailing dots...") == "trailing dots"


@pytest.mark.parametrize("ext", ["flac", "mp3", "m4a", "ogg"])
def test_trimmed_decode_matches_full_decode(fixture_audio, tmp_path, ext):
    path = str(tmp_path / f"fixture.{ext}")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(fixture_audio["mix"]), path], check=True)
    info = audio.probe(path)
    full = audio.decode(path)
    for start in (1.0, 8.0, 15.0, 23.3):
        x = audio.decode(path, start=start, end=start + 1.0, info=info)
        a = round(start * info.sample_rate)
        assert len(x) == info.sample_rate
        assert np.array_equal(x, full[a: a + len(x)])


def test_trimmed_decode_is_exact_on_vorbis_short_blocks(tmp_path):
    # Transients make Vorbis switch to short blocks, which throws off trimming by timestamp.
    t = np.arange(20 * 44100) / 44100
    x = 0.3 * np.random.default_rng(0).standard_normal(t.size) * np.exp(-(t % 0.5) * 20)
    sf.write(tmp_path / "bursts.wav", np.stack([x, x], 1), 44100)
    path = str(tmp_path / "bursts.ogg")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "bursts.wav"), "-c:a", "libvorbis", path],
                   check=True)
    full = audio.decode(path)
    for start in (5.0, 10.0, 15.0):
        clip = audio.decode(path, start=start, end=start + 1.0)
        a = round(start * 44100)
        assert np.array_equal(clip, full[a: a + 44100])


def test_cut_is_bar_exact_and_starts_on_zero_crossings(fixture_audio):
    clip = cut(str(fixture_audio["mix"]), 8.0, 16.0)
    assert clip.sample_rate == 44100 and clip.bits == 16
    assert len(clip.audio) / 44100 == pytest.approx(8.0, abs=0.004)
    mono = clip.audio.mean(axis=1)
    full = sf.read(fixture_audio["mix"], dtype="float32")[0]
    # The cut is the original audio, offset by at most the 2 ms snap.
    start = int(8.0 * 44100)
    offsets = [k for k in range(-88, 89) if np.allclose(full[start + k: start + k + 1000], mono[:1000], atol=1e-4)]
    assert len(offsets) == 1
    assert abs(mono[0]) < 0.02 * np.abs(mono).max()


def test_export_writes_tagged_wav(fixture_audio, tmp_path):
    (r,) = export([section(fixture_audio["mix"])], str(tmp_path))
    assert r.error is None and r.path
    assert r.path.endswith("Fixture Band - Test Track - 4 bars - 120 BPM - 00.08.wav")
    info = sf.info(r.path)
    assert (info.samplerate, info.subtype) == (44100, "PCM_16")
    tags = WAVE(r.path).tags
    assert str(tags["TIT2"]) == "Test Track (bars 5-8)"
    assert str(tags["TPE1"]) == "Fixture Band"
    assert str(tags["TALB"]) == "Album"
    assert str(tags["TBPM"]) == "120"
    assert not list(tmp_path.glob("*.part"))


def test_same_name_from_another_source_gets_a_number(fixture_audio, tmp_path):
    (a,) = export([section(fixture_audio["mix"])], str(tmp_path))
    (b,) = export([section(fixture_audio["drums"])], str(tmp_path))
    assert b.path.endswith("00.08 (2).wav") and a.path != b.path
    (again,) = export([section(fixture_audio["mix"])], str(tmp_path))
    assert again.path == a.path
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_float_source_exports_24_bit(fixture_audio, tmp_path):
    (r,) = export([section(fixture_audio["drums"])], str(tmp_path))
    assert sf.info(r.path).subtype == "PCM_24"


def test_sp404_mode(fixture_audio, tmp_path):
    surround = tmp_path / "six channels.wav"
    x = sf.read(fixture_audio["mix"], dtype="float32")[0]
    sf.write(surround, np.column_stack([x] * 6), 44100, subtype="PCM_24")
    out = tmp_path / "sp"
    first = [section(fixture_audio["mix"]), section(surround, 1, 2, 0.0, 4.0)]
    results = export(first, str(out), sp404="mk2")
    assert [r.path.split("\\")[-1].split("/")[-1] for r in results] == ["BRK_0001.WAV", "BRK_0002.WAV"]
    for r in results:
        info = sf.info(r.path)
        assert (info.samplerate, info.subtype) == (48000, "PCM_16")
        assert info.channels <= 2
        assert info.duration == pytest.approx(r.section.end - r.section.start, abs=0.005)

    # Running it again skips what is there and numbers new clips after it.
    again = export(first + [section(fixture_audio["mix"], 1, 4, 0.0, 8.0)], str(out), sp404="mk2")
    assert [r.skipped for r in again] == [True, True, False]
    assert again[2].path.endswith("BRK_0003.WAV")
    with open(out / SP404_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == ["BRK_0001.WAV", "BRK_0002.WAV", "BRK_0003.WAV"]
    assert rows[0]["start"] == "8.000" and rows[0]["bpm"] == "120.0" and rows[0]["bars"] == "4"


def test_sp404_sx_is_44k_and_never_mixed_with_mk2(loud_48k_stereo, tmp_path):
    s, out = section(loud_48k_stereo, 1, 4, 0.0, 8.0), str(tmp_path / "sp")
    (r,) = export([s], out, sp404="sx")
    info = sf.info(r.path)
    assert (info.samplerate, info.subtype) == (44100, "PCM_16")
    assert info.duration == pytest.approx(8.0, abs=0.005)
    assert export([s], out, sp404="sx")[0].skipped
    with pytest.raises(ValueError, match="already has 44.1 kHz SP-404 clips"):
        export([section(loud_48k_stereo, 5, 6, 8.0, 12.0)], out, sp404="mk2")
    with pytest.raises(ValueError, match="unknown SP-404 model 'sp1200'"):
        export([s], str(tmp_path / "other"), sp404="sp1200")


def test_sp404_index_edited_in_excel_is_kept(fixture_audio, tmp_path):
    export([section(fixture_audio["mix"])], str(tmp_path), sp404="mk2")
    csv_path = tmp_path / SP404_CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows[0]["pad"] = "A1"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    again = export([section(fixture_audio["mix"]), section(fixture_audio["mix"], 1, 4, 0.0, 8.0)],
                   str(tmp_path), sp404="mk2")
    assert [r.skipped for r in again] == [True, False]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [(r["file"], r["pad"]) for r in rows] == [("BRK_0001.WAV", "A1"), ("BRK_0002.WAV", "")]


def read_index(folder):
    with open(folder / SP404_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_sp404_rerun_keeps_a_longer_section_from_the_same_bar(fixture_audio, tmp_path):
    export([section(fixture_audio["mix"], 5, 6, 8.0, 12.0)], str(tmp_path), sp404="mk2")
    (r,) = export([section(fixture_audio["mix"], 5, 8, 8.0, 16.0)], str(tmp_path), sp404="mk2")
    assert r.path.endswith("BRK_0002.WAV")


def test_sp404_rerun_after_a_file_was_deleted(fixture_audio, tmp_path):
    mix = fixture_audio["mix"]
    export([section(mix, 1, 2, 0.0, 4.0), section(mix, 5, 8, 8.0, 16.0)], str(tmp_path), sp404="mk2")
    (tmp_path / "BRK_0002.WAV").unlink()
    again = export([section(mix, 5, 8, 8.0, 16.0), section(mix, 3, 4, 4.0, 8.0)], str(tmp_path), sp404="mk2")
    # The deleted one is exported again, and no number in index.csv is reused.
    assert [os.path.basename(r.path) for r in again] == ["BRK_0003.WAV", "BRK_0004.WAV"]
    assert [r["file"] for r in read_index(tmp_path)] == [f"BRK_000{i}.WAV" for i in range(1, 5)]


def test_sp404_clips_from_a_killed_export_are_listed(fixture_audio, tmp_path):
    mix = fixture_audio["mix"]
    export([section(mix, 1, 2, 0.0, 4.0), section(mix, 5, 8, 8.0, 16.0)], str(tmp_path), sp404="mk2")
    rows = read_index(tmp_path)
    (tmp_path / SP404_CSV).write_text(",".join(rows[0]) + "\n", encoding="utf-8-sig")  # killed before the save
    export([section(mix, 3, 4, 4.0, 8.0)], str(tmp_path), sp404="mk2")
    again = read_index(tmp_path)
    assert again[:2] == rows and again[2]["file"] == "BRK_0003.WAV"


def test_sp404_index_saved_by_excel_as_ansi(fixture_audio, tmp_path):
    export([section(fixture_audio["mix"], 1, 2, 0.0, 4.0, artist="Motörhead")], str(tmp_path), sp404="mk2")
    (row,) = read_index(tmp_path)
    assert row["artist"] == "Motörhead"
    row["start"], row["end"] = "0", "4"  # Excel drops the trailing zeros
    with open(tmp_path / SP404_CSV, "w", newline="", encoding="cp1252") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    (r,) = export([section(fixture_audio["mix"], 1, 2, 0.0, 4.0)], str(tmp_path), sp404="mk2")
    assert r.skipped
    assert read_index(tmp_path)[0]["artist"] == "Motörhead"


def test_lossy_source_past_full_scale_is_not_clipped(tmp_path):
    t = np.arange(4 * 44100) / 44100
    x = np.sign(np.sin(2 * np.pi * 110 * t))  # a full-scale square rings past 1.0 once encoded
    sf.write(tmp_path / "loud.wav", np.stack([x, x], 1), 44100, subtype="FLOAT")
    src = tmp_path / "loud.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "loud.wav"), "-b:a", "128k", str(src)],
                   check=True)
    s = section(src, 1, 1, 1.0, 3.0)
    assert np.abs(cut(str(src), 1.0, 3.0).audio).max() > 1.0
    (plain,) = export([s], str(tmp_path / "plain"))
    assert sf.info(plain.path).subtype == "FLOAT"
    assert np.abs(sf.read(plain.path)[0]).max() > 1.0
    (sp,) = export([s], str(tmp_path / "sp"), sp404="mk2")
    y = sf.read(sp.path)[0]
    assert np.abs(y).max() == pytest.approx(1.0, abs=1e-3) and (np.abs(y) > 0.999).sum() < 50


def test_missing_source_is_reported(tmp_path):
    (r,) = export([section(tmp_path / "gone.flac")], str(tmp_path / "out"))
    assert r.path is None and "source file is gone" in r.error


def test_corrupt_source_is_reported(tmp_path):
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"fLaC" + b"\xff" * 1000)
    (r,) = export([section(bad)], str(tmp_path / "out"))
    assert r.path is None and r.error


def test_sp404_index_saved_by_excel_with_semicolons(fixture_audio, tmp_path):
    export([section(fixture_audio["mix"])], str(tmp_path), sp404="mk2")
    (row,) = read_index(tmp_path)
    row.update(start="8", end="16", pad="A1")  # a comma-decimal locale also rewrites the numbers
    with open(tmp_path / SP404_CSV, "w", newline="", encoding="cp1252") as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter=";")
        w.writeheader()
        w.writerow(row)
    again = export([section(fixture_audio["mix"]), section(fixture_audio["mix"], 1, 4, 0.0, 8.0)],
                   str(tmp_path), sp404="mk2")
    assert [r.skipped for r in again] == [True, False]
    with open(tmp_path / SP404_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert [(r["file"], r["pad"]) for r in rows] == [("BRK_0001.WAV", "A1"), ("BRK_0002.WAV", "")]


@pytest.mark.parametrize("sp404", [None, "mk2"])
def test_moved_source_finds_its_earlier_export(fixture_audio, tmp_path, sp404):
    moved = tmp_path / "moved.wav"
    moved.write_bytes(fixture_audio["mix"].read_bytes())
    out = tmp_path / "out"
    (a,) = export([section(fixture_audio["mix"], key="abc123")], str(out), sp404=sp404)
    (b,) = export([section(moved, key="abc123")], str(out), sp404=sp404)
    assert b.skipped if sp404 else b.path == a.path
    assert len([p for p in out.iterdir() if p.suffix.lower() == ".wav"]) == 1


def test_index_csv_open_in_excel_stops_before_any_clip(fixture_audio, tmp_path, monkeypatch):
    export([section(fixture_audio["mix"])], str(tmp_path), sp404="mk2")
    real = export_module.os.replace

    def locked(src, dst):
        if str(dst).endswith(SP404_CSV):
            raise PermissionError(13, "Access is denied")
        real(src, dst)
    monkeypatch.setattr(export_module.os, "replace", locked)
    with pytest.raises(ValueError, match="open in Excel"):
        export([section(fixture_audio["mix"], 1, 4, 0.0, 8.0)], str(tmp_path), sp404="mk2")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["BRK_0001.WAV", SP404_CSV]


def test_failed_write_leaves_no_part_file(fixture_audio, tmp_path, monkeypatch):
    def denied(src, dst):
        raise PermissionError(13, "Access is denied")
    monkeypatch.setattr(export_module.os, "replace", denied)
    (r,) = export([section(fixture_audio["mix"])], str(tmp_path))
    assert "Access is denied" in r.error and not list(tmp_path.iterdir())


def test_the_same_section_twice_is_written_once(fixture_audio, tmp_path):
    results = export([section(fixture_audio["mix"])] * 2, str(tmp_path))
    assert len(results) == 1


def test_out_must_be_a_folder(fixture_audio, tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="is a file, not a folder"):
        export([section(fixture_audio["mix"])], str(tmp_path / "file"))
    # Also what a folder on a drive that is not plugged in gets.
    with pytest.raises(ValueError, match="^cannot create .*sub: [^[]+$"):
        export([section(fixture_audio["mix"])], str(tmp_path / "file" / "sub"))


@pytest.fixture
def loud_48k_stereo(fixture_audio, tmp_path):
    x = sf.read(fixture_audio["mix"], dtype="float32")[0]
    path = tmp_path / "loud.wav"
    sf.write(path, soxr.resample(np.column_stack([x, -x]) * 1.1, 44100, 48000), 48000, subtype="FLOAT")
    return path


@pytest.mark.parametrize("source", ["mix", "loud"])
def test_isolated_drums_line_up_with_the_plain_cut(fixture_audio, loud_48k_stereo, source):
    path = str(fixture_audio["mix"] if source == "mix" else loud_48k_stereo)
    sep = InputAsDrums()
    plain, iso = cut(path, 8.0, 16.0), cut(path, 8.0, 16.0, {"drums"}, sep)
    assert iso.audio.shape[1] == plain.audio.shape[1] and iso.stems == "drums"
    assert sep.peaks[0] <= 0.5 + 1e-6  # never near the separator's normalisation threshold
    # Both snap to the quietest point nearby, which a resample can move by a frame or two.
    k = round((iso.start - plain.start) * iso.sample_rate)
    assert abs(k) <= 2 * round(0.002 * iso.sample_rate)
    n = min(len(plain.audio), len(iso.audio)) - 2 * abs(k)
    a, b = (plain.audio[k:k + n], iso.audio[:n]) if k >= 0 else (plain.audio[:n], iso.audio[-k:-k + n])
    # The round trip through Demucs's 44.1 kHz drops the top of the 48 kHz file's band.
    err = [np.sqrt(np.mean((np.roll(a, lag, axis=0) - b) ** 2) / np.mean(a ** 2)) for lag in (-1, 0, 1)]
    assert err[1] < (1e-6 if source == "mix" else 0.05) and min(err[0], err[2]) > 0.3


def test_isolating_a_stem_that_is_not_there_gives_silence(fixture_audio):
    clip = cut(str(fixture_audio["mix"]), 8.0, 16.0, {"bass"}, InputAsDrums())
    assert clip.audio.shape[1] == 1 and np.abs(clip.audio).max() == 0


@pytest.mark.parametrize("sp404", [None, "mk2"])
def test_isolated_export_is_its_own_clip(fixture_audio, tmp_path, sp404):
    sep, s = InputAsDrums(), section(fixture_audio["mix"])
    (iso,) = export([s], str(tmp_path), sp404=sp404, keep={"drums"}, separator=sep)
    assert str(WAVE(iso.path).tags["COMM:breakdig:eng"]).endswith("[drums]")
    (again,) = export([s], str(tmp_path), sp404=sp404, keep={"drums"}, separator=sep)
    (plain,) = export([s], str(tmp_path), sp404=sp404)
    assert (again.skipped or again.path == iso.path) and plain.path not in (None, iso.path)
    assert len(sep.peaks) == 1 + (not sp404)  # a plain export overwrites, an SP-404 one skips
    if sp404:
        assert [r["stems"] for r in read_index(tmp_path)] == ["drums", ""]
    else:
        assert iso.path.endswith("Test Track - drums - 4 bars - 120 BPM - 00.08.wav")


def test_the_model_loads_only_when_a_section_needs_it(fixture_audio, tmp_path, monkeypatch):
    def broken():
        raise ValueError("cannot load the separation model: no GPU")
    monkeypatch.setattr(export_module, "load_separator", broken)
    s = section(fixture_audio["mix"])
    export([s], str(tmp_path), sp404="mk2", keep={"drums"}, separator=InputAsDrums())
    assert export([s], str(tmp_path), sp404="mk2", keep={"drums"})[0].skipped
    assert export([s], str(tmp_path / "plain"))[0].path
    with pytest.raises(ValueError, match="no GPU"):
        export([section(fixture_audio["mix"], 1, 4, 0.0, 8.0)], str(tmp_path), sp404="mk2", keep={"drums"})


def cue_points(path) -> list[int]:
    data, i, cues = open(path, "rb").read(), 12, []
    while i < len(data):
        fourcc, size = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        if fourcc == b"cue ":
            n = struct.unpack("<I", data[i + 8:i + 12])[0]
            cues = [struct.unpack("<II4sIII", data[i + 12 + 24 * k: i + 36 + 24 * k])[5] for k in range(n)]
        i += 8 + size + size % 2
    return cues


@pytest.mark.parametrize("sp404", [None, "mk2"])
def test_beats_are_cue_points(fixture_audio, tmp_path, sp404):
    s = section(fixture_audio["mix"], beats=np.arange(8.0, 16.0, 0.5) - 0.001)
    (r,) = export([s], str(tmp_path), sp404=sp404)
    rate = sf.info(r.path).samplerate
    cues = cue_points(r.path)
    assert len(cues) == 16 and cues[0] == 0
    assert np.abs(np.diff(cues[1:]) - 0.5 * rate).max() <= 1
    assert abs(cues[1] - 0.499 * rate) <= 0.002 * rate + 1  # off by at most the snap
    assert str(WAVE(r.path).tags["TIT2"]) == "Test Track (bars 5-8)"
