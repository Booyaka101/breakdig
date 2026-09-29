import errno
import os
import shutil
import sqlite3
import string
import subprocess

import mutagen
import numpy as np
import pytest
import soundfile as sf
from conftest import GridBeats, needs_ffmpeg
from mutagen.id3 import TALB, TIT2, TPE1
from mutagen.wave import WAVE

from breakdig import audio, db, profile, scan
from breakdig.db import STEMS, Index
from breakdig.indexer import Abort, Indexer, match_lag
from breakdig.query import PRESETS, find, get_section

pytestmark = needs_ffmpeg


def index_paths(ix, paths):
    todo, known = ix.pending([str(p) for p in paths])
    return [ix.index_file(f) for f in todo], known


def test_worked_example_end_to_end_on_true_stems(cpu_indexer, fixture_audio):
    statuses, _ = index_paths(cpu_indexer, [fixture_audio["mix"]])
    assert statuses == ["ok"]
    (s,) = find(cpu_indexer.index, PRESETS["drums"])
    assert (s.first_bar, s.last_bar, s.bars) == (5, 8, 4)
    assert s.start == pytest.approx(8.0, abs=0.02)
    assert s.end == pytest.approx(16.0, abs=0.02)
    assert s.bpm == pytest.approx(120.0, abs=0.1)
    row = cpu_indexer.index.track(s.track_id)
    assert row["bpm"] == pytest.approx(120.0, abs=0.1)
    assert row["sample_rate"] == 44100 and row["channels"] == 1
    assert cpu_indexer.index.profile_path(row["key"]).exists()
    # Nothing else in the fixture plays alone.
    assert find(cpu_indexer.index, PRESETS["vocals"]) == []
    assert find(cpu_indexer.index, PRESETS["bass+drums"]) == []


def test_indexing_is_idempotent(cpu_indexer, fixture_audio, tmp_path):
    index_paths(cpu_indexer, [fixture_audio["mix"]])
    copy = tmp_path / "copy of fixture.wav"
    shutil.copy(fixture_audio["mix"], copy)
    statuses, known = index_paths(cpu_indexer, [fixture_audio["mix"], copy])
    assert statuses == [] and known == 2
    assert cpu_indexer.index.stats()["by_status"] == {"ok": 1}


def test_moved_file_keeps_its_index_entry(cpu_indexer, fixture_audio, tmp_path):
    src = tmp_path / "a.wav"
    shutil.copy(fixture_audio["mix"], src)
    index_paths(cpu_indexer, [src])
    moved = tmp_path / "b.wav"
    src.rename(moved)
    statuses, known = index_paths(cpu_indexer, [moved])
    assert statuses == [] and known == 1
    (row,) = cpu_indexer.index.ok_tracks()
    assert row["path"] == str(moved)


def test_same_audio_in_another_format_is_a_duplicate(cpu_indexer, fixture_audio, tmp_path):
    mp3 = tmp_path / "fixture.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(fixture_audio["mix"]), "-b:a", "192k", str(mp3)],
                   check=True)
    statuses, _ = index_paths(cpu_indexer, [fixture_audio["mix"], mp3])
    assert statuses == ["ok", "duplicate"]
    first, dup = cpu_indexer.index.conn.execute("SELECT id, dup_of FROM tracks ORDER BY id").fetchall()
    assert dup["dup_of"] == first["id"]
    assert len(find(cpu_indexer.index, PRESETS["drums"])) == 1
    assert cpu_indexer.index.failures() == []


def test_duplicate_takes_over_when_the_original_is_deleted(cpu_indexer, fixture_audio, tmp_path):
    wav, mp3 = tmp_path / "song.wav", tmp_path / "song.mp3"
    shutil.copy(fixture_audio["mix"], wav)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
    assert index_paths(cpu_indexer, [wav, mp3])[0] == ["ok", "duplicate"]
    wav.unlink()
    assert index_paths(cpu_indexer, [mp3])[0] == ["ok"]
    (s,) = find(cpu_indexer.index, PRESETS["drums"])
    assert s.path == str(mp3) and (s.first_bar, s.last_bar) == (5, 8)
    assert cpu_indexer.index.stats()["by_status"] == {"ok": 1}


def test_moved_original_keeps_its_duplicate(cpu_indexer, fixture_audio, tmp_path):
    wav, mp3 = tmp_path / "song.wav", tmp_path / "song.mp3"
    shutil.copy(fixture_audio["mix"], wav)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
    index_paths(cpu_indexer, [wav, mp3])
    moved = tmp_path / "moved.wav"
    wav.rename(moved)
    # The duplicate is listed first, before the scan reaches the moved original.
    assert index_paths(cpu_indexer, [mp3, moved]) == ([], 2)


def test_match_lag_finds_trims_and_rejects_other_audio(fixture_audio):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    hop = int(profile.HOP_SECONDS * sr)
    a = profile.to_db(profile.power_envelope([x[:, None]], hop))
    b = profile.to_db(profile.power_envelope([np.roll(x, sr)[:, None]], hop))  # one second later
    assert match_lag(a, a) == 0
    assert match_lag(a[2:], a) == 2  # 100 ms trimmed from the start
    assert match_lag(a, b) is None
    assert match_lag(a[:50], a[:50]) is None  # too short to judge


def test_the_bar_after_the_last_downbeat_counts_if_the_audio_covers_it():
    envs = {k: np.ones(200) for k in (*STEMS, "mix")}  # 10 s at 50 ms
    rows = profile.profile(envs, np.array([0.0, 2.0, 4.0, 6.0]), 0.05)
    assert [(r[1], r[2]) for r in rows] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    rows = profile.profile(envs, np.array([1.0, 3.0, 5.0, 7.0, 9.0]), 0.05)
    assert rows[-1][1:3] == (7, 9)  # a bar from 9 s would run past the end
    envs["mix"][140:] = 1e-8  # -80 dB from 7 s: the song fades out halfway into that bar
    rows = profile.profile(envs, np.array([0.0, 2.0, 4.0, 6.0]), 0.05)
    assert rows[-1][1:3] == (4, 6)


def test_the_added_last_bar_is_as_long_as_the_bars_before_it():
    envs = {k: np.ones(400) for k in (*STEMS, "mix")}
    # Six 2 s bars, then the tempo picks up to 1.5 s bars.
    downbeats = np.array([0, 2, 4, 6, 8, 10, 12, 13.5, 15, 16.5, 18])
    rows = profile.profile(envs, downbeats, 0.05)
    assert rows[-1][1:3] == (18, 19.5)


def test_eight_bit_input(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    p = tmp_path / "eight bit.wav"
    sf.write(p, x, sr, subtype="PCM_U8")
    statuses, _ = index_paths(cpu_indexer, [p])
    assert statuses == ["ok"]
    (s,) = find(cpu_indexer.index, PRESETS["drums"])
    assert (s.first_bar, s.last_bar) == (5, 8)


def test_corrupt_file_is_logged_and_skipped(cpu_indexer, fixture_audio, tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"ID3" + bytes(range(256)) * 400)
    statuses, _ = index_paths(cpu_indexer, [bad, fixture_audio["mix"]])
    assert statuses == ["failed", "ok"]
    (row,) = cpu_indexer.index.failures()
    assert row["path"] == str(bad) and "unreadable or corrupt" in row["error"]


def test_drm_file_is_skipped(cpu_indexer, tmp_path):
    p = tmp_path / "Protected.m4p"
    p.write_bytes(b"\0" * 5000)
    assert index_paths(cpu_indexer, [p])[0] == ["failed"]
    assert "DRM" in cpu_indexer.index.failures()[0]["error"]


def test_too_few_downbeats_is_no_grid(cpu_indexer, fixture_audio):
    cpu_indexer._beats = GridBeats(bars=6)  # 7 downbeats
    assert index_paths(cpu_indexer, [fixture_audio["mix"]])[0] == ["no_grid"]
    (row,) = cpu_indexer.index.failures()
    assert "only 7 downbeats" in row["error"]
    assert find(cpu_indexer.index, PRESETS["drums"]) == []


def test_beat_tracker_failure_is_recorded(cpu_indexer, fixture_audio):
    def boom(mono, sr):
        raise RuntimeError("cannot find a tempo")
    cpu_indexer._beats = boom
    assert index_paths(cpu_indexer, [fixture_audio["mix"]])[0] == ["failed"]
    assert "beat tracking failed" in cpu_indexer.index.failures()[0]["error"]


class Broken:
    def separate(self, *a):
        raise RuntimeError("model file is damaged")


def test_retry_failed(cpu_indexer, fixture_audio):
    real = cpu_indexer._sep
    cpu_indexer._sep = Broken()
    assert index_paths(cpu_indexer, [fixture_audio["mix"]])[0] == ["failed"]
    assert "separation failed: RuntimeError: model file is damaged" in cpu_indexer.index.failures()[0]["error"]
    cpu_indexer._sep = real
    assert cpu_indexer.pending([str(fixture_audio["mix"])]) == ([], 1)
    todo, _ = cpu_indexer.pending([str(fixture_audio["mix"])], retry_failed=True)
    assert [cpu_indexer.index_file(f) for f in todo] == ["ok"]


def test_interrupted_run_resumes(cpu_indexer, fixture_audio, tmp_path):
    second = tmp_path / "second.wav"
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    sf.write(second, x[::-1], sr)  # different audio, so not a duplicate
    real = cpu_indexer._sep

    class Interrupt:
        def separate(self, *a):
            raise KeyboardInterrupt

    todo, _ = cpu_indexer.pending([str(fixture_audio["mix"]), str(second)])
    assert cpu_indexer.index_file(todo[0]) == "ok"
    cpu_indexer._sep = Interrupt()
    with pytest.raises(KeyboardInterrupt):
        cpu_indexer.index_file(todo[1])
    cpu_indexer._sep = real
    todo, known = cpu_indexer.pending([str(fixture_audio["mix"]), str(second)])
    assert known == 1 and [f.path for f in todo] == [str(second)]


def test_walk_and_tags(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["03 - Artist - Song.mp3", "sub/Other.FLAC", "notes.txt", "sub/cover.jpg"]:
        (tmp_path / name).write_bytes(b"x")
    found = scan.walk([str(tmp_path)])
    assert [p.replace("\\", "/").split("/")[-1] for p in found] == ["03 - Artist - Song.mp3", "Other.FLAC"]
    for exclude in (["*.flac"], ["SUB"]):
        assert [p.replace("\\", "/").split("/")[-1] for p in scan.walk([str(tmp_path)], exclude)] == [
            "03 - Artist - Song.mp3"]
    tags = scan.read_tags(str(tmp_path / "03 - Artist - Song.mp3"))
    assert (tags["artist"], tags["title"]) == ("Artist", "Song")
    tags = scan.read_tags(str(tmp_path / "07 - Song.mp3"))
    assert (tags["artist"], tags["title"]) == ("Unknown Artist", "Song")
    tags = scan.read_tags(str(tmp_path / "311 - Amber.mp3"))
    assert (tags["artist"], tags["title"]) == ("311", "Amber")
    tags = scan.read_tags(str(tmp_path / "808 State - Pacific.mp3"))
    assert (tags["artist"], tags["title"]) == ("808 State", "Pacific")
    tags = scan.read_tags(str(tmp_path / "sub" / "Other.FLAC"))
    assert (tags["artist"], tags["title"]) == ("Unknown Artist", "Other")


def test_content_key(fixture_audio, tmp_path):
    a = scan.content_key(str(fixture_audio["mix"]))
    copy = tmp_path / "x.wav"
    shutil.copy(fixture_audio["mix"], copy)
    b = scan.content_key(str(copy))
    assert a.key == b.key and a.duration == pytest.approx(32.0, abs=0.01)
    with open(copy, "r+b") as f:
        f.seek(100)
        f.write(b"\x01\x02\x03")
    assert scan.content_key(str(copy)).key != a.key


def test_scratch_left_by_a_killed_run_is_swept(tmp_path):
    left = tmp_path / "tmp" / "run-killed"
    left.mkdir(parents=True)
    (left / "mix.wav").write_bytes(b"x")
    with Index(tmp_path) as index:
        Indexer(index).close()
    assert list((tmp_path / "tmp").iterdir()) == []


def test_wav_edits_with_the_same_intro_get_their_own_keys(fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    edit = x.copy()
    edit[len(x) // 2:] *= 0.5
    sf.write(tmp_path / "main.wav", x, sr)
    sf.write(tmp_path / "instrumental.wav", edit, sr)
    assert (tmp_path / "main.wav").stat().st_size > 2 * scan.SAMPLE_BYTES
    assert scan.content_key(str(tmp_path / "main.wav")).key != scan.content_key(str(tmp_path / "instrumental.wav")).key


def test_overwritten_file_replaces_its_old_entry(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    song = tmp_path / "song.wav"
    sf.write(song, x, sr)
    index_paths(cpu_indexer, [song])
    (old,) = cpu_indexer.index.ok_tracks()
    sf.write(song, np.roll(x, sr), sr)  # a different recording of the same length
    assert index_paths(cpu_indexer, [song]) == (["ok"], 0)
    (row,) = cpu_indexer.index.conn.execute("SELECT key, path FROM tracks").fetchall()
    assert row["key"] != old["key"] and not cpu_indexer.index.profile_path(old["key"]).exists()


# As a tagger that also organises files does.
@pytest.mark.parametrize("rename", [None, "New Artist - New Title.wav", "01 - old - name.wav"],
                         ids=["retag", "retag and rename", "retag and change case"])
def test_retagged_file_keeps_its_analysis(cpu_indexer, fixture_audio, tmp_path, rename):
    song = tmp_path / "01 - Old - Name.wav"
    shutil.copy(fixture_audio["mix"], song)
    index_paths(cpu_indexer, [song])
    if rename:
        song = song.rename(tmp_path / rename)
        assert index_paths(cpu_indexer, [song]) == ([], 1)
    _retag(song, "New Title", artist="New Artist")
    cpu_indexer._sep = None  # separating again would load the real model
    assert index_paths(cpu_indexer, [song]) == (["updated"], 0)
    (s,) = find(cpu_indexer.index, PRESETS["drums"])
    assert (s.artist, s.title, s.first_bar) == ("New Artist", "New Title", 5)
    (row,) = cpu_indexer.index.conn.execute("SELECT key, status FROM tracks").fetchall()
    assert row["status"] == "ok" and cpu_indexer.index.profile_path(row["key"]).exists()
    assert index_paths(cpu_indexer, [song]) == ([], 1)


@pytest.mark.parametrize("ext, fmt", [("aiff", "AIFF"), ("wav", "WAV")])
def test_id3_tags_in_aiff_and_wav(fixture_audio, tmp_path, ext, fmt):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    p = tmp_path / f"untitled.{ext}"
    sf.write(p, x[:sr], sr, format=fmt)
    f = mutagen.File(p)
    f.add_tags()
    for frame in (TPE1(encoding=3, text="Real Artist"), TIT2(encoding=3, text="Real Title"),
                  TALB(encoding=3, text="Real Album")):
        f.tags.add(frame)
    f.save()
    assert scan.read_tags(str(p)) == {"artist": "Real Artist", "title": "Real Title", "album": "Real Album"}


def test_unchanged_files_are_not_hashed_again(cpu_indexer, fixture_audio, monkeypatch):
    index_paths(cpu_indexer, [fixture_audio["mix"]])

    def no_hashing(path):
        raise AssertionError(f"hashed {path} again")
    monkeypatch.setattr(scan, "content_key", no_hashing)
    assert index_paths(cpu_indexer, [fixture_audio["mix"]]) == ([], 1)


def test_file_changed_after_the_scan_is_left_for_next_time(cpu_indexer, fixture_audio, tmp_path):
    song = tmp_path / "song.wav"
    shutil.copy(fixture_audio["mix"], song)
    (f,), _ = cpu_indexer.pending([str(song)])
    with open(song, "ab") as out:
        out.write(b"\0" * 10)
    assert cpu_indexer.index_file(f) == "skipped"
    song.unlink()
    assert cpu_indexer.index_file(f) == "skipped"
    assert cpu_indexer.index.stats()["by_status"] == {}


@pytest.mark.parametrize("stage", ["_sep", "_beats"])
def test_gpu_out_of_memory_stops_the_run(cpu_indexer, fixture_audio, stage):

    class NoMemory:
        def separate(self, *a, **kw):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        __call__ = separate
    setattr(cpu_indexer, stage, NoMemory())
    (f,), _ = cpu_indexer.pending([str(fixture_audio["mix"])])
    with pytest.raises(Abort):
        cpu_indexer.index_file(f)
    assert cpu_indexer.index.stats()["by_status"] == {}


@pytest.mark.parametrize("stage", ["decode", "separate"])
def test_a_full_disk_stops_the_run(cpu_indexer, fixture_audio, monkeypatch, stage):


    def full(*a, **kw):
        if stage == "decode":
            raise audio.DecodeError("mix.wav: No space left on device")
        raise OSError(errno.ENOSPC, "There is not enough space on the disk")
    if stage == "decode":
        monkeypatch.setattr(audio, "decode_to_wav", full)
    else:
        monkeypatch.setattr(cpu_indexer._sep, "separate", full)
    (f,), _ = cpu_indexer.pending([str(fixture_audio["mix"])])
    with pytest.raises(Abort, match="disk is full"):
        cpu_indexer.index_file(f)
    assert cpu_indexer.index.stats()["by_status"] == {}


def test_a_cut_short_model_download_is_deleted_and_stops_the_run(cpu_indexer, fixture_audio, tmp_path,
                                                                  monkeypatch):
    monkeypatch.setenv("BREAKDIG_MODELS", str(tmp_path / "models"))
    part = tmp_path / "models" / "f7e0c4bc-ba3fe64a.th"
    part.parent.mkdir()
    part.write_bytes(b"x" * 100)

    class ModelLoadingError(RuntimeError):
        pass

    class Damaged:
        def separate(self, *a, **kw):
            raise ModelLoadingError(f"Invalid checksum for file {part}, expected f7e0c4bc but got 1a2b3c4d")
    cpu_indexer._sep = Damaged()
    (f,), _ = cpu_indexer.pending([str(fixture_audio["mix"])])
    with pytest.raises(Abort, match="Run the same command again"):
        cpu_indexer.index_file(f)
    assert not part.exists()
    assert cpu_indexer.index.stats()["by_status"] == {}


def test_files_that_swap_names_take_their_rows_along(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    one, two = tmp_path / "01.wav", tmp_path / "02.wav"
    sf.write(one, x, sr)
    sf.write(two, x[: 24 * sr], sr)
    index_paths(cpu_indexer, [one, two])
    one.rename(tmp_path / "tmp.wav")
    two.rename(one)
    (tmp_path / "tmp.wav").rename(two)
    assert index_paths(cpu_indexer, [one, two]) == ([], 2)
    rows = cpu_indexer.index.conn.execute("SELECT path, duration FROM tracks ORDER BY path").fetchall()
    assert [(r["path"], round(r["duration"])) for r in rows] == [(str(one), 24), (str(two), 32)]


def test_a_file_renamed_only_in_case_takes_its_row_along(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    song = tmp_path / "Song.wav"
    sf.write(song, x, sr)
    index_paths(cpu_indexer, [song])
    song = song.rename(tmp_path / "song.wav")
    assert index_paths(cpu_indexer, [song]) == ([], 1)
    sf.write(song, np.roll(x, sr), sr)
    assert index_paths(cpu_indexer, [song]) == (["ok"], 0)
    assert [r["path"] for r in cpu_indexer.index.conn.execute("SELECT path FROM tracks")] == [str(song)]


def test_file_overwritten_with_a_copy_of_another_loses_its_row(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    one, two = tmp_path / "one.wav", tmp_path / "two.wav"
    sf.write(one, x, sr)
    sf.write(two, np.roll(x, sr), sr)
    index_paths(cpu_indexer, [one, two])
    shutil.copy(one, two)
    assert index_paths(cpu_indexer, [one, two]) == ([], 2)
    assert [r["path"] for r in cpu_indexer.index.conn.execute("SELECT path FROM tracks")] == [str(one)]


def test_file_trimmed_in_place_is_analysed_again(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    song = tmp_path / "song.wav"
    sf.write(song, x, sr)
    index_paths(cpu_indexer, [song])
    sf.write(song, x[sr // 10:], sr)  # 100 ms of lead-in cut, which moves every bar
    assert index_paths(cpu_indexer, [song]) == (["ok"], 0)
    assert cpu_indexer.index.stats()["by_status"] == {"ok": 1}


def test_edits_that_differ_only_between_the_sampled_bytes(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    x = np.column_stack([x, x])
    edit = x.copy()
    edit[4 * sr: 12 * sr] *= 0.25
    main, other = tmp_path / "main.wav", tmp_path / "instrumental.wav"
    sf.write(main, x, sr, subtype="PCM_32")  # FLOAT adds a PEAK chunk stamped with the write time
    sf.write(other, edit, sr, subtype="PCM_32")
    assert scan.content_key(str(main)).key == scan.content_key(str(other)).key
    statuses, known = index_paths(cpu_indexer, [main, other])
    assert len(statuses) == 2 and known == 0
    assert index_paths(cpu_indexer, [main, other]) == ([], 2)


def test_touched_file_is_hashed_once(cpu_indexer, fixture_audio, tmp_path, monkeypatch):
    song = tmp_path / "song.wav"
    shutil.copy(fixture_audio["mix"], song)
    index_paths(cpu_indexer, [song])
    os.utime(song, (1, 1))
    assert index_paths(cpu_indexer, [song]) == ([], 1)
    monkeypatch.setattr(scan, "content_key", lambda path: pytest.fail(f"hashed {path} again"))
    assert index_paths(cpu_indexer, [song]) == ([], 1)


def test_walk_skips_the_index_home_and_spells_paths_as_the_disk_does(tmp_path):
    music = tmp_path / "Music"
    (music / "breakdig" / "tmp").mkdir(parents=True)
    (music / "a.wav").write_bytes(b"x")
    (music / "breakdig" / "tmp" / "mix.wav").write_bytes(b"x")
    root = str(music).lower() if os.name == "nt" else str(music)
    assert scan.walk([root], skip=[music / "breakdig"]) == [os.path.realpath(music / "a.wav")]


def _retag(path, title, artist=None):
    w = WAVE(path)
    w.add_tags()
    w.tags.add(TIT2(encoding=3, text=title))
    if artist:
        w.tags.add(TPE1(encoding=3, text=artist))
    w.save()


def test_a_profile_another_program_has_open_does_not_stop_the_run(cpu_indexer, fixture_audio, tmp_path,
                                                                  monkeypatch):
    song = tmp_path / "song.wav"
    shutil.copy(fixture_audio["mix"], song)
    index_paths(cpu_indexer, [song])
    old = cpu_indexer.index.profile_path(cpu_indexer.index.by_path(str(song))["key"])

    def locked(*a, **kw):
        raise PermissionError("in use by another process")
    monkeypatch.setattr(db.os, "replace", locked)
    monkeypatch.setattr(db.Path, "unlink", locked)
    _retag(song, "New Title")
    assert index_paths(cpu_indexer, [song]) == (["updated"], 0)
    new = cpu_indexer.index.by_path(str(song))["key"]
    assert cpu_indexer.index.profile_path(new).exists() and old.exists()
    monkeypatch.undo()
    cpu_indexer.close()
    Indexer(cpu_indexer.index).close()  # the next run clears what was left
    assert [p.stem for p in cpu_indexer.index.profiles.glob("*.npz")] == [new]


def test_a_rekey_that_fails_keeps_the_profile(cpu_indexer, fixture_audio):
    index_paths(cpu_indexer, [fixture_audio["mix"]])
    (row,) = cpu_indexer.index.ok_tracks()
    with pytest.raises(sqlite3.OperationalError):
        cpu_indexer.index.rekey(row["id"], {"key": "new", "no_such_column": 1})
    assert cpu_indexer.index.profile_path(row["key"]).exists()
    assert cpu_indexer.index.by_key(row["key"])["id"] == row["id"]


def test_a_second_index_run_on_one_home_stops(cpu_indexer):
    with pytest.raises(Abort, match="another breakdig index run"):
        Indexer(cpu_indexer.index)
    cpu_indexer.close()
    Indexer(cpu_indexer.index).close()


@pytest.mark.skipif(os.name != "nt", reason="drive letters")
def test_originals_on_an_unplugged_drive_keep_their_rows(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    orig, copy = tmp_path / "song.wav", tmp_path / "song.flac"
    sf.write(orig, x, sr)
    sf.write(copy, x, sr, format="FLAC")
    assert index_paths(cpu_indexer, [orig, copy]) == (["ok", "duplicate"], 0)
    letter = next(c for c in reversed(string.ascii_uppercase) if not os.path.exists(f"{c}:\\"))
    row = cpu_indexer.index.by_path(str(orig))
    cpu_indexer.index.update(row["id"], path=f"{letter}:\\Music\\song.wav")
    assert index_paths(cpu_indexer, [copy]) == ([], 1)
    assert cpu_indexer.index.track(row["id"]) is not None
    orig.unlink()  # deleted, where the drive is there
    cpu_indexer.index.update(row["id"], path=str(orig))
    assert index_paths(cpu_indexer, [copy]) == (["ok"], 0)


def test_retry_failed_after_a_copy_of_the_same_audio_was_indexed_and_deleted(cpu_indexer, fixture_audio,
                                                                            tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    flac, wav = tmp_path / "song.flac", tmp_path / "song.wav"
    sf.write(flac, x, sr, format="FLAC")
    sep = cpu_indexer._sep

    class Crashes:
        def separate(self, *a, **kw):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
    cpu_indexer._sep = Crashes()
    assert index_paths(cpu_indexer, [flac]) == (["failed"], 0)
    cpu_indexer._sep = sep
    sf.write(wav, x, sr)
    assert index_paths(cpu_indexer, [flac, wav]) == (["ok"], 1)
    wav.unlink()
    todo, _ = cpu_indexer.pending([str(flac)], retry_failed=True)
    assert [cpu_indexer.index_file(f) for f in todo] == ["updated"]
    (row,) = cpu_indexer.index.conn.execute("SELECT path, status FROM tracks").fetchall()
    assert (row["path"], row["status"]) == (str(flac), "ok")


def test_a_copy_is_checked_again_when_its_original_is_overwritten(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    orig, copy = tmp_path / "song.wav", tmp_path / "song.flac"
    sf.write(orig, x, sr)
    sf.write(copy, x, sr, format="FLAC")
    index_paths(cpu_indexer, [orig, copy])
    sf.write(orig, x[: 20 * sr], sr)  # re-rendered as something else
    assert index_paths(cpu_indexer, [orig, copy]) == (["ok", "ok"], 0)
    assert index_paths(cpu_indexer, [orig, copy]) == ([], 2)


def test_a_section_id_never_names_other_audio(cpu_indexer, fixture_audio, tmp_path):
    x, sr = sf.read(fixture_audio["mix"], dtype="float32")
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, x[: 20 * sr], sr)
    sf.write(b, x, sr)
    index_paths(cpu_indexer, [a, b])
    tid = cpu_indexer.index.by_path(str(b))["id"]
    get_section(cpu_indexer.index, f"{tid}:5-8")
    sf.write(b, x[::-1].copy(), sr)
    index_paths(cpu_indexer, [a, b])
    with pytest.raises(ValueError):
        get_section(cpu_indexer.index, f"{tid}:5-8")


@pytest.mark.skipif(os.name != "nt", reason="drive letters")
def test_a_substituted_drive_keeps_its_letter_but_not_the_typed_case(tmp_path, monkeypatch):
    (tmp_path / "Music" / "idx" / "tmp").mkdir(parents=True)
    (tmp_path / "Music" / "Song.wav").write_bytes(b"x")
    (tmp_path / "Music" / "idx" / "tmp" / "mix.wav").write_bytes(b"x")
    real, real_walk = os.path.realpath, os.walk

    def behind_q(p):  # as after `subst Q: <tmp_path>`
        p = os.path.abspath(p)
        return real(str(tmp_path) + p[2:]) if p[:2].upper() == "Q:" else real(p)

    def walk_q(root):
        for dirpath, dirnames, filenames in real_walk(behind_q(root)):
            yield "Q:" + dirpath[len(real(tmp_path)):], dirnames, filenames
    monkeypatch.setattr(scan.os.path, "realpath", behind_q)
    monkeypatch.setattr(scan.os, "walk", walk_q)
    assert scan.on_disk(r"q:\MUSIC\song.WAV") == r"Q:\Music\Song.wav"
    assert scan.on_disk(r"q:\gone.wav") == r"Q:\gone.wav"
    # The index home arrives resolved, as Index.home_for gives it.
    files = scan.walk([r"q:\music"], skip=[behind_q(r"Q:\Music\idx")])
    assert files == [r"Q:\Music\Song.wav"]


def test_byte_identical_copies_are_not_read_again(cpu_indexer, fixture_audio, tmp_path, monkeypatch):
    (tmp_path / "pack").mkdir()
    (tmp_path / "pack copy").mkdir()
    songs = [shutil.copy2(fixture_audio["mix"], tmp_path / d / "song.wav") for d in ("pack", "pack copy")]
    assert index_paths(cpu_indexer, songs) == (["ok"], 1)
    monkeypatch.setattr(scan, "full_key", lambda path: pytest.fail(f"hashed {path} again"))
    monkeypatch.setattr(scan, "content_key", lambda path: pytest.fail(f"hashed {path} again"))
    assert index_paths(cpu_indexer, songs) == ([], 2)
    monkeypatch.undo()
    with open(songs[1], "r+b") as f:  # changed between the sampled blocks, same size and length
        f.seek(os.path.getsize(songs[1]) // 4)
        f.write(b"\x7f" * 64)
    assert index_paths(cpu_indexer, songs) == (["duplicate"], 1)  # read again, then matched by its audio
