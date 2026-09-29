import json
import socket
import sqlite3
import sys

import pytest
import uvicorn
from conftest import GridBeats, InputAsDrums, TrueStems, needs_ffmpeg

from breakdig import cli, indexer
from breakdig import export as export_module
from breakdig.db import Index

pytestmark = needs_ffmpeg


def run(capsys, *argv):
    """Run the CLI; returns (exit code, stdout, stderr)."""
    code = 0
    try:
        cli.main(list(argv))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture(autouse=True)
def no_server(monkeypatch):
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: pytest.fail("started a server"))


@pytest.fixture
def stub_models(monkeypatch, fixture_audio):
    class StubIndexer(indexer.Indexer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._sep = TrueStems(fixture_audio)
            self._beats = GridBeats()
    monkeypatch.setattr(indexer, "Indexer", StubIndexer)


def test_index_find_export_stats(capsys, tmp_path, fixture_audio, stub_models, monkeypatch):
    home = str(tmp_path / "home")
    code, out, _ = run(capsys, "--home", home, "index", str(fixture_audio["mix"].parent))
    assert code == 0
    # The fixture folder also holds the four stem files, which are real audio too.
    assert "0 already indexed, 5 to do" in out
    assert "Done in" in out

    code, out, _ = run(capsys, "--home", home, "index", str(fixture_audio["mix"].parent))
    assert "5 already indexed, 0 to do" in out and "nothing new" in out

    code, out, _ = run(capsys, "--home", home, "find", "--only", "drums", "--search", "fixture.wav")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["ID", "ARTIST", "-", "TITLE", "BARS", "START", "END", "BPM", "CLEAN"]
    assert ":5-8" in lines[2] and "0:08.000" in lines[2] and "0:16.000" in lines[2] and "120.0" in lines[2]
    assert "1 section(s)" in out

    code, out, _ = run(capsys, "--home", home, "find", "--search", "no such track")
    assert code == 0 and "No indexed track matches --search 'no such track'." in out
    code, out, _ = run(capsys, "--home", home, "find", "--only", "vocals", "--search", "fixture WAV")
    assert code == 0 and "No matching sections. Try --min-bars 1 or --silence-db 25." in out

    code, out, _ = run(capsys, "--home", home, "find", "--search", "fixture.wav", "--json")
    (hit,) = json.loads(out)
    assert hit["first_bar"] == 5 and hit["bars"] == 4

    for limit, n in ((hit["clean"], 1), (hit["clean"] - 1, 0)):
        code, out, _ = run(capsys, "--home", home, "find", "--search", "fixture.wav", "--json",
                           "--cleaner-than", str(limit))
        assert len(json.loads(out)) == n

    code, out, _ = run(capsys, "--home", home, "find", "--pick", hit["id"], "--json")
    (picked,) = json.loads(out)  # strict JSON: a picked section has no score, so clean is null
    assert picked["start"] == hit["start"] and picked["clean"] is None

    code, out, _ = run(capsys, "--home", home, "export", "--pick", hit["id"], "--out", str(tmp_path / "out"))
    assert code == 0 and "wrote" in out and "1 file(s)" in out
    assert len(list((tmp_path / "out").glob("*.wav"))) == 1
    monkeypatch.setattr(export_module, "load_separator", InputAsDrums)
    code, out, _ = run(capsys, "--home", home, "export", "--pick", hit["id"], "--keep", "bass,drums",
                       "--out", str(tmp_path / "out"))
    assert code == 0 and " - drums+bass - 4 bars - " in out

    (tmp_path / "a file").write_text("")
    code, _, err = run(capsys, "--home", home, "export", "--pick", hit["id"], "--out", str(tmp_path / "a file"))
    assert code != 0 and err.startswith("breakdig: ") and "a file" in err

    code, out, _ = run(capsys, "--home", home, "stats", "--failures")
    assert "indexed" in out and "Speed:" in out


def test_empty_index_and_no_results(capsys, tmp_path):
    Index(tmp_path).close()
    code, out, _ = run(capsys, "--home", str(tmp_path), "find")
    assert code == 0 and f"The index at {tmp_path} is empty. Run: breakdig index <folder>" in out
    code, again, _ = run(capsys, "find", "--home", str(tmp_path))
    assert again == out
    code, out, _ = run(capsys, "--home", str(tmp_path), "export", "--out", str(tmp_path / "o"))
    assert "Nothing to export. The index at" in out


def test_empty_index_with_failures_says_how_to_retry(capsys, tmp_path):
    with Index(tmp_path) as index:
        index.save({"key": "k", "path": "x.mp3", "status": "failed", "error": "separation failed"})
    code, out, _ = run(capsys, "--home", str(tmp_path), "find")
    assert "1 file(s) failed; see breakdig stats --failures" in out and "--retry-failed" in out


@pytest.mark.parametrize("cmd", [["find"], ["stats"], ["export", "--out", "o"], ["ui", "--no-browser"]])
def test_a_mistyped_home_is_not_an_empty_index(capsys, tmp_path, cmd):
    code, _, err = run(capsys, "--home", str(tmp_path / "typo"), *cmd)
    assert code != 0 and "there is no index at" in err and not (tmp_path / "typo").exists()


@pytest.mark.parametrize("argv, message", [
    (["find", "--bpm", "fast"], "--bpm takes a number or a range"),
    (["find", "--only", "guitar"], "--only takes one to three of"),
    (["find", "--only", ""], "--only takes one to three of"),
    (["find", "--only", "drums", "--no", "bass"], "use --only or --no, not both"),
    (["find", "--silence-db", "0"], "--silence-db takes 1 to 60"),
    (["find", "--limit", "0"], "must be 1 or more"),
    (["find", "--min-bars", "-1"], "must be 1 or more"),
    (["find", "--min-bars", "two"], "expected a whole number, got 'two'"),
    (["ui", "--port", "70000"], "ports go up to 65535"),
    (["index", r'D:\My Music" --exclude *.tmp'], "has a quote in it"),
    (["export", "--pick", "banana", "--out", "x"], "section ids look like 12:5-8"),
    (["export", "--pick", "99:1-4", "--out", "x"], "no indexed track with id 99"),
    (["export", "--keep", "drums,kazoo", "--out", "x"], "--keep takes one to three of"),
    (["export", "--sp404", "sp1200", "--out", "x"], "invalid choice: 'sp1200'"),
    (["index", "Z:/definitely/not/here"], "not found: Z:/definitely/not/here"),
])
def test_bad_input_gives_a_clear_message(capsys, tmp_path, argv, message):
    Index(tmp_path).close()
    code, _, err = run(capsys, "--home", str(tmp_path), *argv)
    assert code != 0 and message in err


def test_model_that_will_not_load_stops_before_any_file(capsys, tmp_path, fixture_audio, monkeypatch):
    def broken(*a, **kw):
        raise ValueError("Model file htdemucs_typo.yaml not found")
    monkeypatch.setattr(indexer, "Separator", broken)
    home = tmp_path / "home"
    code, _, err = run(capsys, "--home", str(home), "index", str(fixture_audio["mix"]), "--model", "htdemucs_typo.yaml")
    assert code != 0 and "cannot load the models" in err and "htdemucs_typo.yaml" in err
    code, out, _ = run(capsys, "--home", str(home), "stats")
    assert "0 indexed" in out


@pytest.mark.parametrize("stage, error, code, message", [
    ("BeatTracker", RuntimeError("cannot download final0.ckpt"), 1, "cannot load the models"),
    ("Separator", KeyboardInterrupt(), 130, "Stopped."),  # during the first model download
])
def test_models_load_before_any_file(capsys, tmp_path, fixture_audio, monkeypatch,
                                     stage, error, code, message):
    def broken(*a, **kw):
        raise error
    monkeypatch.setattr(indexer, "Separator", lambda *a, **kw: object())
    monkeypatch.setattr(indexer, "BeatTracker", lambda *a, **kw: object())
    monkeypatch.setattr(indexer, stage, broken)
    home = tmp_path / "home"
    got, out, err = run(capsys, "--home", str(home), "index", str(fixture_audio["mix"]))
    assert got == code and message in out + err
    assert "0 indexed" in run(capsys, "--home", str(home), "stats")[1]


def test_a_database_error_mid_run_says_what_was_kept(capsys, tmp_path, fixture_audio, stub_models,
                                                      monkeypatch):
    real = indexer.Indexer.index_file

    def fills_up(self, f):
        if self.index.stats()["ok"]:
            raise sqlite3.OperationalError("database or disk is full")
        return real(self, f)
    monkeypatch.setattr(indexer.Indexer, "index_file", fills_up)
    code, _, err = run(capsys, "--home", str(tmp_path), "index", str(fixture_audio["mix"]),
                       str(fixture_audio["drums"]))
    assert code == 1 and err.splitlines()[-1] == ("breakdig: database or disk is full. 1 tracks were "
                                                  "committed; run the same command again to carry on.")


def test_a_damaged_index_file_gives_a_message(capsys, tmp_path):
    (tmp_path / "index.sqlite").write_bytes(b"not sqlite" * 100)
    code, _, err = run(capsys, "--home", str(tmp_path), "find")
    assert code == 1 and err.startswith("breakdig: file is not a database")


def test_folder_without_audio(capsys, tmp_path):
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "readme.txt").write_text("hi")
    code, out, _ = run(capsys, "--home", str(tmp_path / "h"), "index", str(tmp_path / "music"))
    assert code == 0 and "No audio files found" in out
    # As cmd.exe passes "C:\...\music\\" (the backslash escapes the closing quote).
    code, out, _ = run(capsys, "--home", str(tmp_path / "h"), "index", str(tmp_path / "music") + '"')
    assert code == 0 and "No audio files found" in out


def test_missing_ffmpeg(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.audio, "have_ffmpeg", lambda: False)
    code, _, err = run(capsys, "--home", str(tmp_path), "index", str(tmp_path))
    assert code != 0 and "ffmpeg" in err and "winget install" in err


def test_ui_port_in_use(capsys, tmp_path):
    with socket.create_server(("127.0.0.1", 0)) as held:
        port = held.getsockname()[1]
        Index(tmp_path).close()
        code, _, err = run(capsys, "ui", "--home", str(tmp_path), "--port", str(port), "--no-browser")
    assert code != 0 and f"port {port} is in use" in err


def test_clock_rounds_before_splitting_minutes():
    assert cli._clock(59.9996) == "1:00.000"
    assert cli._clock(67.6) == "1:07.600"


def test_version(capsys):
    code, out, _ = run(capsys, "--version")
    assert code == 0 and out.strip() == "breakdig 0.1.0"
