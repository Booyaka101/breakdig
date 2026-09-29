import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import soundfile as sf
from conftest import InputAsDrums, needs_ffmpeg
from fastapi.testclient import TestClient

from breakdig.web import app as web

pytestmark = needs_ffmpeg


@pytest.fixture
def client(cpu_indexer, fixture_audio, monkeypatch):
    todo, _ = cpu_indexer.pending([str(fixture_audio["mix"])])
    cpu_indexer.index_file(todo[0])
    revealed = []
    monkeypatch.setattr(web, "reveal", revealed.append)
    c = TestClient(web.create_app(cpu_indexer.index.home), base_url="http://127.0.0.1:8765")
    c.revealed = revealed
    return c


def test_page_and_stats(client):
    page = client.get("/")
    assert "<title>breakdig</title>" in page.text
    assert page.headers["x-frame-options"] == "DENY" and page.headers["cache-control"] == "no-cache"
    st = client.get("/api/stats").json()
    assert st["ok"] == 1 and st["bars"] == 16 and st["stems"] == ["drums", "bass", "vocals", "other"]


def test_sections(client):
    r = client.get("/api/sections", params={"only": "drums"}).json()
    assert r["total"] == 1 and r["pattern"] == "drums only"
    (s,) = r["sections"]
    assert (s["first_bar"], s["last_bar"]) == (5, 8)
    assert s["beats"] == [8.0 + 0.5 * i for i in range(16)]
    assert len(s["levels"]) == 4 and all(len(bar) == 4 and bar[0] > -3 > max(bar[1:]) for bar in s["levels"])
    assert client.get("/api/sections", params={"no": "drums"}).json()["total"] == 0
    stricter = {"only": "drums", "cleaner_than": s["clean"] - 1}
    assert client.get("/api/sections", params=stricter).json()["total"] == 0
    for search, no_track in (("fixture wav", False), ("fixture soul", True)):
        r = client.get("/api/sections", params={"no": "drums", "search": search}).json()
        assert r["total"] == 0 and r["no_track_matches"] is no_track


@pytest.mark.parametrize("params, message", [
    ({"only": ""}, "--only takes one to three of"),
    ({"only": "drums", "no": "bass"}, "not both"),
    ({"bpm": "fast"}, "BPM takes a number or a range"),
    ({"sort": "loudness"}, "sort must be one of"),
])
def test_bad_filters(client, params, message):
    r = client.get("/api/sections", params=params)
    assert r.status_code == 400 and message in r.json()["detail"]


def test_clip_is_range_capable(client):
    r = client.get("/api/clip/1:5-8.wav")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    size = len(r.content)
    assert sf.info(io.BytesIO(r.content)).duration == pytest.approx(8.0, abs=0.005)
    r = client.get("/api/clip/1:5-8.wav", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and r.headers["content-range"] == f"bytes 100-199/{size}"
    assert len(r.content) == 100
    assert client.get("/api/clip/1:5-99.wav").status_code == 404
    assert client.get("/api/clip/7:1-2.wav").status_code == 404


def test_export_and_reveal(client, tmp_path):
    out = tmp_path / "exports"
    r = client.post("/api/export", json={"ids": ["1:5-8"], "out": str(out), "sp404": "mk2"}).json()
    (res,) = r["results"]
    assert res["path"].endswith("BRK_0001.WAV") and res["error"] is None
    assert client.post("/api/reveal", json={"path": res["path"]}).json() == {"ok": True}
    assert client.revealed == [res["path"]]
    again = client.post("/api/export", json={"ids": ["1:5-8"], "out": str(out), "sp404": "mk2"}).json()
    assert again["results"][0]["skipped"]


def test_isolated_preview_and_export(client, tmp_path, monkeypatch):
    sep = InputAsDrums()
    monkeypatch.setattr(web, "load_separator", lambda: sep)
    mix = client.get("/api/clip/1:5-8.wav").content
    drums = client.get("/api/clip/1:5-8.wav", params={"keep": "drums"}).content
    assert drums != mix and len(sep.peaks) == 1
    assert client.get("/api/clip/1:5-8.wav", params={"keep": "drums"}).content == drums  # cached
    silent = sf.read(io.BytesIO(client.get("/api/clip/1:5-8.wav", params={"keep": "bass"}).content))[0]
    assert abs(silent).max() == 0
    r = client.post("/api/export", json={"ids": ["1:5-8"], "out": str(tmp_path), "keep": "drums"}).json()
    assert r["results"][0]["path"].endswith(" - drums - 4 bars - 120 BPM - 00.08.wav")
    assert len(sep.peaks) == 3
    for bad in ("drums,bass,vocals,other", "kazoo"):
        assert "keep takes one to three of" in client.get("/api/clip/1:5-8.wav", params={"keep": bad}).text


def test_export_errors(client, tmp_path):
    assert client.post("/api/export", json={"ids": [], "out": str(tmp_path)}).status_code == 400
    for out in ("  ", '""', "exports"):
        assert client.post("/api/export", json={"ids": ["1:5-8"], "out": out}).status_code == 400
    (tmp_path / "file").write_text("x")
    r = client.post("/api/export", json={"ids": ["1:5-8"], "out": str(tmp_path / "file")})
    assert r.status_code == 400 and r.json()["detail"].startswith(f"{tmp_path / 'file'} is a file")


def test_a_stale_pick_fails_on_its_own(client, tmp_path):
    r = client.post("/api/export", json={"ids": ["9:1-4", "1:5-8"], "out": f'"{tmp_path}"'}).json()
    stale, ok = r["results"]
    assert stale["id"] == "9:1-4" and stale["error"] and ok["path"]
    assert r["out"] == str(tmp_path.resolve())


def test_out_folder_under_home(client, tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    (res,) = client.post("/api/export", json={"ids": ["1:5-8"], "out": "~/exports"}).json()["results"]
    assert res["path"].startswith(str(tmp_path / "exports"))


def test_reveal_after_a_restart(client, cpu_indexer, tmp_path, monkeypatch):
    r = client.post("/api/export", json={"ids": ["1:5-8"], "out": str(tmp_path)}).json()
    path = r["results"][0]["path"]
    restarted = TestClient(web.create_app(cpu_indexer.index.home), base_url="http://127.0.0.1:8765")
    assert restarted.post("/api/reveal", json={"path": path}).status_code == 200
    (tmp_path / "not ours.wav").write_bytes(open(path, "rb").read()[:44])
    assert restarted.post("/api/reveal", json={"path": str(tmp_path / "not ours.wav")}).status_code == 403


def test_reveal_refuses_unknown_paths(client):
    r = client.post("/api/reveal", json={"path": "C:/Windows/win.ini"})
    assert r.status_code == 403 and client.revealed == []


def test_posts_must_be_json(client):
    r = client.post("/api/export", content="ids=1:5-8&out=x",
                    headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 422


@pytest.mark.parametrize("headers", [
    {"sec-fetch-site": "cross-site"},
    {"sec-fetch-site": "same-site"},
    {"origin": "http://attacker.example"},
    {"origin": "null"},
])
def test_requests_from_other_sites_are_refused(client, tmp_path, headers):
    # No content-type: older FastAPI parses such a body as JSON, and a page can send it without CORS.
    r = client.post("/api/export", content=f'{{"ids": ["1:5-8"], "out": "{tmp_path.as_posix()}/x"}}',
                    headers=headers)
    assert r.status_code == 403 and not (tmp_path / "x").exists()
    assert client.get("/api/clip/1:5-8.wav", headers=headers).status_code == 403


def test_own_page_is_allowed(client):
    headers = {"sec-fetch-site": "same-origin", "origin": "http://127.0.0.1:8765"}
    assert client.get("/api/stats", headers=headers).status_code == 200


def test_concurrent_requests_for_one_clip(client):
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda i: client.get(f"/api/clip/1:{5 + i % 2}-8.wav").status_code, range(16)))
    assert codes == [200] * 16


def test_a_slow_clip_does_not_hold_up_another(client, monkeypatch):
    started, release, cut = threading.Event(), threading.Event(), web.cut

    def slow_for_bar_5(path, start, end, *isolating):
        if start == 8.0:
            started.set()
            release.wait(10)
        return cut(path, start, end, *isolating)
    monkeypatch.setattr(web, "cut", slow_for_bar_5)
    with ThreadPoolExecutor(2) as pool:
        slow = pool.submit(client.get, "/api/clip/1:5-8.wav")
        assert started.wait(10)
        try:
            other = pool.submit(client.get, "/api/clip/1:6-8.wav").result(timeout=5)
        finally:
            release.set()
        assert other.status_code == 200 and slow.result().status_code == 200


@pytest.mark.skipif(os.name != "nt", reason="windows paths")
def test_reveal_passes_the_path_as_its_own_argument(monkeypatch):
    calls = []
    monkeypatch.setattr(web.sys, "platform", "win32")
    monkeypatch.setattr(web.subprocess, "Popen", calls.append)
    web.reveal("D:/Samples/A - B - 2 bars.wav")
    assert calls == [["explorer", "/select,", r"D:\Samples\A - B - 2 bars.wav"]]


def test_foreign_host_header_is_refused(cpu_indexer):
    c = TestClient(web.create_app(cpu_indexer.index.home), base_url="http://attacker.example")
    assert c.get("/api/stats").status_code == 400


def test_ipv6_loopback_is_allowed(cpu_indexer):
    c = TestClient(web.create_app(cpu_indexer.index.home, host="::1"), base_url="http://[::1]:8765")
    assert c.get("/api/stats").status_code == 200


def test_clip_cache_is_pruned(tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    for i in range(5):
        p = folder / f"{i}.wav"
        p.write_bytes(b"x" * 10)
        os.utime(p, (i, i))
    web.prune(folder, 25)
    assert sorted(p.name for p in folder.iterdir()) == ["3.wav", "4.wav"]
    web.prune(tmp_path / "none yet", 25)


def test_each_new_clip_prunes_the_cache(client, cpu_indexer, monkeypatch):
    old = cpu_indexer.index.home / "cache" / "clips" / "old.wav"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"x")
    monkeypatch.setattr(web, "CLIP_CACHE_BYTES", 0)
    assert client.get("/api/clip/1:5-8.wav").status_code == 200 and not old.exists()


def test_silence_threshold_is_bounded(client):
    assert client.get("/api/sections", params={"silence_db": 20}).status_code == 200
    assert client.get("/api/sections", params={"silence_db": 0}).status_code == 422
