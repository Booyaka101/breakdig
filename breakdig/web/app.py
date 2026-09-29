"""Local web UI: filter sections, audition them, export the keepers."""

import os
import subprocess
import sys
import threading
import weakref
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .. import __version__, audio
from ..db import STEMS, Index
from ..export import cut, export, exported_as, load_separator, write_wav
from ..isolate import stems_name
from ..query import SILENT_DB, SORTS, find, get_section, keep_from, parse_bpm, pattern_from, search_tracks

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = ["127.0.0.1", "localhost", "::1"]
CLIP_CACHE_BYTES = 500 * 2**20


class ExportRequest(BaseModel):
    ids: list[str]
    out: str
    sp404: str | None = None
    keep: str | None = None


class RevealRequest(BaseModel):
    path: str


def default_out() -> Path:
    music = Path.home() / "Music"
    return (music if music.is_dir() else Path.home()) / "breakdig"


def reveal(path: str) -> None:
    if sys.platform == "win32":
        # explorer exits 1 even when it worked, so there is no status to check. The path must be
        # its own argument: quoted together with /select, (as a path with spaces is) it is ignored.
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])


def parse_keep(text: str | None) -> frozenset:
    """The stems to keep, as 'drums' or 'bass+drums', or none for the whole mix."""
    try:
        return keep_from(text)
    except ValueError as e:
        raise HTTPException(400, str(e).replace("--keep", "keep")) from None


def prune(folder: Path, keep_bytes: int) -> None:
    """Delete the oldest preview clips until the rest fit in keep_bytes."""
    files = []
    for p in folder.glob("*.wav"):
        try:
            files.append((p.stat(), p))
        except OSError:
            pass  # another request pruned it meanwhile
    total = 0
    for st, p in sorted(files, key=lambda f: f[0].st_mtime, reverse=True):
        total += st.st_size
        if total > keep_bytes:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass  # being served right now; it goes next time


class SharedSeparator:
    """Loaded on first use, and separating one section at a time so two cannot fill the GPU."""

    def __init__(self):
        self._sep, self._lock = None, threading.Lock()

    def separate(self, *args, **kwargs):
        with self._lock:
            if self._sep is None:
                self._sep = load_separator()
            return self._sep.separate(*args, **kwargs)


def create_app(home=None, host: str = "127.0.0.1") -> FastAPI:
    app = FastAPI(title="breakdig", version=__version__, docs_url=None, redoc_url=None)
    # Refuse requests addressed to any other name, so a web page cannot reach this
    # server through DNS rebinding. Binding to another interface is an explicit opt-out.
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=[*LOCAL_HOSTS, "[::1]"] if host in LOCAL_HOSTS else ["*"])

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        # A page on any other site can send simple requests here without a CORS preflight.
        origin = request.headers.get("origin")
        if request.url.path.startswith("/api/") and (
                request.headers.get("sec-fetch-site") in ("cross-site", "same-site")
                or (origin and origin.split("://", 1)[-1] != request.headers.get("host"))):
            return JSONResponse({"detail": "requests from other sites are refused"}, status_code=403)
        return await call_next(request)

    clips = Index.home_for(home) / "cache" / "clips"
    prune(clips, CLIP_CACHE_BYTES)
    exported: set[str] = set()
    # Two tabs exporting to one SP-404 folder would otherwise pick the same BRK numbers.
    export_lock = threading.Lock()
    # Two requests for one uncached clip would write the same file; other clips need not wait.
    clip_locks, clip_locks_guard = weakref.WeakValueDictionary(), threading.Lock()
    separator = SharedSeparator()

    def open_index() -> Index:
        return Index(home)

    @app.get("/")
    def page():
        # No framing, so another site cannot overlay it and trick clicks onto Export.
        # no-cache, so the page from an older version is not run against a newer API.
        return FileResponse(STATIC / "index.html", headers={
            "X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'",
            "Cache-Control": "no-cache"})

    @app.get("/api/stats")
    def stats():
        with open_index() as index:
            st = index.stats()
            return {**st, "home": str(index.home), "version": __version__, "stems": STEMS,
                    "sorts": sorted(SORTS), "default_out": str(default_out())}

    @app.get("/api/sections")
    def sections(only_: str | None = Query(None, alias="only"), no: str | None = None,
                 min_bars: int = Query(2, ge=1), bpm: str | None = None, search: str | None = None,
                 skip_seams: bool = False, sort: str = "clean", limit: int = Query(500, ge=1),
                 silence_db: float = Query(SILENT_DB, ge=1, le=60), cleaner_than: float | None = None):
        if sort not in SORTS:
            raise HTTPException(400, f"sort must be one of {', '.join(sorted(SORTS))}")
        try:
            pattern = pattern_from(only_, no)
            bpm_range = parse_bpm(bpm)
        except ValueError as e:
            raise HTTPException(400, str(e).replace("--bpm", "BPM")) from None
        with open_index() as index:
            found = find(index, pattern, min_bars=min_bars, bpm=bpm_range, text=search,
                         skip_seams=skip_seams, sort=sort, silent_db=silence_db, cleaner_than=cleaner_than)
            no_track = not found and bool(search) and not search_tracks(index.ok_tracks(), search)
        return {"pattern": pattern.name, "total": len(found), "no_track_matches": no_track,
                "sections": [s.to_dict() for s in found[:limit]]}

    @app.get("/api/clip/{section_id}.wav")
    def clip(section_id: str, keep: str | None = None):
        keep = parse_keep(keep)
        with open_index() as index:
            try:
                s = get_section(index, section_id)
            except ValueError as e:
                raise HTTPException(404, str(e)) from None
        # Keyed by content, not track id, so a re-indexed track never serves a stale clip.
        dest = clips / f"{s.key[:16]}_{s.first_bar}-{s.last_bar}{'_' + stems_name(keep) if keep else ''}.wav"
        with clip_locks_guard:
            lock = clip_locks.setdefault(dest, threading.Lock())
        with lock:
            if not dest.exists():
                if not os.path.exists(s.path):
                    raise HTTPException(404, f"source file is gone: {s.path}")
                try:
                    c = cut(s.path, s.start, s.end, keep, separator)
                except (audio.DecodeError, ValueError, RuntimeError) as e:
                    raise HTTPException(500, str(e)) from None
                clips.mkdir(parents=True, exist_ok=True)
                prune(clips, CLIP_CACHE_BYTES)
                write_wav(c, dest, s)
        return FileResponse(dest, media_type="audio/wav")

    @app.post("/api/export")
    def do_export(req: ExportRequest):
        keep = parse_keep(req.keep)
        if not req.ids:
            raise HTTPException(400, "nothing selected")
        # As pasted from Explorer's "Copy as path", or typed the way a shell would take it.
        out = os.path.expandvars(os.path.expanduser(req.out.strip().strip('"')))
        if not out:
            raise HTTPException(400, "choose an output folder")
        if not os.path.isabs(out):
            raise HTTPException(400, f"use a full folder path, like {default_out()}")
        picked, failed = [], []
        with open_index() as index:
            for i in req.ids:
                try:
                    picked.append(get_section(index, i))
                except ValueError as e:  # re-indexed since the page listed it
                    failed.append({"id": i, "path": None, "error": str(e), "skipped": False})
        try:
            with export_lock:
                results = export(picked, out, sp404=req.sp404, keep=keep, separator=separator)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except OSError as e:
            raise HTTPException(400, f"cannot write to {out}: {e}") from None
        exported.update(r.path for r in results if r.path)
        return {"out": str(Path(out).resolve()),
                "results": failed + [{"id": r.section.id, "path": r.path, "error": r.error,
                                      "skipped": r.skipped} for r in results]}

    @app.post("/api/reveal")
    def do_reveal(req: RevealRequest):
        # Only files breakdig knows about, so this endpoint cannot be used to probe the disk.
        known = req.path in exported
        if not known:
            with open_index() as index:
                known = index.by_path(req.path) is not None
        if not known and not exported_as(Path(req.path)):  # exported before a restart
            raise HTTPException(403, "not a file breakdig indexed or exported")
        if not os.path.exists(req.path):
            raise HTTPException(404, f"file is gone: {req.path}")
        reveal(req.path)
        return {"ok": True}

    return app
