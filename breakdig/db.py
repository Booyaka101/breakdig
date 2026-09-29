"""SQLite index: one row per track, one row per bar."""

import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "vocals", "other")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    artist TEXT,
    title TEXT,
    album TEXT,
    duration REAL,
    bpm REAL,
    status TEXT NOT NULL,
    error TEXT,
    sample_rate INTEGER,
    channels INTEGER,
    chunk_seconds REAL,
    dup_of INTEGER,
    beats BLOB,
    seconds REAL,
    size INTEGER,
    mtime_ns INTEGER,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bars (
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    drums_db REAL, bass_db REAL, vocals_db REAL, other_db REAL, mix_db REAL,
    PRIMARY KEY (track_id, idx)
);
CREATE INDEX IF NOT EXISTS tracks_status ON tracks(status);
CREATE INDEX IF NOT EXISTS tracks_duration ON tracks(duration);
CREATE INDEX IF NOT EXISTS tracks_path ON tracks(path);
CREATE TABLE IF NOT EXISTS copies (
    path TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    duration REAL,
    size INTEGER,
    mtime_ns INTEGER,
    of TEXT,
    of_size INTEGER,
    of_mtime_ns INTEGER
);
"""


def _unlink(path: Path) -> None:
    # A profile is only used to spot duplicates, and another program may have it open.
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _copy(src: Path, dst: Path) -> None:
    try:
        shutil.copyfile(src, dst)
    except OSError:
        pass


def _data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".local", "share")
    return Path(base) / "breakdig"


def default_home() -> Path:
    env = os.environ.get("BREAKDIG_HOME")
    return Path(env) if env else _data_dir()


def model_dir() -> Path:
    """Where separation models are cached. Shared by every index, since they are 300+ MB."""
    env = os.environ.get("BREAKDIG_MODELS")
    return Path(env) if env else _data_dir() / "models"


class Index:
    @staticmethod
    def home_for(home: Path | str | None) -> Path:
        return Path(home).resolve() if home else default_home()

    def __init__(self, home: Path | str | None = None):
        self.home = self.home_for(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.profiles = self.home / "profiles"
        self.profiles.mkdir(exist_ok=True)
        # The web UI serves requests from a threadpool; each request gets its own Index.
        self.conn = sqlite3.connect(self.home / "index.sqlite", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(tracks)")}
        for col in ("size INTEGER", "mtime_ns INTEGER"):  # added after the first 0.1.0 builds
            if col.split()[0] not in have:
                self.conn.execute(f"ALTER TABLE tracks ADD COLUMN {col}")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def by_key(self, key: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM tracks WHERE key = ?", (key,)).fetchone()

    def by_path(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM tracks WHERE path = ?", (path,)).fetchone()

    def track(self, track_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()

    def copy_at(self, path: str) -> sqlite3.Row | None:
        """A file without a row of its own, last found byte-identical to the file at copies.of."""
        return self.conn.execute("SELECT * FROM copies WHERE path = ?", (path,)).fetchone()

    def save_copy(self, copy: dict):
        with self.conn:
            self.conn.execute(f"INSERT OR REPLACE INTO copies ({', '.join(copy)}) "
                              f"VALUES ({', '.join('?' * len(copy))})", list(copy.values()))

    def keys_at(self, path: str) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT key FROM tracks WHERE path = ?", (path,))]

    def _set(self, track_id: int, fields: dict):
        self.conn.execute(f"UPDATE tracks SET {', '.join(f'{c} = ?' for c in fields)} WHERE id = ?",
                          [*fields.values(), track_id])

    def update(self, track_id: int, **fields):
        with self.conn:
            self._set(track_id, fields)

    def rekey(self, track_id: int, fields: dict):
        """Move a track to a new key, keeping its analysis. fields must include key."""
        old = self.track(track_id)["key"]
        # Copied, and the old one removed after the commit, so a crash between leaves the
        # profile that the committed row names.
        if fields["key"] != old:
            _copy(self.profile_path(old), self.profile_path(fields["key"]))
        with self.conn:
            # A row under the new key is an earlier failed try at the same file.
            self.conn.execute("DELETE FROM tracks WHERE key = ? AND id != ?", (fields["key"], track_id))
            self._set(track_id, fields)
        if fields["key"] != old:
            _unlink(self.profile_path(old))

    def forget(self, key: str):
        with self.conn:
            self.conn.execute("DELETE FROM tracks WHERE key = ?", (key,))
        _unlink(self.profile_path(key))

    def sweep_profiles(self):
        """Delete profiles that no track has, left by an interrupted run or a file that was open."""
        keys = {r[0] for r in self.conn.execute("SELECT key FROM tracks")}
        for p in self.profiles.glob("*.npz"):
            if p.stem not in keys:
                _unlink(p)

    def save(self, track: dict, bars: list[tuple] = ()) -> int:
        """Insert or replace a track and its bars in one transaction.

        The commit is what marks a track as done, so an interrupted run redoes
        at most the file it was working on."""
        cols = [c for c in track if c != "id"]
        # Another key at this path is what used to be in the file before it was overwritten.
        stale = [r[0] for r in self.conn.execute(
            "SELECT key FROM tracks WHERE path = ? AND key != ?", (track["path"], track["key"]))]
        with self.conn:
            # A new id above the rows about to go, so an old section id never names the new audio.
            tid = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM tracks").fetchone()[0]
            self.conn.execute("DELETE FROM tracks WHERE key = ?", (track["key"],))
            self.conn.executemany("DELETE FROM tracks WHERE key = ?", [(k,) for k in stale])
            self.conn.execute(
                f"INSERT INTO tracks (id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                [tid, *(track[c] for c in cols)])
            self.conn.executemany(
                'INSERT INTO bars (track_id, idx, start, "end", drums_db, bass_db, vocals_db, other_db, mix_db) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [(tid, *b) for b in bars])
        for k in stale:
            _unlink(self.profile_path(k))
        return tid

    def candidates_near(self, duration: float, tolerance: float = 1.0) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, key, path, duration FROM tracks WHERE status IN ('ok', 'no_grid') "
            "AND duration BETWEEN ? AND ?", (duration - tolerance, duration + tolerance)).fetchall()

    def profile_path(self, key: str) -> Path:
        return self.profiles / f"{key}.npz"

    def ok_tracks(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM tracks WHERE status = 'ok' ORDER BY id").fetchall()

    def bars_for(self, track_ids: list[int] | None = None) -> dict[int, np.ndarray]:
        """Bars grouped by track as float arrays with columns
        start, end, drums, bass, vocals, other, mix."""
        sql = ('SELECT track_id, start, "end", drums_db, bass_db, vocals_db, other_db, mix_db '
               "FROM bars")
        args: tuple = ()
        if track_ids is not None:
            sql += f" WHERE track_id IN ({', '.join('?' * len(track_ids))})"
            args = tuple(track_ids)
        rows = self.conn.execute(sql + " ORDER BY track_id, idx", args).fetchall()
        if not rows:
            return {}
        a = np.array(rows, dtype=np.float64)
        ids = a[:, 0].astype(np.int64)
        cuts = np.flatnonzero(np.diff(ids)) + 1
        return {int(chunk[0, 0]): chunk[:, 1:] for chunk in np.split(a, cuts)}

    def stats(self) -> dict:
        by_status = dict(self.conn.execute(
            "SELECT status, COUNT(*) FROM tracks GROUP BY status").fetchall())
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration), 0), COALESCE(SUM(seconds), 0) "
            "FROM tracks WHERE status = 'ok'").fetchone()
        nbars = self.conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        return {"by_status": by_status, "ok": row[0], "audio_seconds": row[1],
                "index_seconds": row[2], "bars": nbars}

    def failures(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path, status, error FROM tracks WHERE status NOT IN ('ok', 'duplicate') "
            "ORDER BY status, path"
        ).fetchall()
