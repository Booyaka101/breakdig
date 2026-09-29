"""Activity patterns and the section finder.

A stem is ACTIVE in a bar if its level is at least the mix level minus 12 dB.
It is SILENT if it sits more than 30 dB below the loudest active stem in that
bar, or below -90 dBFS. Silent wins, so a bar of digital silence has no active stems at all.
The 30 dB is the silence threshold; Demucs bleed often sits 20-30 dB down, so
lowering it trades purity for more hits.
"""

import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np

from .db import STEMS, Index

ACTIVE_BELOW_MIX_DB = 12.0
SILENT_DB = 30.0
INAUDIBLE_DB = -90.0

# Column layout of the per-track bar arrays from Index.bars_for.
START, END = 0, 1
LEVELS = slice(2, 6)
MIX = 6


@dataclass(frozen=True)
class Pattern:
    active: frozenset
    silent: frozenset

    @property
    def name(self) -> str:
        if self.active:
            return "+".join(s for s in STEMS if s in self.active) + " only"
        return "no " + "+".join(s for s in STEMS if s in self.silent)


def _some_stems(stems, flag: str) -> frozenset:
    stems = frozenset(stems)
    if not stems or stems - set(STEMS) or stems == set(STEMS):
        raise ValueError(f"{flag} takes one to three of: {', '.join(STEMS)}")
    return stems


def only(stems) -> Pattern:
    stems = _some_stems(stems, "--only")
    return Pattern(stems, frozenset(STEMS) - stems)


def without(stems) -> Pattern:
    return Pattern(frozenset(), _some_stems(stems, "--no"))


PRESETS = {
    "drums": only({"drums"}),
    "vocals": only({"vocals"}),
    "bass+drums": only({"bass", "drums"}),
    "no-drums": without({"drums"}),
}


def parse_stems(text: str) -> set[str]:
    return {s.strip().lower() for s in text.replace("+", ",").split(",") if s.strip()}


def keep_from(text: str | None) -> frozenset:
    """The --keep stems, or none for the whole mix."""
    return _some_stems(parse_stems(text), "--keep") if text else frozenset()


def pattern_from(only_text: str | None = None, no_text: str | None = None) -> Pattern:
    """The --only / --no filter as typed; drums only when neither is given."""
    if only_text is not None and no_text is not None:
        raise ValueError("use --only or --no, not both")
    if no_text is not None:
        return without(parse_stems(no_text))
    return only(parse_stems("drums" if only_text is None else only_text))


def activity(levels: np.ndarray, mix: np.ndarray,
             silent_db: float = SILENT_DB) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (bars, 4) arrays: active and silent, stems in STEMS order."""
    inaudible = levels < INAUDIBLE_DB
    active = (levels >= mix[:, None] - ACTIVE_BELOW_MIX_DB) & ~inaudible
    loudest = np.where(active, levels, -np.inf).max(axis=1)
    silent = inaudible | (levels < (loudest - silent_db)[:, None])
    return active & ~silent, silent


def _cols(stems) -> list[int]:
    return [STEMS.index(s) for s in STEMS if s in stems]


def matches(pattern: Pattern, active: np.ndarray, silent: np.ndarray) -> np.ndarray:
    ok = silent[:, _cols(pattern.silent)].all(axis=1)
    if pattern.active:
        return ok & active[:, _cols(pattern.active)].all(axis=1)
    return ok & active[:, _cols(set(STEMS) - pattern.silent)].any(axis=1)


def cleanliness(pattern: Pattern, levels: np.ndarray) -> np.ndarray:
    """Per bar: loudest unwanted stem minus the target level. More negative is cleaner.

    The target is the quietest required stem, or for a --no pattern the
    loudest of the stems that are allowed to play."""
    unwanted = levels[:, _cols(pattern.silent)].max(axis=1)
    if pattern.active:
        target = levels[:, _cols(pattern.active)].min(axis=1)
    else:
        target = levels[:, _cols(set(STEMS) - pattern.silent)].max(axis=1)
    return unwanted - target


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Maximal runs of True as (first, last) inclusive index pairs."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def crosses_seam(start: float, end: float, duration: float, chunk_seconds: float | None) -> bool:
    """True if a chunked separation joined two chunks strictly inside [start, end]."""
    if not chunk_seconds:
        return False
    seams = np.arange(chunk_seconds, duration, chunk_seconds)
    return bool(((seams > start) & (seams < end)).any())


def bpm_between(beats: np.ndarray, start: float, end: float) -> float | None:
    """60 / median beat interval of the beats inside [start, end]."""
    inside = beats[(beats >= start - 0.03) & (beats <= end + 0.03)]
    if len(inside) < 2:
        return None
    return 60.0 / float(np.median(np.diff(inside)))


@dataclass
class Section:
    track_id: int
    first_bar: int  # 1-based, as a musician counts
    last_bar: int
    start: float
    end: float
    bpm: float | None
    clean: float | None  # None when looked up by id, since there is no pattern to score against
    seam: bool
    artist: str = ""
    title: str = ""
    album: str = ""
    path: str = ""
    key: str = ""  # the track's content key, which survives a move
    beats: list[float] = field(default_factory=list)  # the beats inside it, in seconds
    levels: list[list[float]] = field(default_factory=list)  # per bar, each stem's dB against the mix

    @property
    def bars(self) -> int:
        return self.last_bar - self.first_bar + 1

    @property
    def id(self) -> str:
        return f"{self.track_id}:{self.first_bar}-{self.last_bar}"

    def to_dict(self) -> dict:
        return {**asdict(self), "id": self.id, "bars": self.bars}


def sections_in_track(bars: np.ndarray, beats: np.ndarray, pattern: Pattern, min_bars: int = 2,
                      duration: float = 0.0, chunk_seconds: float | None = None,
                      track_id: int = 0, silent_db: float = SILENT_DB) -> list[Section]:
    """Find matching sections in one track's bar array (columns start, end, 4 stems, mix)."""
    if len(bars) == 0:
        return []
    levels, mix = bars[:, LEVELS], bars[:, MIX]
    active, silent = activity(levels, mix, silent_db)
    # A bar far off the track's usual length is a downbeat the tracker missed or doubled.
    length = bars[:, END] - bars[:, START]
    typical = np.median(length)
    hit = matches(pattern, active, silent) & (length > typical / 2) & (length < typical * 2)
    score = cleanliness(pattern, levels)
    # A section is only as clean as its dirtiest bar.
    return [_section(bars, beats, a, b, duration, chunk_seconds, track_id, float(score[a: b + 1].max()))
            for a, b in runs(hit, min_bars)]


def _section(bars: np.ndarray, beats: np.ndarray, a: int, b: int, duration: float,
             chunk_seconds: float | None, track_id: int, clean: float | None) -> Section:
    """Bars a to b, 0-based and inclusive, of one track."""
    start, end = float(bars[a, START]), float(bars[b, END])
    inside = beats[(beats >= start - 0.03) & (beats < end - 0.03)]
    return Section(track_id, a + 1, b + 1, start, end, bpm_between(beats, start, end), clean,
                   crosses_seam(start, end, duration, chunk_seconds),
                   beats=[round(float(t), 4) for t in inside],
                   levels=np.round(bars[a: b + 1, LEVELS] - bars[a: b + 1, MIX, None], 1).tolist())


def _describe(s: Section, t) -> Section:
    s.artist, s.title, s.album, s.path, s.key = t["artist"], t["title"], t["album"] or "", t["path"], t["key"]
    return s


def _beats(t) -> np.ndarray:
    return np.frombuffer(t["beats"], dtype=np.float64) if t["beats"] else np.zeros(0)


def parse_bpm(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    try:
        if "-" in text:
            lo, hi = (float(x) for x in text.split("-", 1))
        else:
            lo = hi = float(text)
    except ValueError:
        raise ValueError(f"--bpm takes a number or a range like 85-100, got {text!r}") from None
    if lo > hi:
        lo, hi = hi, lo
    # A single number means "about this tempo".
    return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)


def _fold(text: str) -> str:
    # NFC because macOS and some taggers store accents decomposed.
    return unicodedata.normalize("NFC", text).casefold()


def search_tracks(tracks: list, text: str) -> list:
    """Tracks where every word of text is in the artist, title, album or path."""
    words = _fold(text).split()
    return [t for t in tracks
            if all(any(w in _fold(t[c] or "") for c in ("artist", "title", "album", "path")) for w in words)]


SORTS = {
    "clean": lambda s: (s.clean, -s.bars),
    "bars": lambda s: (-s.bars, s.clean),
    "bpm": lambda s: (s.bpm if s.bpm is not None else 1e9, s.clean),
    "artist": lambda s: (s.artist.lower(), s.title.lower(), s.album.lower(), s.track_id, s.start),
}


def find(index: Index, pattern: Pattern, min_bars: int = 2, bpm: tuple[float, float] | None = None,
         text: str | None = None, skip_seams: bool = False, sort: str = "clean",
         limit: int | None = None, silent_db: float = SILENT_DB,
         cleaner_than: float | None = None) -> list[Section]:
    if not 1 <= silent_db <= 60:
        raise ValueError(f"--silence-db takes 1 to 60, got {silent_db:g}")
    tracks = index.ok_tracks()
    if text:
        tracks = search_tracks(tracks, text)
    if not tracks:
        return []
    all_bars = index.bars_for()
    found = []
    for t in tracks:
        bars = all_bars.get(t["id"])
        if bars is None:
            continue
        for s in sections_in_track(bars, _beats(t), pattern, min_bars, t["duration"] or 0.0,
                                   t["chunk_seconds"], t["id"], silent_db):
            if bpm and (s.bpm is None or not bpm[0] <= s.bpm <= bpm[1]):
                continue
            if skip_seams and s.seam:
                continue
            if cleaner_than is not None and s.clean > cleaner_than:
                continue
            found.append(_describe(s, t))
    found.sort(key=SORTS[sort])
    return found[:limit] if limit else found


def get_section(index: Index, section_id: str) -> Section:
    """Look up a section by its '<track>:<first>-<last>' id, bars counted from 1."""
    try:
        tid, span = section_id.split(":")
        first, last = (int(x) for x in span.split("-"))
        tid = int(tid)
    except ValueError:
        raise ValueError(f"section ids look like 12:5-8, got {section_id!r}") from None
    t = index.track(tid)
    if t is None or t["status"] != "ok":
        raise ValueError(f"no indexed track with id {tid}")
    bars = index.bars_for([tid]).get(tid)
    if bars is None or not 1 <= first <= last <= len(bars):
        raise ValueError(f"track {tid} has bars 1-{0 if bars is None else len(bars)}, not {first}-{last}")
    return _describe(_section(bars, _beats(t), first - 1, last - 1, t["duration"], t["chunk_seconds"], tid,
                              None), t)
