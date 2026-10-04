"""breakdig command line."""

import argparse
import errno
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from . import __version__, audio
from .db import STEMS, Index, default_home
from .query import SILENT_DB, SORTS, find, get_section, keep_from, parse_bpm, pattern_from, search_tracks

log = logging.getLogger("breakdig")


def _need_ffmpeg():
    if not audio.have_ffmpeg():
        sys.exit(audio.FFMPEG_HINT)


def _open(args) -> Index:
    """The index, or exit if --home names a folder that has none, so a typo does not look empty."""
    if args.home is not None and not (Index.home_for(args.home) / "index.sqlite").exists():
        sys.exit(f"breakdig: there is no index at {Index.home_for(args.home)}. Check --home.")
    return Index(args.home)


def _none_found(args, index) -> str:
    st = index.stats()
    if not st["ok"]:
        failed = st["by_status"].get("failed", 0)
        return (f"The index at {index.home} is empty. " + (
            f"{failed} file(s) failed; see breakdig stats --failures, then run breakdig index "
            "--retry-failed <folder>." if failed else "Run: breakdig index <folder>"))
    if getattr(args, "search", None) and not search_tracks(index.ok_tracks(), args.search):
        return f"No indexed track matches --search {args.search!r}."
    return "No matching sections. Try --min-bars 1 or --silence-db 25."


def _sections(args, index):
    if getattr(args, "pick", None):
        return [get_section(index, p) for p in args.pick]
    return find(index, pattern_from(args.only, args.no), min_bars=args.min_bars, bpm=parse_bpm(args.bpm),
                text=args.search, skip_seams=args.skip_seams, sort=args.sort, limit=args.limit,
                silent_db=args.silence_db, cleaner_than=args.cleaner_than)


def _clock(t: float) -> str:
    m, s = divmod(round(t, 3), 60)
    return f"{int(m)}:{s:06.3f}"


def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "~"


def print_table(sections):
    head = f"{'ID':<11} {'ARTIST - TITLE':<44} {'BARS':>4} {'START':>9} {'END':>9} {'BPM':>6} {'CLEAN':>6}"
    print(head)
    print("-" * len(head))
    for s in sections:
        bpm = f"{s.bpm:.1f}" if s.bpm else "?"
        clean = "-" if s.clean is None else f"{s.clean:.1f}"
        flag = "  seam" if s.seam else ""
        print(f"{s.id:<11} {_fit(f'{s.artist} - {s.title}', 44):<44} {s.bars:>4} {_clock(s.start):>9} "
              f"{_clock(s.end):>9} {bpm:>6} {clean:>6}{flag}")


def _index_files(index, files, model=None, shifts=1, retry_failed=False, say=print) -> dict[str, int]:
    """Index whatever in files is not in index yet, with progress output; resumable.

    The shared core of `breakdig index` and `breakdig drill`. `say` prints the
    progress lines, so a `--json` run can keep its stdout parseable."""
    from tqdm import tqdm

    from . import scan
    from .indexer import Abort, Indexer

    import torch
    if not torch.cuda.is_available():
        print("warning: CUDA is not available, so separation runs on the CPU and is 10-30x slower.\n"
              "  For an NVIDIA GPU install the CUDA build of torch (cu126 for drivers older than 580):\n"
              "  pip install --force-reinstall --no-deps torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu130", file=sys.stderr)
    counts, times = {}, []
    logfile = index.home / "index.log"
    handler = logging.FileHandler(logfile, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        with Indexer(index, model=model, shifts=shifts) as indexer:
            say(f"Scanning {len(files)} files...")
            todo, known = indexer.pending(tqdm(files, unit="file", dynamic_ncols=True, leave=False),
                                          retry_failed=retry_failed)
            say(f"{known} already indexed, {len(todo)} to do. Index: {index.home}")
            if todo:
                indexer.load_models()
            t_start = time.perf_counter()
            with tqdm(todo, unit="track", dynamic_ncols=True, disable=not todo) as bar:
                for f in bar:
                    bar.set_postfix_str(_fit(Path(f.path).name, 40))
                    t0 = time.perf_counter()
                    status = indexer.index_file(f)
                    counts[status] = counts.get(status, 0) + 1
                    if status == "ok":
                        times.append(time.perf_counter() - t0)
                        log.info("ok %s (%.1fs)", f.path, times[-1])
                    elif status == "updated":
                        log.info("updated %s (new tags or bytes, same audio)", f.path)
                    else:
                        _report_problem(bar, index, f, status)
    except (KeyboardInterrupt, Abort, sqlite3.Error) as e:
        done = sum(counts.values())
        note = f" {done} tracks were committed; run the same command again to carry on." if done else ""
        if isinstance(e, KeyboardInterrupt):
            print(f"\nStopped.{note}")
            sys.exit(130)
        sys.exit(f"breakdig: {str(e).rstrip('.')}.{note}")
    elapsed = time.perf_counter() - t_start
    parts = [f"{n} {s}" for s, n in sorted(counts.items())]
    say(f"Done in {elapsed:.0f}s: {', '.join(parts) or 'nothing new'}.")
    if times:
        say(f"Separated tracks took {sum(times) / len(times):.1f}s each on average.")
    if any(s not in ("ok", "updated", "skipped", "duplicate") for s in counts):
        say(f"Details: breakdig stats --failures, or {logfile}")
    return counts


def cmd_index(args):
    _need_ffmpeg()
    from . import scan

    missing = [p for p in args.paths if not Path(p).exists()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")
    # The index's own scratch files and preview clips are WAVs too.
    files = scan.walk(args.paths, args.exclude, skip=[Index.home_for(args.home)])
    if not files:
        print(f"No audio files found (looked for {', '.join(sorted(e[1:] for e in scan.AUDIO_EXTS))}).")
        return
    with Index(args.home) as index:
        _index_files(index, files, args.model, args.shifts, args.retry_failed)


def _report_problem(bar, index, f, status):
    error = (index.by_key(f.key)["error"] if status != "skipped"
             else "moved or changed since the scan; the next run picks it up")
    bar.write(f"{status}: {f.path}\n  {error}")
    log.warning("%s %s: %s", status, f.path, error)


def cmd_find(args):
    with _open(args) as index:
        sections = _sections(args, index)
        if args.json:
            print(json.dumps([s.to_dict() for s in sections], indent=2))
            return
        if not sections:
            print(_none_found(args, index))
            return
        print_table(sections)
        print(f"\n{len(sections)} section(s). CLEAN is the loudest unwanted stem minus the target, "
              "in dB; lower is cleaner.")


def cmd_export(args):
    _need_ffmpeg()
    from .export import export

    keep = keep_from(args.keep)
    with _open(args) as index:
        sections = _sections(args, index)
        if not sections:
            print(f"Nothing to export. {_none_found(args, index)}")
            return
    results = export(sections, args.out, sp404=args.sp404, keep=keep)
    out, wrote = Path(args.out).resolve(), 0
    for r in results:
        if r.path:
            wrote += 1
            print(f"wrote   {r.path}")
        elif r.skipped:
            print(f"skipped {r.section.id} (already in {out})")
        else:
            print(f"failed  {r.section.id}: {r.error}")
    print(f"\n{wrote} file(s) in {out}")
    if any(r.error for r in results):
        sys.exit(1)


PLAYLIST_EXTS = {".m3u", ".m3u8", ".txt"}


def _playlist(path: Path) -> list[str]:
    """One path per line; # lines are comments (including Extended M3U's #EXTINF);
    relative entries sit next to the list."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line.strip('"'))
        entries.append(str(p if p.is_absolute() else path.parent / p))
    return entries


def _drill_inputs(inputs: list[str], skip: list[str]) -> list[str]:
    """Songs in input order: folders walked, playlists read line by line, files as they are.
    Folders in skip are left alone, so a --out folder inside the library is not indexed and
    drilled on the next run."""
    from . import scan

    files = []
    for item in inputs:
        p = Path(item)
        if not p.exists():
            sys.exit(f"not found: {item}")
        if p.is_dir():
            files.extend(scan.walk([item], skip=skip))
        elif p.suffix.lower() in PLAYLIST_EXTS:
            files.extend(_playlist(p))
        else:
            files.append(scan.on_disk(item))
    return files


def _drop_to_keep(text: str) -> tuple[frozenset, str]:
    """The stems --drop leaves playing and a short label for them, or exit if that would
    silence the drill."""
    from .query import parse_stems

    drop = parse_stems(text)
    unknown = sorted(drop - set(STEMS))
    if unknown:
        sys.exit(f"breakdig: --drop takes some of {', '.join(STEMS)}, got {', '.join(unknown)}")
    keep = frozenset(s for s in STEMS if s not in drop)
    if not keep:
        sys.exit("breakdig: --drop would leave no stems playing; use --keep to pick what does")
    return keep, "no " + "+".join(sorted(drop, key=STEMS.index))


def _drill_row(index, f: str):
    """The indexed track behind a setlist file, following duplicates to their original,
    as (row, reason). reason is set when the file cannot be drilled."""
    from . import scan

    row = index.by_path(f)
    if row is None:
        try:
            row = index.by_key(scan.content_key(f).key)
        except OSError:
            row = None
    note = None
    if row is None:
        note = "was not indexed; see breakdig stats --failures"
        return row, note
    if row["status"] == "duplicate":
        row = index.track(row["dup_of"])
        if row is None:
            return None, "is a copy of a track that is no longer indexed"
    if row["status"] == "failed":
        note = (f"failed to index: {row['error']}; run breakdig index --retry-failed "
                "on its folder to try again")
    elif row["status"] == "skipped":
        note = "moved or changed during indexing; run breakdig index on it again"
    elif row["status"] == "no_grid":
        note = None  # a drill without --bars is still possible; the caller decides
    return row, note


def _rung_cues(rungs, passes: int, rate: int) -> list[tuple[int, str]]:
    """Cue points for a WAV drill: the count-in of each rung, and every repeat after it."""
    cues = []
    for r in rungs:
        for rep in range(passes):
            t = r.offset if rep == 0 else r.offset + r.count_in_seconds \
                + rep * (r.pass_seconds + r.gap_seconds)
            label = f"{r.percent}%" if rep == 0 else f"{r.percent}% x{rep + 1}"
            cues.append((round(t * rate), label))
    return cues


def cmd_drill(args):
    _need_ffmpeg()
    import csv

    import numpy as np

    from . import drill, scan
    from .drill import (SWEET_SPOT_FLOOR, drill_name, free_path, mine, open_sink, origin_text,
                        plan, render, separator_beep, total_seconds, wav_subtype)
    from .export import load_separator
    from .isolate import isolate, stems_name
    from .query import get_section

    if args.keep and args.drop:
        sys.exit("breakdig: use --keep or --drop, not both")
    keep = keep_from(args.keep)
    stems_label = ""
    if args.drop:
        keep, stems_label = _drop_to_keep(args.drop)
    elif keep:
        stems_label = stems_name(keep)

    ladder = args.ladder
    out = Path(args.out).resolve()
    if out.exists() and not out.is_dir():
        sys.exit(f"breakdig: {out} is a file, not a folder")
    files = _drill_inputs(args.inputs, skip=[str(out)])
    if not files:
        sys.exit("breakdig: no audio files found. Give folders, audio files, or .m3u/.txt lists "
                 "with one path per line.")
    if min(ladder.percents) < SWEET_SPOT_FLOOR:
        print(f"warning: {min(ladder.percents)}% is outside the 0.75x-1.5x stretch sweet spot; "
              "expect artifacts on the slowest rungs", file=sys.stderr)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"breakdig: cannot create {out}: {e.strerror}")

    rendered, rows, manifest = 0, [], []
    song_cues: list[tuple[int, str]] = []  # where each song starts inside a --combined WAV
    beep_seconds = drill.SEPARATOR_BEEP[1] * 2 + drill.SEPARATOR_BEEP[2]
    # In --json mode the indexer's progress lines move to stderr, so stdout stays parseable.
    say = print if not args.json else (lambda *a, **k: print(*a, file=sys.stderr, **k))
    with Index(args.home) as index:
        _index_files(index, files, say=say)
        separator = None
        combined = None
        cursor = 0.0  # where the next song starts inside a --combined file
        for f in files:
            if not os.path.exists(f):
                print(f"skipped {f}: in the setlist but not on disk", file=sys.stderr)
                continue
            row, note = _drill_row(index, f)
            if note is None and args.bars and row["status"] == "no_grid":
                note = "has no beat grid (fewer than 8 downbeats), so --bars does not apply"
            if note:
                print(f"skipped {f}: {note}", file=sys.stderr)
                continue

            start, end, bars_label = 0.0, row["duration"] or 0.0, ""
            if args.bars:
                try:
                    section = get_section(index, f"{row['id']}:{args.bars[0]}-{args.bars[1]}")
                except ValueError as e:
                    print(f"skipped {f}: {e}", file=sys.stderr)
                    continue
                start, end, bars_label = section.start, section.end, f"{args.bars[0]}-{args.bars[1]}"
            artist, title, album = row["artist"], row["title"], row["album"] or ""
            bpm = row["bpm"] or drill.DEFAULT_BPM
            try:
                probe = audio.probe(f)
                mix = audio.decode(f, start=start or None, end=end if args.bars else None, info=probe)
            except audio.DecodeError as e:
                print(f"failed  {f}: {e}", file=sys.stderr)
                continue
            rate = probe.sample_rate
            try:
                if mix.shape[1] > 2:
                    mix = np.ascontiguousarray(mix[:, :2])
                if keep:
                    if separator is None:
                        separator = load_separator()
                    mix = isolate(mix, rate, keep, separator)
                if args.combined:
                    if mix.shape[1] == 1:  # one width for the whole file
                        mix = np.repeat(mix, 2, axis=1)
                    peak = float(np.abs(mix).max(initial=0.0))
                    if peak > 1.0:
                        mix = mix / peak  # int containers have no headroom; turn down, don't clip
                rungs = plan(end - start, bpm, ladder, args.passes, args.count_in, args.gap_bars)
            except (audio.DecodeError, OSError, RuntimeError, ValueError) as e:
                hint = " Close other programs using the GPU, then run the same command again." \
                    if "out of memory" in str(e).lower() else ""
                print(f"failed  {f}: {e}{hint}", file=sys.stderr)
                continue
            seconds = total_seconds(rungs, args.passes)
            if not args.json:
                print(f"\n{artist} - {title}" + (f" (bars {bars_label})" if bars_label else "")
                      + f"  {bpm:.1f} BPM")
                for r in rungs:
                    m, s = divmod(round(r.pass_seconds), 60)
                    print(f"  {r.percent:>3}%  {bpm * r.time_factor:6.1f} BPM  {int(m)}:{s:02d}")

            out_rate = 44100 if (args.format == "mp3" or args.combined) else rate
            peak = float(np.abs(mix).max(initial=0.0))
            if args.format == "mp3" and peak > 1.0:
                mix = mix / peak  # lossy masters decode past full scale; turn down, don't clip
            subtype = wav_subtype(probe.bits, peak)
            # The tag stores the stems in the machine form (bass+drums+other), which is what
            # a rerun with the opposite --keep/--drop spelling must still recognise as its own.
            tag_stems = stems_name(keep)
            comment = origin_text(f, ladder.label, args.passes, args.count_in, args.gap_bars,
                                  bars_label, tag_stems, row["key"] or "")
            drill_title = f"{title} - drill {ladder.label}" + (f" ({stems_label})" if stems_label else "")
            try:
                if args.combined:
                    if combined is None:
                        name = f"drill {ladder.label}" + (f" - {stems_label}" if stems_label else "") \
                            + f".{args.format}"
                        dest = out / name
                        combined = open_sink(args.format, dest, 2, out_rate, subtype)
                    else:
                        combined.write(separator_beep(out_rate, 2))
                        cursor += beep_seconds
                    song_offset = cursor
                    if args.format == "wav":
                        song_cues.append((round(song_offset * out_rate), title[:80]))
                    for block in render(mix, rate, out_rate, rungs, args.passes, args.count_in):
                        combined.write(block)
                    cursor += seconds
                else:
                    ident = mine(f, row["key"] or "", ladder.label, args.passes, args.count_in,
                                 args.gap_bars, bars_label, tag_stems)
                    dest = free_path(out, drill_name(artist, title, ladder.label, args.format), ident)
                    song_offset = 0.0
                    sink = open_sink(args.format, dest, mix.shape[1], out_rate, subtype)
                    # A WAV drill carries a cue per rung's count-in (and per repeat), so an
                    # editor can jump straight to any speed.
                    cues = _rung_cues(rungs, args.passes, out_rate) if args.format == "wav" else None
                    try:
                        for block in render(mix, rate, out_rate, rungs, args.passes, args.count_in):
                            sink.write(block)
                        sink.finish(artist, drill_title, album, seconds, comment, cues)
                    except BaseException:
                        sink.abort()
                        raise
                    rendered += 1
                    if not args.json:
                        print(f"wrote   {dest}")
            except (audio.DecodeError, OSError, RuntimeError) as e:
                if combined is not None:
                    combined.abort()
                    sys.exit(f"breakdig: the combined drill failed at {f}: {e}")
                hint = " Close other programs using the GPU, then run the same command again." \
                    if "out of memory" in str(e).lower() else ""
                print(f"failed  {f}: {e}{hint}", file=sys.stderr)
                continue
            rows.append({"file": dest.name, "artist": artist, "title": title,
                         "rungs": " ".join(str(p) for p in ladder.percents),
                         "seconds": f"{seconds:.1f}", "offset": f"{song_offset:.1f}", "source": f})
            manifest.append({"file": dest.name, "artist": artist, "title": title, "album": album,
                             "source": f, "bpm": round(bpm, 1), "bars": bars_label or None,
                             "ladder": ladder.label, "stems": stems_label or None,
                             "seconds": round(seconds, 3), "offset": round(song_offset, 3),
                             "path": str(dest),
                             "rungs": [{"percent": r.percent, "bpm": round(bpm * r.time_factor, 1),
                                        "seconds": round(r.pass_seconds, 3),
                                        "offset": round(r.offset + song_offset, 3)} for r in rungs]})
        if combined is not None:
            dest = combined.dest
            combined.finish("breakdig", f"drill {ladder.label}"
                            + (f" - {stems_label}" if stems_label else ""), "", cursor,
                            origin_text(f"combined: {len(rows)} song(s)", ladder.label, args.passes,
                                        args.count_in, args.gap_bars, "", tag_stems, ""),
                            song_cues if args.format == "wav" else None)
            rendered += 1
            if not args.json:
                print(f"wrote   {dest}")
    if rows:
        with open(out / "drills.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "artist", "title", "rungs", "seconds",
                                               "offset", "source"])
            w.writeheader()
            w.writerows(rows)
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"\n{rendered} file(s) in {out}")
    if not rendered:
        sys.exit(1)


def cmd_stats(args):
    with _open(args) as index:
        st = index.stats()
        print(f"Index:      {index.home}")
        print(f"Tracks:     {st['ok']} indexed, {st['audio_seconds'] / 3600:.1f} h of audio, {st['bars']} bars")
        for status, n in sorted(st["by_status"].items()):
            print(f"  {status:<10} {n}")
        if st["ok"] and st["index_seconds"]:
            print(f"Speed:      {st['index_seconds'] / st['ok']:.1f}s per track, "
                  f"{st['audio_seconds'] / st['index_seconds']:.1f}x realtime")
        if args.failures:
            rows = index.failures()
            print("\nNot indexed:" if rows else "\nNo failures.")
            for r in rows:
                print(f"  [{r['status']}] {r['path']}\n      {r['error']}")


def cmd_ui(args):
    _need_ffmpeg()
    import socket
    import threading
    import webbrowser

    import uvicorn

    from .web.app import LOCAL_HOSTS, create_app

    _open(args).close()
    # A browser cannot open the any-address itself, only this machine through it.
    shown = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(args.host, args.host)
    url = f"http://{f'[{shown}]' if ':' in shown else shown}:{args.port}"
    app = create_app(args.home, host=args.host)
    # Checked before the browser opens, or it would open on whatever already holds the port.
    try:
        with socket.create_server((args.host, args.port),
                                  family=socket.AF_INET6 if ":" in args.host else socket.AF_INET):
            pass
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            sys.exit(f"breakdig: port {args.port} is in use. Is breakdig ui already running? "
                     "If not, pick another with --port.")
        sys.exit(f"breakdig: cannot listen on {args.host} port {args.port}: {e.strerror or e}")
    if args.host not in LOCAL_HOSTS:
        print(f"warning: listening on {args.host}, so anyone who can reach this machine can browse "
              "your index and write exports to any folder.", file=sys.stderr)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"breakdig UI on {url}  (Ctrl+C to stop)")
    # The default proactor loop on Windows prints a traceback each time the browser drops a
    # preview request, which it does on every play.
    loop = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", loop=loop)


def _positive(text: str, least: int = 1) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if n < least:
        raise argparse.ArgumentTypeError(f"must be {least} or more, got {n}")
    return n


def _port(text: str) -> int:
    n = _positive(text)
    if n > 65535:
        raise argparse.ArgumentTypeError(f"ports go up to 65535, got {n}")
    return n


def _path(text: str) -> str:
    # cmd.exe reads the \" in "D:\My Music\" as a literal quote, and then the rest of the line
    # can run on into the same argument.
    if '"' in text.rstrip('"'):
        raise argparse.ArgumentTypeError(
            f"{text!r} has a quote in it. Leave out the backslash before a closing quote, "
            r'as in "D:\My Music"')
    return text.rstrip('"')


def _ladder(text: str):
    from .drill import parse_ladder

    try:
        return parse_ladder(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _bars(text: str) -> tuple[int, int]:
    try:
        a, b = (int(x) for x in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--bars takes A-B like 9-16, got {text!r}") from None
    if a < 1 or b < a:
        raise argparse.ArgumentTypeError(f"--bars takes A-B with 1 <= A <= B, got {text!r}")
    return a, b


def _filters(p):
    g = p.add_argument_group("what to look for")
    g.add_argument("--only", metavar="STEMS",
                   help="stems that play alone, e.g. drums, vocals, bass,drums (default: drums)")
    g.add_argument("--no", metavar="STEMS", help="stems that must be silent, e.g. drums")
    g.add_argument("--min-bars", type=_positive, default=2, help="shortest section in bars (default 2)")
    g.add_argument("--silence-db", type=float, default=SILENT_DB, metavar="DB",
                   help=f"how far below the rest a stem must sit to count as silent (default {SILENT_DB:g}); "
                        "20-25 finds more at the cost of some bleed")
    g.add_argument("--cleaner-than", type=float, metavar="DB",
                   help="only sections whose CLEAN is at or below this, e.g. -25")
    g.add_argument("--bpm", help="tempo or range, e.g. 90 or 85-100")
    g.add_argument("--search", help="only tracks with every word of this in the artist, title, album or path")
    g.add_argument("--skip-seams", action="store_true",
                   help="drop sections that cross a chunk seam in a long file")
    g.add_argument("--sort", choices=sorted(SORTS), default="clean", help="default: clean")
    g.add_argument("--limit", type=_positive, help="at most this many sections")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="breakdig",
        description="Find bar-exact drum breaks, acapellas and other sections in your own music.")
    p.add_argument("--version", action="version", version=f"breakdig {__version__}")
    home_help = f"index folder (default: $BREAKDIG_HOME or {default_home()})"
    p.add_argument("--home", type=_path, default=None, help=home_help)
    # Also accepted after the command. SUPPRESS keeps the subcommand from resetting a --home given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--home", type=_path, default=argparse.SUPPRESS, help=home_help)
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("index", parents=[common], help="separate and profile audio files (resumable)")
    s.add_argument("paths", nargs="+", type=_path, help="folders or files")
    s.add_argument("--model", help="audio-separator model file (default htdemucs_ft.yaml)")
    s.add_argument("--shifts", type=lambda t: _positive(t, 0), default=1,
                   help="Demucs random shifts; 2 is slightly cleaner and twice as slow (default 1)")
    s.add_argument("--retry-failed", action="store_true", help="try previously failed files again")
    s.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help='skip files and folders whose name matches, e.g. "*.removed.*" (repeatable)')
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("find", parents=[common], help="list matching sections")
    _filters(s)
    s.add_argument("--pick", nargs="+", metavar="ID", help="show these section ids instead of searching")
    s.add_argument("--json", action="store_true", help="print JSON instead of a table")
    s.set_defaults(func=cmd_find)

    s = sub.add_parser("export", parents=[common], help="cut matching sections from the original files")
    _filters(s)
    s.add_argument("--pick", nargs="+", metavar="ID", help="export these section ids from find, e.g. 12:5-8")
    s.add_argument("--out", required=True, type=_path, help="output folder")
    s.add_argument("--sp404", nargs="?", const="mk2", choices=["mk2", "sx"], metavar="MODEL",
                   help="16-bit, BRK_0001.WAV names and an index.csv; 48 kHz for the MKII, "
                        "or --sp404 sx for 44.1 kHz on the SX, A and original 404")
    s.add_argument("--keep", metavar="STEMS",
                   help="only these stems, separated again from each section, e.g. drums or drums,bass; "
                        "seconds per section on a GPU, far longer on CPU")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("stats", parents=[common], help="what is in the index")
    s.add_argument("--failures", action="store_true", help="list files that were not indexed and why")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("drill", parents=[common],
                       help="render a tempo-ladder practice drill per song: a click count-in, "
                            "each speed, a gap, climbing to full speed")
    s.add_argument("inputs", nargs="+", type=_path, metavar="INPUT",
                   help="folders, audio files, or .m3u/.txt lists with one path per line "
                        "(# lines are comments)")
    s.add_argument("--out", required=True, type=_path, help="output folder")
    s.add_argument("--ladder", type=_ladder, default="70-100:5", metavar="START-END:STEP",
                   help="speeds in percent, 50 <= start < end <= 100, step 1-20 "
                        "(default 70-100:5)")
    s.add_argument("--passes", type=_positive, default=1,
                   help="times to repeat each rung (default 1)")
    s.add_argument("--keep", metavar="STEMS",
                   help="only these stems, separated again per song, e.g. drums,bass; needs a "
                        "GPU and about a separation per song")
    s.add_argument("--drop", metavar="STEMS", help="the mix without these stems, e.g. vocals")
    s.add_argument("--count-in", type=_positive, default=4,
                   help="clicks before each rung (default 4)")
    s.add_argument("--gap-bars", type=_positive, default=1,
                   help="bars of silence after each rung (default 1)")
    s.add_argument("--bars", type=_bars, metavar="A-B",
                   help="drill only bars A to B from the beat grid, e.g. 9-16")
    s.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                   help="mp3 at 192 kbps CBR (resampled to 44.1 kHz), or wav at the source "
                        "rate (default mp3)")
    s.add_argument("--combined", action="store_true",
                   help="one file for the whole setlist, songs separated by two beeps")
    s.add_argument("--json", action="store_true", help="print a JSON manifest instead of the table")
    s.set_defaults(func=cmd_drill)

    s = sub.add_parser("ui", parents=[common], help="open the web UI")
    s.add_argument("--host", default="127.0.0.1",
                   help="address to listen on; anything but this machine lets others on the network in")
    s.add_argument("--port", type=_port, default=8765, help="default 8765")
    s.add_argument("--no-browser", action="store_true", help="start the server without opening a browser tab")
    s.set_defaults(func=cmd_ui)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return
    try:
        args.func(args)
    except (ValueError, OSError, sqlite3.Error) as e:
        sys.exit(f"breakdig: {e}")


if __name__ == "__main__":
    main()
