"""breakdig command line."""

import argparse
import errno
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from . import __version__, audio
from .db import Index, default_home
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


def cmd_index(args):
    _need_ffmpeg()
    from tqdm import tqdm

    from . import scan
    from .indexer import Abort, Indexer

    missing = [p for p in args.paths if not Path(p).exists()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")
    # The index's own scratch files and preview clips are WAVs too.
    files = scan.walk(args.paths, args.exclude, skip=[Index.home_for(args.home)])
    if not files:
        print(f"No audio files found (looked for {', '.join(sorted(e[1:] for e in scan.AUDIO_EXTS))}).")
        return
    import torch
    if not torch.cuda.is_available():
        print("warning: CUDA is not available, so separation runs on the CPU and is 10-30x slower.\n"
              "  For an NVIDIA GPU install the CUDA build of torch (cu126 for drivers older than 580):\n"
              "  pip install --force-reinstall --no-deps torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu130", file=sys.stderr)
    counts, times = {}, []
    with Index(args.home) as index:
        logfile = index.home / "index.log"
        handler = logging.FileHandler(logfile, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            with Indexer(index, model=args.model, shifts=args.shifts) as indexer:
                print(f"Scanning {len(files)} files...")
                todo, known = indexer.pending(tqdm(files, unit="file", dynamic_ncols=True, leave=False),
                                              retry_failed=args.retry_failed)
                print(f"{known} already indexed, {len(todo)} to do. Index: {index.home}")
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
        print(f"Done in {elapsed:.0f}s: {', '.join(parts) or 'nothing new'}.")
        if times:
            print(f"Separated tracks took {sum(times) / len(times):.1f}s each on average.")
        if any(s not in ("ok", "updated", "skipped", "duplicate") for s in counts):
            print(f"Details: breakdig stats --failures, or {logfile}")


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
