# breakdig progress

State as of 2026-09-29: v0.1.0 is released. The repo is public at
https://github.com/Booyaka101/breakdig, the GitHub release has the wheel, sdist and Windows
zip, and the same wheel and sdist are on PyPI. The r/SP404 post has not gone up.
On 2026-09-30 a fourth review pass (UI and core) was fixed on the `review-fixes` branch;
see the Unreleased section of CHANGELOG.md.

## What is verified (on this PC: Windows 11, RTX 4090, driver 610.88, Python 3.11)

- Test suite: `BREAKDIG_MODELS=.scratch/models .venv/Scripts/python -m pytest -q -rs`
  gives 186 passed in 112 s, no skips, the `gpu` tests running real Demucs and beat_this on
  the 4090. `.github/workflows/ci.yml` runs the rest on Windows and Linux with CPU torch;
  the same steps in a clean CPU-only venv gave 170 passed and the 4 gpu tests skipped.
  The first GitHub run failed four tests that only held on this PC (Windows path
  separators, a case-sensitive `*.wav` glob on Linux, and float WAVs whose PEAK chunk
  records the second they were written). With those fixed, CI and the Release workflow
  are green on both runners at v0.1.0.
- A full review pass (each finding reproduced before it was fixed) led to:
  - Export: SP-404 reruns key sections on source and exact span, so a longer section from
    the same first bar is no longer skipped; numbering continues past deleted files; an
    index.csv saved by Excel (ANSI or with extra columns) is read and its columns kept;
    loud MP3s that decode past full scale are written as float instead of clipped, and
    SP-404 clips are turned down to fit.
  - Index: the key now hashes the start, middle and end of the file, since WAV edits with
    the same first MiB collided. An unchanged file (same size and mtime) is not read on a
    rerun. A retagged file keeps its analysis ("updated"), an overwritten one replaces its
    old entry, ID3 tags in AIFF and WAV are read, a file that changes mid-run is left for
    next time, and a GPU out-of-memory error stops the run instead of failing every file.
  - Web: requests from other sites are refused (Sec-Fetch-Site and Origin checks), reveal
    passes the path as its own argument, concurrent previews of one clip no longer race,
    and on Windows the UI runs on a selector event loop. The default proactor loop logged 12
    ConnectionResetError tracebacks over 10 previews; the selector loop logs none.
  - UI: ticked rows survive a sort or new search, the filters are remembered, and times
    such as 1:59.9996 no longer show as 1:60.000.
  - New: `find --cleaner-than DB`, also in the UI.
  - install.bat: pauses only when double-clicked, picks cu126 for drivers older than 580,
    and skips the torch swap when CUDA torch is already there.
- A second review pass, again reproducing each finding first:
  - Query: a stem that is silent all through a track (an instrumental's vocals) set its
    own floor, so it never counted as silent; anything under -90 dBFS now does. `--search`
    matches accented names whichever Unicode form they were typed in. Sort by artist is
    stable for tracks with no tags.
  - Export: the comment tag records the content key, so a re-export after moving the
    library finds its earlier files instead of writing "(2)" copies, and SP-404 reruns
    read what is done from the BRK files themselves. An index.csv saved by Excel with
    semicolons is read and kept that way. An index.csv open in Excel stops the export
    before any clip is written, with a message saying so. A failed write leaves no
    `.part` file, the same section picked twice is written once, and an output path that
    is a file is refused.
  - Web: a stale pick fails on its own instead of failing the whole export; the output
    folder may be quoted or start with `~`, and a relative one is refused with an example.
    The page cannot be framed and is not cached. `::1` works. `breakdig ui` on a port in
    use says so instead of opening the browser on whatever holds it.
  - UI, driven in Chrome with Playwright (`.scratch/r2/ui_verify.py`, all 14 checks pass):
    a stale saved filter no longer blanks the page, a slow earlier search cannot
    overwrite a later one, the playing row survives a sort, Space ticks a focused
    checkbox, Export stays disabled while it runs, an export unticks what it handled, a
    new "clear" button unticks rows the current search hides, and a clip that cannot play
    shows the server's reason.
  - CLI: `--home` works after the subcommand, and the empty-index message names the index.
  - install.bat: from PowerShell it paused, since `%cmdcmdline%` names the script there
    too. It now pauses only on Explorer's `cmd /c ""path" "` form; checked both ways, a
    ShellExecute launch (as a double click does) waits and a PowerShell call does not.
  - CI ran `python -m pytest`, which puts the checkout on sys.path and so tested the
    source tree rather than the installed package. It now runs `pytest`.
  - Index, each case scripted in `.scratch/review2-pipeline/` and run before and after:
    two files that swap names take their rows along instead of trading analyses; a file
    overwritten with a copy of another no longer leaves two rows for one sound; a retag
    plus rename is "updated" without separating again; audio trimmed or shifted in place
    gets fresh bars instead of keeping the old grid; files whose sampled bytes collide are
    told apart by a full hash; an unchanged file whose mtime was touched is hashed once;
    a path typed in another case is not hashed again; the index home inside a scanned
    folder is skipped. A float file peaking at +12 dB was clipped going into Demucs; each
    file now gets its own input gain to keep 6 dB of headroom. A GPU out-of-memory error
    in beat tracking stops the run, as it already did in separation. A folder of 400
    one-shots took 614 s, mostly comparing each against every profile; files too short
    to compare now skip that, and it takes 130 s. Duplicates no longer show in
    `stats --failures`, and Ctrl+C during the scan exits cleanly.
  - Re-indexing the 14 CC tracks with the new gain and comparing against the old index:
    158 of 166 sections (every preset at 30, 25 and 20 dB) are identical, with clean
    scores within 0.01 dB. The other 8 are one track, where beat_this dropped a downbeat
    it had placed half a bar early, so later bars are numbered one lower. Per-bar stem
    levels agree within 0.43 dB.
  - New: a progress bar while scanning, and .opus, .aac and .wma files. Opus and WMA copies
    of a CC track were recognised as the same audio as its MP3; a raw .aac copy was not
    (see Known gaps).
- A third review pass, four reviewers (audio, regressions, first-run journey, UI). Every fix
  has a test that was run against a copy of the code from before the pass
  (`.scratch/r3-base`) and fails there:
  - Silence rule: a stem also counted as silent when it sat below its own 95th percentile,
    which let a "no drums" section through with the drums only 9.2 dB under "other". That
    test is gone. On home-real (51 tracks), "no drums" at 30 dB went from 87 sections to 83
    and the worst clean score from -9.2 to -30.1; on the 14 CC tracks from 10 to 9, -16.9 to
    -30.9. "drums only" kept all 40 sections on home-real. The README's examples are
    unchanged.
  - Export edges snap to the sample in the 2 ms window where the loudest channel is
    quietest. With the old zero crossing of the channel average, 20% of edges were still
    above 0.1 on some channel; now 1% are.
  - The bar after the last downbeat was dropped. 7 of the 14 CC tracks and 16 of the 51
    home-real tracks have a bar or more of sound after it. It now counts when the audio
    runs a full typical bar past the downbeat.
  - A model file cut short by an interrupted download failed every track for good. The
    run now stops, deletes the partial file and says to run again. Both models load before
    the first file, and Ctrl+C during that download exits cleanly.
  - Python 3.12: audio-separator imports audioread, which newer librosa no longer pulls in.
    It is now a direct dependency; the rebuilt wheel installed it into the reviewer's 3.12
    venv and the imports work.
  - Index: a profile another program holds open no longer stops a rekey or forget, and
    orphaned profiles are swept on the next run. A second index run on the same home stops
    with a message (a lock file). Originals on an unplugged drive keep their rows instead of
    being replaced by their duplicates. `--retry-failed` after a copy of the same audio was
    indexed and deleted no longer hits a SQLite IntegrityError. A duplicate is checked again
    when its original is overwritten. A section id can no longer point at different audio
    after a re-index. On a subst drive the index home is skipped and a path typed in
    another case is not hashed again. Byte-identical copies were fully hashed on every
    rerun; the rerun on the reviewer's folder went from 6 full hashes to 0.
  - CLI: `find`, `export`, `stats` and `ui` with a `--home` that has no index say so
    instead of reporting an empty index; an empty index with failures says how to retry
    them; a search that matches no track says that rather than suggesting looser filters;
    `--search` matches every word across artist, title, album and path; a path ending in
    `\"` (as cmd passes `"D:\My Music\"`) works, and one with a quote in the middle is
    refused with the fix; `--port` is checked; "two" for a number gets a plain message; an
    export folder on a missing drive gives "cannot create" instead of a WinError.
  - install.bat stops with "unzip the whole folder first" when the wheel is not next to
    it, and its last line no longer breaks on a folder with `&` in its name (both run for
    real in scratch folders).
  - UI (`.scratch/r3/ui_verify3.py` in Playwright, every check passes): previews loop through
    Web Audio. The `<audio loop>` element left 3.6 to 11.8 ms of silence at each loop point on
    14:165-166; recorded the same way, the new player leaves none over three loops. The
    dock no longer covers the last rows when it wraps. Space on a focused button presses
    that button instead of toggling playback. Sort headers work from the keyboard and carry
    aria-sort; row checkboxes, play buttons, select-all and the folder field have labels.
    Hovering no longer hides the playing row. A search that matches no track, and an empty
    index with failures, get their own messages. A slow clip no longer holds up previews
    of other clips.
- A fourth review pass, four reviewers (audio, data integrity, UI, maintainability). The
  integrity reviewer also fuzzed index runs over 15 seeds, about 660 steps and 170 injected
  crashes. Every fix has a test that fails against the code from before the pass
  (`.scratch/r4-base`):
  - The added last bar was decided by file length, so a track that faded out or stopped
    early still got a bar of near silence. It now needs the mix to sound through 90% of
    that bar. On the 14 CC tracks the extra bar went from 11 tracks to 6; the 5 dropped
    tails were 16 to 60% sound.
  - Index: renaming a file only in case (Song.mp3 to song.mp3) now keeps its row. A full
    disk stops the run with a message instead of failing every file. A SQLite error
    mid-run, or a damaged index file, gives a message saying how many tracks were kept
    rather than a traceback. A crash during a rekey can no longer leave a row with no
    profile. Scratch folders from any killed run are swept, not only the last.
  - SP-404: clips written by an export that was killed before it saved index.csv are
    listed again on the next export, rebuilt from their tags.
  - UI (`.scratch/r4/ui_verify4.py` in Playwright, all 11 checks pass): decoded previews
    are capped at about 150 MB instead of 8 clips, and stepping through rows aborts the
    fetches it skips. Space on a focused play button plays that row, and the arrow keys
    move focus along with playback. Two quick empty searches no longer print the message
    twice. A bad BPM says "BPM" instead of "--bpm" and clears the old table; ticking all
    four stems says why that finds nothing. If the server has stopped, the page says so.
    Reveal works on exports from before a server restart. Volume and loop are remembered.
    The dock fits an 800 px window (200% zoom on a 1600 px screen).
  - The preview clip cache on disk was pruned only at startup, by count. It is now capped
    at 500 MB and pruned before each new clip.
  - `python -m breakdig` works. Dead parameters and fields are gone, `ui --help` explains
    its flags, and the README no longer promises "no click" at the cut edges: on low bass
    the snap still leaves up to 0.39 of full scale at the edge (p99 0.083).
  - The rebuilt zip installed into an empty folder: exit 0 in 238 s, torch sees the GPU,
    and `find --search "fidget kaos"` gave 13:107-108.
- Three features after the fourth pass:
  - `export --keep STEMS` and the UI's keep menu separate each section again, with 3 s of
    context either side, and write only those stems at the source rate. On the fixture's
    bars 1-4 (drums under a bassline and pad), the energy left over after subtracting the
    true drums is 1.5% of the drums with `--keep drums`, against 421% for the plain cut.
    Against a fake separator that hands the input back, the isolated clip matches the plain
    cut to 1e-8 at 44.1 kHz and within 2.4% RMS for a 48 kHz file above full scale, with no
    lag. On the 14 CC tracks, drums play alone for 2 or more bars in 4 bars of 2 tracks;
    they play at all in 1096 bars across 13 tracks, which is the pool `--keep drums` opens
    up. How clean the isolated drums are on real records has not been measured, since
    there is no ground truth for them, and nobody has listened to them yet. Eight real
    sections (7.5 minutes of audio) exported in 78 s on the 4090, the first preview in the
    UI in 9 s with the model load.
  - Every exported WAV has a RIFF `cue ` chunk with one cue per beat and `beat N` labels.
    mutagen's ID3 save keeps it, soundfile still reads the file, and the tests read the
    cues back. Which samplers and editors show them was not checked.
  - The UI draws each row's per-bar stem levels as a 4-row grid.
    `.scratch/r4/ui_verify5.py` in Playwright (9 checks, all pass) covers the grid, a real
    isolated preview and export with its tag, the remembered keep choice, and the 800 px
    dock; `ui_verify4.py` still passes.
  - Eight mutants of the new code (cues not written, gain not undone, stems left out of
    the export identity, wrong clip start, clip cache ignoring keep, no input gain, beats
    picked at the wrong edge, model loaded for a plain export) are each caught.
- `--sp404 sx` and the UI's format menu write 44.1 kHz for the SX, A and original 404. An
  SP-404 folder with clips at the other rate is refused before anything is written. The
  test resamples a 48 kHz source to 44.1 kHz, and a guard-removed mutant and a hard-coded
  48 kHz mutant both fail it. In Chrome, an SX export gave a 44.1 kHz BRK_0001.WAV, the
  choice survived a reload, and an MKII export into the same folder showed the refusal.
- Upgrading an index built with the old key: a copy of the E2E index re-indexed over
  `.scratch/cc-music` gave "14 updated" in 6 s with no separation, the next run did
  nothing in 1.5 s, and `find --json` matched the original byte for byte.
- Indexing, unattended:
  - 50 tracks (1.8 h of audio) at 9.4 s per separated track.
  - 64 tracks (2.5 h) at 11.3 s/track, 12.6x realtime.
  - A 21-minute MP3 in 89 s, split into 10-minute chunks with seam flags.
  - A rerun on an indexed folder does nothing.
  - Indexing is repeatable. Demucs separates at a random time offset, and two unseeded
    runs gave vocal bleed of -35 dB against -51 dB on the same bars, so a borderline
    section came and went. `separate.py` now seeds it. Two independent indexes of the
    SOSLP008 album then matched to 0.0000 dB on all 279 bars, and
    `test_separation_is_repeatable` guards it.
  - Killing the process mid-run and starting again carries on from where it stopped.
  - DRM .m4p and corrupt files are skipped and listed under `stats --failures`.
  - The same audio as MP3 and FLAC is separated once.
  - After grooveclean 1.2.1 (`batch --format flac`): its `.removed.flac` difference file
    for a break-heavy track has a beat grid and got indexed as a track, so `index --exclude`
    was added. With `--exclude "*.removed.*"` the cleaned folder scans 3 of 9 files and
    the break is still found. grooveclean removed nothing inside the break itself.
- find, export, `--sp404` and its rerun (skips what is there) on the 14 CC tracks in
  `.scratch/cc-music`: 189 s, 13.5 s/track, 14.4x realtime. The full transcript is
  `.scratch/final-e2e.txt`, and README.md's quick start is that run.
- The web UI, driven in Chrome with Playwright: filters, the table, a looped range preview,
  exporting selected rows, and reveal. No console errors. `docs/*.png` and `docs/demo.gif`
  come from `.scratch/capture.py` on the 64-track demo index, so their IDs differ from the
  README's. They were redone after the Stems column, keep menu and format menu went in.
  The gif's drums preview came from the preview cache, so it skips the few seconds a first
  separation takes.
- Packaging:
  - `python -m build` gives the wheel and an sdist (the sdist includes the tests), and
    `twine check` passes on both.
  - `python packaging/make_windows_zip.py` gives `dist/breakdig-0.1.0-windows.zip`.
  - The final zip was unzipped into an empty folder and its `install.bat` run: exit 0 in
    234 s, torch 2.14.0+cu130 with CUDA on the 4090. Running it again took 3 s and left
    torch alone. The installed `breakdig.bat` ran
    `--version`, `stats` and `find` against the E2E index. After the third pass the rebuilt
    zip was installed the same way (the first attempt was cut off mid-pip, and running
    `install.bat` again finished it: exit 0 in 98 s). audioread came in with the wheel,
    torch 2.14.0+cu130 sees the GPU, and `find --search "fidget kaos"` on the 14-track
    review index gave 13:107-108.
  - This PC sets `NoDefaultCurrentDirectoryInExePath`, so a bare `install.bat` or
    `breakdig` typed in that folder is not found. Double-clicking works. `install.bat`
    now prints the launcher's full path for that reason.
- Fourth review pass, each checked with the reviewer's own script before and after:
  - A lossy original followed by a lossless copy now exports from the WAV
    (`.scratch/review-core/lossy_first.py`).
  - Indexing 400 short loops keeps a flat time per file (0.20 s for the first files, 0.18 s
    for the last) where it grew from 0.15 s to 0.43 s, because `no_grid` loops no longer
    keep a profile to compare against (`.scratch/review-core/loops.py`).
  - The reviewer's .wma and LIST/INFO .wav read their tags, and a drive root with a
    Recycle Bin and `._` files indexes only the real tracks.
  - In Chromium (`.scratch/review-ui/pw3.py`), a new search that drops the playing row
    stops the preview and the next Space plays the new first row; a search with no rows
    stops it too.
  - `.scratch/abort_queue.py` against a live server on the review home: of five isolated
    previews abandoned while queued, only the one already separating was written, and the
    wanted preview came back in 5.6 s. `request.is_disconnected()` could not do this: the
    `@app.middleware` wrapper drops the disconnect message it polls for, so the endpoint
    waits on the next receive instead. `test_an_isolated_preview_the_page_gave_up_on_is_not_separated`
    fails without that check.
  - `ui --host 0.0.0.0` shows a 127.0.0.1 URL, `--host 192.0.2.7` says it cannot listen,
    a held port still says it is in use, `--shifts -1` is refused, bare `breakdig` prints
    help.
- House rules:
  - No em dashes anywhere in the shipped files.
  - The difflib clone check (`.scratch/clonecheck.py`) finds 2 function pairs at or above
    0.45: two index tests at 0.56 that share their setup, as before, and two duplicate
    takeover tests at 0.50. Nothing is at 0.6 or above. The check now skips a function
    nested inside another, which it used to score as a copy of its parent.
  - The query refactor was checked byte for byte against a recorded baseline
    (`.scratch/baseline.py`, 1106 sections).
- The published release: its zip holds the same wheel byte for byte, both .bat files are
  CRLF, twine check passes, the wheel installs and imports outside the repo, `pip
  download breakdig==0.1.0` fetches it from PyPI, and the README GIF loads from GitHub.

## Next steps for the owner

1. Post to r/SP404. The draft is `.scratch/post/r-sp404-draft.md`, with notes on the
   parts to check first.
2. Later releases: bump `__version__`, add a dated CHANGELOG section, push, wait for CI on
   that commit, then push a `v<version>` tag. The Release workflow makes a draft release;
   publish it, then `twine upload` its wheel and sdist (the PyPI token is in
   `~/.pypirc`).

## Known gaps and decisions left open

- The beat_this checkpoint (81 MB) goes to the PyTorch hub cache
  (`%USERPROFILE%\.cache\torch\hub\checkpoints`), not to `BREAKDIG_MODELS`, because
  beat_this downloads it itself. It lives outside the project folder on this PC as well.
- Demucs bleed is the main limit on hits. On the 14 CC tracks, the default
  `--silence-db 30` finds two 2-bar breaks, both in the fidget tracks, and 25 finds three
  sections of 2 to 3 bars. The 12 electro-swing tracks give none at either setting.
  Whether 30 or 25 should be the default is a judgement call; 30 keeps false positives out.
- The indexes in `.scratch/home-demo` and `.scratch/home-real` were built before the seed
  fix, so their bar levels are one random draw, and before the last-bar change, so they
  lack each track's final bar. Re-index them before using them for new screenshots.
- `find` still lists sections from files that have been deleted or moved outside every
  indexed folder. Export reports each one as "source missing". A `prune` command would fix
  this, but an unplugged drive looks just like deleted files, so it is left for a decision
  rather than built.
- A raw ADTS .aac copy of a track indexed in another format is separated again. It keeps
  its encoder delay, about half a 50 ms envelope frame, and at that offset the best whole
  frame lag correlates at 0.95 against the 0.99 needed. Averaging 4 frames lifts it to
  0.992, but also puts an unmastered and mastered pair of different files in home-real at
  0.992, held apart only by their 1.5 dB level difference. A miss costs one separation; a
  false match hides a track, so matching was left alone.
- Envelope frames are 50 ms, so a bar edge can be up to half a frame off in the levels. The
  audio reviewer rated it low; it only shifts which frames a bar averages.
- Two files that swap names while keeping the same size and modified time are not noticed,
  because unchanged stats are not read. Rare enough to leave.
- A section id is now always above every id in use, but if the newest track is forgotten
  and another indexed, its id can come back. Only matters for a UI tab left open across
  both.
- On Linux and macOS, a missing mount point looks like a deleted file, since the drive
  root test only helps with Windows drive letters.
- `--bpm 90` does not match tracks that beat_this counted at half or double time (45 or
  180). The audio reviewer suggested matching those too; not built.
- A track whose copy is detected as "updated" (retagged, or the same audio moved) keeps
  its old bar times; if the new file is offset by less than one 50 ms frame, the bars are
  off by that much. The old audio is not kept, so there is nothing to realign against.
- Killing a plain (non SP-404) export mid-file can leave a `.part` file in the output
  folder. The copies table keeps rows for copies that were deleted; they are harmless and
  only used to skip rehashing.
- Two small refactors were left: one "why is this empty" helper shared by the CLI and the
  web page (their messages differ in form, the page builds links), and simplifying the
  counters in `Indexer.pending`.
- CI only covers the CPU path. The gpu tests need CUDA, which hosted runners do not have.
- `--keep` always separates with the default model, even for an index built with `--model`.
  The UI loads the model on the first isolated preview and holds it (and its GPU memory)
  until the server stops. Cue labels count beats, not bar.beat.
- There is no search for "drums playing, whatever else is"; `--no vocals` plus the Stems
  column is the way to find bars for `--keep drums` today.

- Export does not warn when a source file changed after it was indexed. A retag changes the
  modified time too, so the warning would mostly fire for nothing.
- Filenames like `01_Artist_-_Title` are not split into artist and title; underscores are
  ambiguous (`50_Cent_-_In_Da_Club`).
- An MP2 copy of the synthetic test fixture is not recognised as a duplicate of the WAV:
  its envelope correlates at 0.987 against the 0.99 needed. Real music is denser and was not
  tried. .ape and .mpc are listed but untested end to end, since ffmpeg cannot write them.
- Not checked on hardware: whether an SP-404 accepts WAVs with cue and id3 chunks, and
  paths past MAX_PATH on Windows.
- Ideas from the UI review, not built: a warning in the page header when the CPU build of
  torch is in use, and a Stop button in the player dock.

## Features not built (worth considering after 0.1.0)

- A "with" search mode (these stems play, the rest can do anything), scored by how loud
  the wanted stems are, to feed `--keep`.
- `breakdig prune` (see above), and `index --watch` to pick up new files in a folder.
- Tempo-matched export: time-stretch every clip to one BPM for a kit.
- macOS: runs on the CPU only, untested.
