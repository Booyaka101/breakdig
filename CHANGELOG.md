# Changelog

## Unreleased

- Reads WavPack, Monkey's Audio, Musepack, MP2 and AIFC files too.
- Artist and title come from WAV LIST/INFO and WMA tags, which were ignored before.
- Indexing a drive root skips the Recycle Bin, System Volume Information and the `._` files
  macOS leaves on FAT and exFAT drives.
- A lossless copy of a track first indexed from an MP3 or other lossy file now takes its
  place, so sections are cut from the lossless file. The lossy one becomes its duplicate.
- The duplicate check no longer slows down as a sample pack of short loops is indexed, and
  long mixes use less memory during beat tracking.
- `breakdig ui --host 0.0.0.0` prints and opens a URL a browser can use, and a host that
  cannot be bound says so instead of claiming the port is in use.
- In the UI, a preview stops when a new search no longer lists its row, and isolated
  previews you skipped past while they queued are dropped instead of separated, so the one
  you want is not kept waiting.
- `--shifts` rejects negative numbers up front, and `breakdig` on its own prints the help.

## 0.1.0 (2026-09-29)

First release.

- `breakdig index` separates each track into drums, bass, vocals and other with Demucs
  (htdemucs_ft through audio-separator), finds downbeats with beat_this, and stores
  per-bar stem levels in a local SQLite index. Resumable and idempotent; the same audio
  in another format or at another path is recognised and not separated twice, and a
  retagged, moved or renamed file keeps its analysis. `--exclude` skips files and folders
  by name, and a second run on the same index stops rather than racing the first. Reads
  MP3, FLAC, WAV, AIFF, M4A, AAC, OGG, Opus and WMA.
- `breakdig find` lists bar-exact sections by what is playing: drums only, vocals only,
  bass and drums only, anything with no drums, and so on. Filters for length, tempo and
  words in the tags or path; sorted by how clean the section is, and `--cleaner-than`
  sets a floor.
  `--silence-db` loosens the silence rule when separation bleed hides sections you can hear.
- `breakdig export` cuts sections from the original file at the bar lines, snapped to
  the quietest point within 2 ms, as tagged WAV. `--sp404` writes 16-bit 48 kHz files named
  BRK_0001.WAV with an index.csv, and `--sp404 sx` writes 44.1 kHz for the SX, A and
  original 404. Exporting again overwrites or skips what is there, even
  after the library moves, and relists clips from an export that was cut off.
- `breakdig export --keep drums` (or `drums,bass`, and so on) exports only those stems:
  each section is separated again with a few seconds either side and put back at the
  source sample rate, so the drums can be taken out of bars where a bassline or pad plays
  over them.
- Exported WAVs have a cue point on every beat.
- `breakdig ui` opens a local web UI to filter, audition and export sections. Previews loop
  with no gap, ticked rows are kept across searches, and the filters, volume and loop
  setting are remembered. Each row shows a small grid of the four stems' levels per bar,
  and a keep menu previews and exports just the stems you pick.
- `breakdig stats` shows what is indexed, what failed and why, and indexing speed.
- `python -m breakdig` works as well as the `breakdig` command.
- `find --json` includes each section's beat times and per-bar stem levels.
