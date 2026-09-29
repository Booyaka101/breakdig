# Changelog

## 0.1.0 (unreleased)

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
