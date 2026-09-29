import math

import numpy as np
import pytest

from breakdig.query import (PRESETS, Section, activity, bpm_between, cleanliness, crosses_seam,
                            matches, only, parse_bpm, parse_stems, runs, search_tracks, sections_in_track,
                            without)

# Columns: drums, bass, vocals, other.
FULL = [-20.0, -22.0, -18.0, -24.0]
DRUMS_ALONE = [-20.0, -70.0, -80.0, -65.0]
ACAPELLA = [-75.0, -80.0, -18.0, -60.0]
SILENCE = [-120.0, -120.0, -120.0, -120.0]


def mix_of(levels):
    return 10 * np.log10(np.sum(10 ** (np.asarray(levels) / 10), axis=-1))


def bars_array(level_rows, bar_seconds=2.0):
    """A track's bar array (start, end, drums, bass, vocals, other, mix) from stem levels."""
    levels = np.asarray(level_rows, dtype=float)
    n = len(levels)
    starts = np.arange(n) * bar_seconds
    return np.column_stack([starts, starts + bar_seconds, levels, mix_of(levels)])


def beats_for(n_bars, bpm=120.0):
    return np.arange(n_bars * 4 + 1) * 60.0 / bpm


def test_worked_example_levels_give_bars_5_to_8():
    rows = [FULL] * 4 + [DRUMS_ALONE] * 4 + [FULL] * 8
    found = sections_in_track(bars_array(rows), beats_for(16), PRESETS["drums"])
    assert len(found) == 1
    s = found[0]
    assert (s.first_bar, s.last_bar, s.bars) == (5, 8, 4)
    assert s.start == pytest.approx(8.0) and s.end == pytest.approx(16.0)
    assert s.bpm == pytest.approx(120.0)
    assert s.clean == pytest.approx(-65.0 - -20.0)


def test_active_needs_to_be_within_12_db_of_the_mix():
    levels = np.array([[-20.0, -31.0, -33.0, -60.0]])
    mix = np.array([-20.0])
    active, silent = activity(levels, mix)
    assert active[0].tolist() == [True, True, False, False]
    # -60 is 40 dB under the loudest active stem, so it is silent; -33 is neither.
    assert silent[0].tolist() == [False, False, False, True]


def test_a_stem_far_under_its_usual_level_is_not_silent_if_the_bar_is_quiet():
    # A quiet intro: vocals 15 dB under the drums are audible, however loud the vocals are later.
    rows = [[-30.0, -95.0, -45.0, -95.0]] * 4 + [[-5.0, -95.0, -10.0, -95.0]] * 8
    assert sections_in_track(bars_array(rows), beats_for(12), PRESETS["drums"], min_bars=1) == []


def test_lower_silence_threshold_never_loses_a_section():
    # The quiet intro of the target stem used to drop out as the threshold went down.
    rows = [[-40.0, -95.0, -95.0, -95.0]] * 4 + [[-5.0, -95.0, -95.0, -95.0]] * 8
    spans = [[(s.first_bar, s.last_bar) for s in sections_in_track(
        bars_array(rows), beats_for(12), PRESETS["drums"], silent_db=db)] for db in (30, 20, 15)]
    assert spans == [[(1, 12)]] * 3


def test_digital_silence_has_no_active_stems():
    levels = np.array([SILENCE])
    active, silent = activity(levels, mix_of(levels))
    assert not active.any()
    assert silent.all()
    assert not matches(PRESETS["drums"], active, silent).any()


def test_digital_silence_in_a_track_that_never_has_vocals():
    levels = np.array([SILENCE])
    active, silent = activity(levels, mix_of(levels))
    assert silent.all()
    assert not matches(PRESETS["vocals"], active, silent).any()


def test_bleed_between_the_two_limits_blocks_a_match():
    # Bass 25 dB under the drums is neither active nor silent: not "drums only".
    levels = np.array([[-20.0, -45.0, -90.0, -90.0]])
    active, silent = activity(levels, mix_of(levels))
    assert not matches(PRESETS["drums"], active, silent)[0]


def test_bars_with_a_broken_grid_are_skipped():
    bars = bars_array([FULL] * 2 + [DRUMS_ALONE] * 6 + [FULL] * 2)
    # Bar 4 is a doubled downbeat and bar 7 spans a stretch the tracker found no downbeats in.
    lengths = np.array([2.0, 2.0, 2.0, 0.1, 2.0, 2.0, 30.0, 2.0, 2.0, 2.0])
    bars[:, 1] = np.cumsum(lengths)
    bars[:, 0] = bars[:, 1] - lengths
    found = sections_in_track(bars, beats_for(10), PRESETS["drums"], min_bars=1)
    assert [(s.first_bar, s.last_bar) for s in found] == [(3, 3), (5, 6), (8, 8)]


def test_lower_silence_threshold_lets_bleed_through():
    rows = [FULL] * 2 + [[-20.0, -45.0, -90.0, -90.0]] * 4 + [FULL] * 2
    assert sections_in_track(bars_array(rows), beats_for(8), PRESETS["drums"]) == []
    (s,) = sections_in_track(bars_array(rows), beats_for(8), PRESETS["drums"], silent_db=20.0)
    assert (s.first_bar, s.last_bar, s.clean) == (3, 6, -25.0)


def test_multi_stem_and_without_patterns():
    rows = np.array([[-20.0, -22.0, -80.0, -80.0],   # bass and drums
                     [-80.0, -22.0, -18.0, -24.0],   # no drums
                     SILENCE])
    active, silent = activity(rows, mix_of(rows))
    assert matches(PRESETS["bass+drums"], active, silent).tolist() == [True, False, False]
    # A --no pattern still needs something to be playing.
    assert matches(PRESETS["no-drums"], active, silent).tolist() == [False, True, False]


def test_cleanliness_targets():
    rows = np.array([[-20.0, -26.0, -70.0, -60.0]])
    # Quietest required stem is the target for a multi-stem pattern.
    assert cleanliness(PRESETS["bass+drums"], rows)[0] == pytest.approx(-60.0 - -26.0)
    # For --no, the target is the loudest allowed stem.
    assert cleanliness(without({"vocals"}), rows)[0] == pytest.approx(-70.0 - -20.0)


def test_section_cleanliness_is_its_worst_bar():
    rows = [FULL, [-20.0, -70.0, -80.0, -65.0], [-20.0, -60.0, -80.0, -55.0], FULL]
    (s,) = sections_in_track(bars_array(rows), beats_for(4), PRESETS["drums"])
    assert s.clean == pytest.approx(-55.0 - -20.0)


def test_runs():
    mask = [1, 1, 0, 1, 0, 1, 1, 1]
    assert runs(mask, 1) == [(0, 1), (3, 3), (5, 7)]
    assert runs(mask, 2) == [(0, 1), (5, 7)]
    assert runs(mask, 3) == [(5, 7)]
    assert runs([], 1) == []
    assert runs([1, 1, 1], 2) == [(0, 2)]


def test_min_bars_filters_short_runs():
    rows = [FULL, DRUMS_ALONE, FULL, DRUMS_ALONE, DRUMS_ALONE, FULL]
    track = bars_array(rows)
    assert [s.id for s in sections_in_track(track, beats_for(6), PRESETS["drums"], 1, track_id=3)] == \
        ["3:2-2", "3:4-5"]
    assert [s.id for s in sections_in_track(track, beats_for(6), PRESETS["drums"], 2, track_id=3)] == ["3:4-5"]


def test_empty_track():
    assert sections_in_track(np.zeros((0, 7)), np.zeros(0), PRESETS["drums"]) == []


def test_bpm_from_median_interval_survives_one_bad_beat():
    beats = np.array([0.0, 0.5, 1.0, 1.43, 2.0, 2.5, 3.0])
    assert bpm_between(beats, 0.0, 3.0) == pytest.approx(120.0)
    assert bpm_between(beats, 10.0, 12.0) is None


def test_three_four_bars_come_from_downbeats():
    # Waltz at 90 BPM: bars are 2 s, three beats each.
    beats = np.arange(0, 20.01, 60 / 90)
    rows = [FULL] * 2 + [DRUMS_ALONE] * 3 + [FULL] * 5
    (s,) = sections_in_track(bars_array(rows), beats, PRESETS["drums"])
    assert (s.first_bar, s.last_bar) == (3, 5)
    assert s.bpm == pytest.approx(90.0)


def test_tempo_drift_uses_real_bar_boundaries():
    # Bars stretch from 2.0 s to 2.4 s; the section must follow the real downbeats.
    lengths = np.linspace(2.0, 2.4, 8)
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    levels = np.array([FULL] * 3 + [DRUMS_ALONE] * 3 + [FULL] * 2)
    track = np.column_stack([starts, starts + lengths, levels, mix_of(levels)])
    (s,) = sections_in_track(track, np.zeros(0), PRESETS["drums"])
    assert s.start == pytest.approx(starts[3]) and s.end == pytest.approx(starts[5] + lengths[5])
    assert s.bpm is None


def test_crosses_seam():
    assert crosses_seam(590.0, 610.0, 1500.0, 600.0)
    assert crosses_seam(1195.0, 1205.0, 1500.0, 600.0)
    assert not crosses_seam(600.0, 610.0, 1500.0, 600.0)  # starting on a seam is fine
    assert not crosses_seam(580.0, 600.0, 1500.0, 600.0)
    assert not crosses_seam(590.0, 610.0, 1500.0, None)


def test_seam_flag_on_sections():
    rows = [FULL] + [DRUMS_ALONE] * 4 + [FULL]
    (s,) = sections_in_track(bars_array(rows, 200.0), np.zeros(0), PRESETS["drums"],
                             duration=1200.0, chunk_seconds=600.0)
    assert s.start == 200.0 and s.end == 1000.0 and s.seam


def test_patterns_validate_input():
    assert only(parse_stems("bass+drums")) == PRESETS["bass+drums"]
    assert only(parse_stems(" Drums , BASS ")) == PRESETS["bass+drums"]
    assert PRESETS["no-drums"].name == "no drums"
    assert PRESETS["bass+drums"].name == "drums+bass only"
    for bad in ["", "guitar", "drums,bass,vocals,other"]:
        with pytest.raises(ValueError):
            only(parse_stems(bad))
        with pytest.raises(ValueError):
            without(parse_stems(bad))


def test_parse_bpm():
    assert parse_bpm(None) is None
    assert parse_bpm("") is None
    assert parse_bpm("90") == (89.5, 90.5)
    assert parse_bpm("85-100") == (85.0, 100.0)
    assert parse_bpm("100-85") == (85.0, 100.0)
    with pytest.raises(ValueError, match="--bpm"):
        parse_bpm("fast")


def test_section_id_and_dict():
    s = Section(7, 5, 8, 8.0, 16.0, 120.0, -30.0, False, "A", "B")
    assert s.id == "7:5-8"
    d = s.to_dict()
    assert d["id"] == "7:5-8" and d["bars"] == 4 and not math.isnan(d["clean"])


def test_search_needs_every_word_somewhere():
    t = {"artist": "The Winstons", "title": "Amen, Brother", "album": None, "path": "D:/Funk/amen.flac"}
    assert search_tracks([t], "winstons amen") == [t]
    assert search_tracks([t], "WINSTONS  funk") == [t]
    assert search_tracks([t], "winstons soul") == []
