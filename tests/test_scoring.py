import json
import math

import numpy as np

from ltstress.phonetics import variants
from ltstress.scoring import Thresholds, annotation_decision, candidate_scores, decide


def _logp(T, frames_class):  # log(0.97) for class 0 everywhere, the given frames – for the given class
    p = np.full((T, 4), 0.01); p[:, 0] = 0.97
    for (a, b), k in frames_class:
        p[a:b] = 0.01; p[a:b, k] = 0.97
    return np.log(p)

NUC = [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5)]  # 3 syllables, frame_s 0.02


def test_best_circumflex_on_second():
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    assert max(s, key=s.get) == (1, 2)


def test_pitch_accent_mismatch():
    lp = _logp(30, [((10, 15), 3)])
    s = candidate_scores(lp, NUC, variants("dalykus"), 0.02)
    d = decide(s, (1, 1), "dalykus", Thresholds.default(), 0.9)
    assert (d.verdict, d.kind) == ("mismatch", "pitch_accent")


def test_position_mismatch():
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    assert decide(s, (2, 0), "dalykus", Thresholds.default(), 0.9).kind == "stress_position"


def test_match_and_unclear():
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    assert decide(s, (1, 2), "dalykus", Thresholds.default(), 0.9).verdict == "match"
    high_confident = Thresholds(confident=99.0, position=0.5, pitch_accent={}, min_alignment=0.2)
    assert decide(s, (1, 2), "dalykus", high_confident, 0.9).verdict == "unclear"


def test_bad_alignment():
    s = candidate_scores(_logp(30, []), NUC, variants("dalykus"), 0.02)
    d = decide(s, None, "dalykus", Thresholds.default(), 0.05)
    assert (d.verdict, d.reason) == ("unclear", "low_alignment")


def test_nucleus_without_frames():
    s = candidate_scores(_logp(30, [((5, 6), 1)]), [(0.101, 0.102)], [None, (0, 0)], 0.02)
    assert all(np.isfinite(v) for v in s.values())


def test_nucleus_without_frames_uses_nearest_frame():
    # nucleus middle 0.1015 s – the nearest centre is 0.11 (frame 5)
    s = candidate_scores(_logp(30, [((5, 6), 1)]), [(0.101, 0.102)], [None, (0, 0)], 0.02)
    assert s[(0, 0)] > s[None]


def test_score_is_frame_mean():
    lp = _logp(30, [((10, 12), 3)])
    s = candidate_scores(lp, [(0.2, 0.3), (0.4, 0.5)], [(0, 2)], 0.02)
    # 1st nucleus: frames 10..14 (centres 0.21..0.29): 2 frames in class 3, 3 frames – 0.01
    first = (2 * math.log(0.97) + 3 * math.log(0.01)) / 5
    # 2nd nucleus (frames 20..24) expects 0 -> log 0.97
    assert math.isclose(s[(0, 2)], (first + math.log(0.97)) / 2)


def test_no_candidate():
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    d = decide({k: v for k, v in s.items() if k != (2, 0)}, (2, 0), "dalykus", Thresholds.default(), 0.9)
    assert (d.verdict, d.reason) == ("unclear", "no_candidate")


def test_single_candidate_margin_infinite():
    d = decide({(0, 0): -0.1}, (0, 0), "dalykus", Thresholds.default(), 0.9)
    assert d.verdict == "match" and d.margin == math.inf


def test_too_small_margin_unclear_not_mismatch():
    s = {(0, 0): -1.0, (1, 0): -1.1}
    d = decide(s, (1, 0), "dalykus", Thresholds.default(), 0.9)
    assert (d.verdict, d.kind, d.reason) == ("unclear", "stress_position", "low_margin")


def test_unstressed_text_vs_stressed():
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    d = decide(s, None, "dalykus", Thresholds.default(), 0.9)
    assert (d.verdict, d.kind, d.best) == ("mismatch", "stress_position", (1, 2))


def test_pitch_accent_threshold_by_nucleus_type():
    from ltstress.phonetics import nucleus_type
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    nt = nucleus_type("dalykus", 1)
    th = Thresholds(confident=0.5, position=0.5, pitch_accent={nt: 99.0}, min_alignment=0.2)
    assert decide(s, (1, 1), "dalykus", th, 0.9).verdict == "unclear"
    assert decide(s, (2, 0), "dalykus", th, 0.9).verdict == "mismatch"


def test_annotation_decision():
    s = candidate_scores(_logp(30, [((10, 15), 3)]), NUC, variants("dalykus"), 0.02)
    d = annotation_decision(s, Thresholds.default())
    assert (d.best, d.verdict, d.kind) == ((1, 2), "match", None)
    assert annotation_decision(s, Thresholds(99.0, 0.5, {}, 0.2)).verdict == "unclear"


def test_thresholds_json(tmp_path):
    th = Thresholds(confident=1.5, position=0.7, pitch_accent={"long": 0.9}, min_alignment=0.3)
    path = tmp_path / "th.json"
    th.save(path)
    assert set(json.loads(path.read_text())) == {"confident", "position", "pitch_accent", "min_alignment"}
    assert Thresholds.load(path) == th
    assert Thresholds.default() == Thresholds(0.5, 0.5, {}, 0.2)
    assert th.pitch_accent_threshold("long") == 0.9 and th.pitch_accent_threshold("other") == 0.7
