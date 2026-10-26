"""
Candidate scoring and the decision for a word: frame log-probabilities (T, 4) + nucleus times → scores → verdict.

A candidate is a Mark or None (unstressed). Expected class for nucleus k: the stressed one → accent + 1, others → 0.
Frame i is centred at (i + 0.5) * frame_s; a frame belongs to nucleus [s, e) if its centre lies in it.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ltstress.phonetics import Mark, nucleus_type


@dataclass
class Thresholds:
    confident: float            # min margin (best vs. second best) for a confident "match"
    position: float
    pitch_accent: dict[str, float] = field(default_factory=dict)
    min_alignment: float = 0.2

    @classmethod
    def default(cls) -> "Thresholds":
        return cls(0.5, 0.5, {}, 0.2)

    @classmethod
    def load(cls, path) -> "Thresholds":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(d["confident"]), float(d["position"]),
                   {k: float(v) for k, v in d["pitch_accent"].items()}, float(d["min_alignment"]))

    def to_dict(self) -> dict:
        return {"confident": self.confident, "position": self.position, "pitch_accent": dict(self.pitch_accent),
                "min_alignment": self.min_alignment}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def pitch_accent_threshold(self, nucleus: str) -> float:
        return self.pitch_accent.get(nucleus, self.position)


@dataclass
class Decision:
    best: Mark | None
    margin: float
    verdict: str                # "match" | "mismatch" | "unclear"
    kind: str | None            # "stress_position" | "pitch_accent" | None (check also: "function_word")
    reason: str | None          # "low_alignment" | "no_candidate" | "low_margin" | None


def _nucleus_frames(centres: np.ndarray, s: float, e: float) -> np.ndarray:
    """Indices of the frames in nucleus [s, e); if no centre falls inside – the frame nearest to the interval middle."""
    idx = np.flatnonzero((centres >= s) & (centres < e))
    if idx.size:
        return idx
    return np.array([int(np.argmin(np.abs(centres - (s + e) / 2)))])


def candidate_scores(logp: np.ndarray, nuclei: list[tuple[float, float]],
                     candidates: list[Mark | None], frame_s: float) -> dict[Mark | None, float]:
    """Score of each candidate = mean over nuclei of the mean of logp[frame, expected class] over the nucleus frames."""
    centres = (np.arange(len(logp)) + 0.5) * frame_s
    frames = [_nucleus_frames(centres, s, e) for s, e in nuclei]
    scores = {}
    for m in candidates:
        classes = [(m[1] + 1 if m is not None and m[0] == k else 0) for k in range(len(nuclei))]
        scores[m] = float(np.mean([logp[idx, cl].mean() for idx, cl in zip(frames, classes)]))
    return scores


def _best_and_margin(scores: dict) -> tuple[Mark | None, float]:
    """The best candidate and its margin over the second best (a single candidate → +inf)."""
    ranked = sorted(scores, key=scores.get, reverse=True)
    b = ranked[0]
    return b, (scores[b] - scores[ranked[1]]) if len(ranked) > 1 else float("inf")


def decide(scores: dict, text: Mark | None, word: str, thresholds: Thresholds,
           alignment_score: float) -> Decision:
    """Decision for a word against its text mark: match / mismatch / unclear."""
    b = max(scores, key=scores.get) if scores else None
    if alignment_score < thresholds.min_alignment:
        return Decision(b, 0.0, "unclear", None, "low_alignment")
    if text not in scores:
        return Decision(b, 0.0, "unclear", None, "no_candidate")
    b, second_margin = _best_and_margin(scores)
    if b == text:
        ok = second_margin >= thresholds.confident
        return Decision(b, second_margin, "match" if ok else "unclear", None, None if ok else "low_margin")
    margin = scores[b] - scores[text]
    if b is not None and text is not None and b[0] == text[0]:
        kind, threshold = "pitch_accent", thresholds.pitch_accent_threshold(nucleus_type(word, text[0]))
    else:
        kind, threshold = "stress_position", thresholds.position
    ok = margin >= threshold
    return Decision(b, margin, "mismatch" if ok else "unclear", kind, None if ok else "low_margin")


def annotation_decision(scores: dict, thresholds: Thresholds) -> Decision:
    """Without a text mark: match = "confident" (margin over the second best ≥ confident), otherwise unclear."""
    b, margin = _best_and_margin(scores)
    ok = margin >= thresholds.confident
    return Decision(b, margin, "match" if ok else "unclear", None, None if ok else "low_margin")
