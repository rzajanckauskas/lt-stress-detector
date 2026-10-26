"""
Check: manifest (id, audio_path, stress-marked text) -> a verdict for every word (JSONL).

Run: ltstress check --input X.jsonl --output Y.jsonl [--model renaldux/lt-stress-detector | DIR] \
    [--thresholds thresholds.json] [--audio-dir DIR] [--annotate] [--device cuda|cpu]

Input line: {"id", "audio_path", "text", "candidates"?: {"<word index>": ["form", ...]}} ;
a line without a non-empty `id` is an error reporting the line number (nothing is processed).
Output format "ltstress-check-1" – FORMATS.md. A word's `word` is as in the text (`text[s0:s1]`),
`text_form` and the forms in `scores` / `best_form` are in model form (NFC letters + separate U+0300/0301/0303;
precomposed "à" is decomposed, "i̇" U+0307 is dropped). The output is written line by line; on a re-run, ids whose
last line has no `error` are skipped, while records with an `error` (missing audio, OOM etc.) are retried – the new
line replaces the old one (readers take the last line of each id). Words – every word found in the text
(letters/digits + stress marks, hyphenated counts as one word); words where nothing can be decided get "unclear"
with a reason:
  no_letters           – the word has no letters from the aligner vocabulary (a number, "ō" etc.);
  multiple_marks       – several stress marks (e.g. "kõkį-nórs");
  mark_outside_nucleus – a mark on a consonant outside the syllable nucleus ("aš̃", "pirm̀oji");
  mark_not_canonical   – a mark inside the nucleus but not on the right letter ("dar̀", "jĩe");
                         in both cases scores and best_form are still computed (an acoustic suggestion), the margin
                         is best vs. second best; such a word never gets "match";
  no_nucleus           – the word has no syllable nucleus (an abbreviation without vowels);
  alignment_error      – the whole sentence could not be aligned (all words);
  low_alignment        – the word's alignment score is too low;
  low_margin / no_candidate – from scoring.decide.
A stressed↔unstressed mismatch on a function word (phonetics.FUNCTION_WORDS), i.e. exactly one of the text mark and
the best candidate is None, gets the kind "function_word" instead of "stress_position" (the verdict is unchanged):
their stress is barely audible, so aggregate puts them at the end of the queue. Other function-word mismatches
(both stressed) get the usual kind.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf

from ltstress.alignment import AlignmentError, Aligner
from ltstress.phonetics import (MultipleMarksError, is_function_word, mark_error, marked_form, nucleus_type,
                                strip_marks, to_model_form, variants, word_mark)
from ltstress.scoring import Decision, Thresholds, annotation_decision, candidate_scores, decide

FORMAT = "ltstress-check-1"
SAMPLE_RATE = 16_000
MIN_SAMPLES = SAMPLE_RATE // 10          # 0.1 s; shorter audio is a record error
# letters / digits and combining marks (U+0300–U+036F: stress marks, also U+0307 etc.); a hyphen joins
WORD = re.compile(r"[\ẁ-ͯ]+(?:-[\ẁ-ͯ]+)*")


def words_in_text(text: str) -> list[str]:
    """The words of the text as they appear in it (output `word`); model form – to_model_form(word)."""
    return WORD.findall(text)


def word_spans(text: str) -> list[tuple[str, int, int]]:
    """(word, s0, s1): word = text[s0:s1] (character offsets in the input `text` string)."""
    return [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]


def _num(x: float | None, digits: int = 4) -> float | None:
    """Rounded number for strict JSON: ±inf / NaN -> None (e.g. the margin of a single candidate)."""
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), digits)


# ---------------------------------------------------------------------------
# Frame model
# ---------------------------------------------------------------------------
class FrameModel:
    """Frame stress classifier loaded from a model directory (model + preprocessor_config.json)."""
    frame_s = 0.02

    def __init__(self, path: Path, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
        self._cls = Wav2Vec2BertForAudioFrameClassification
        self.path = Path(path)
        self.device = device
        self.features = AutoFeatureExtractor.from_pretrained(str(path))
        self.model = self._cls.from_pretrained(str(path)).to(device).eval()
        self._cpu = None if device.startswith("cuda") else self.model

    @torch.inference_mode()
    def _logits(self, model, device: str, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Labels are not passed: the internal loss expects one-hot labels.
        x, mask = x.to(device), mask.to(device)
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                return model(input_features=x, attention_mask=mask).logits.float().cpu()
        return model(input_features=x, attention_mask=mask).logits.float()

    def logp(self, audio: np.ndarray) -> np.ndarray:
        """log_softmax (T, 4) float32; T – number of feature extractor frames (50 Hz). GPU OOM -> CPU."""
        p = self.features(np.asarray(audio, dtype=np.float32), sampling_rate=SAMPLE_RATE, return_attention_mask=True,
                          return_tensors="pt")
        x, mask = p["input_features"], p["attention_mask"]
        if self._cpu is self.model:
            logits = self._logits(self.model, "cpu", x, mask)
        else:
            try:
                logits = self._logits(self.model, self.device, x, mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self._cpu is None:
                    self._cpu = self._cls.from_pretrained(str(self.path)).eval()
                logits = self._logits(self._cpu, "cpu", x, mask)
        return torch.log_softmax(logits[0], dim=-1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self, aligner, frame_model, thresholds: Thresholds, annotate: bool = False,
                 model_name: str | None = None, thresholds_path: str | None = None):
        self.aligner = aligner
        self.frame_model = frame_model
        self.thresholds = thresholds
        self.annotate = annotate
        self.model_name = model_name
        self.thresholds_path = thresholds_path

    def _row(self, record: dict) -> dict:
        return {"format": FORMAT, "id": record.get("id"), "audio_path": record.get("audio_path"),
                "model": self.model_name, "thresholds": self.thresholds.to_dict(),
                "thresholds_path": self.thresholds_path, "error": None, "words": []}

    def error_row(self, record: dict, error: str) -> dict:
        return {**self._row(record), "error": error}

    @staticmethod
    def _nuclei(timing) -> list[list[float]] | None:
        return [[_num(s, 3), _num(e, 3)] for s, e in timing.nuclei] if timing else None

    @staticmethod
    def _unclear(i: int, span: tuple[str, int, int], reason: str, timing=None, text=None) -> dict:
        word, s0, s1 = span
        return {"i": i, "word": word, "s0": s0, "s1": s1,
                "t0": _num(timing.t0, 3) if timing else None, "t1": _num(timing.t1, 3) if timing else None,
                "text_form": to_model_form(word), "text_mark": list(text) if text is not None else None,
                "best_form": None, "scores": {}, "margin": None,
                "verdict": "unclear", "kind": None, "nucleus_type": None,
                "nuclei": Checker._nuclei(timing),
                "alignment_score": _num(timing.score) if timing else None, "reason": reason}

    def _candidates(self, plain: str, extra: list[str], text=None) -> list:
        """variants + the text mark (if variants do not produce it – otherwise no_candidate) + extra ones from the
        input (only forms of the same word: strip_marks(f) == plain, ignoring letter case)."""
        cand = variants(plain)
        if text is not None and text not in cand:
            cand.append(text)
        for f in extra:
            f = to_model_form(f)
            if strip_marks(f).lower() != plain.lower():
                continue
            try:
                m = word_mark(f)
            except MultipleMarksError:
                continue
            if m not in cand:
                cand.append(m)
        return cand

    def check_record(self, record: dict, audio: np.ndarray) -> dict:
        row = self._row(record)
        spans = word_spans(record.get("text") or "")
        extra = {int(k): v for k, v in (record.get("candidates") or {}).items()}
        if not spans:
            return row

        # Word properties that do not depend on the audio (marks in model form: no U+0307, decomposed)
        text: dict[int, object] = {}
        reason: dict[int, str] = {}
        bad_mark: dict[int, str] = {}
        for i, (w, _, _) in enumerate(spans):
            if not any(c.isalpha() for c in w):
                reason[i] = "no_letters"
                continue
            try:
                text[i] = word_mark(to_model_form(w))
            except MultipleMarksError:
                reason[i] = "multiple_marks"
                continue
            if (me := mark_error(to_model_form(w))) is not None:
                bad_mark[i] = me

        def tm(i: int):                                    # output text_mark
            return None if i in bad_mark else text.get(i)

        try:
            timings = self.aligner.align(audio, [strip_marks(w).lower() for w, _, _ in spans])
        except AlignmentError:
            row["words"] = [self._unclear(i, s, reason.get(i, "alignment_error"), text=tm(i))
                            for i, s in enumerate(spans)]
            return row

        logp = None
        for i, s in enumerate(spans):
            wt, w = timings[i], s[0]
            if i in reason or wt is None:
                row["words"].append(self._unclear(i, s, reason.get(i, "no_letters"), wt, tm(i)))
                continue
            if not wt.nuclei:
                row["words"].append(self._unclear(i, s, "no_nucleus", wt, tm(i)))
                continue
            if logp is None:
                logp = self.frame_model.logp(audio)
            plain = strip_marks(w)
            scores = candidate_scores(logp, wt.nuclei, self._candidates(plain, extra.get(i, []), text[i]),
                                      self.frame_model.frame_s)
            row["words"].append(self._word(i, s, plain, text[i], scores, wt, bad_mark.get(i)))
        return row

    def _word(self, i: int, span: tuple[str, int, int], plain: str, text, scores: dict, wt,
              bad_mark: str | None = None) -> dict:
        th = self.thresholds
        w, s0, s1 = span
        if bad_mark is not None:
            # malformed mark: never "match"; acoustic suggestion – the best candidate and its margin over the second
            b = annotation_decision(scores, th)
            d = Decision(b.best, b.margin, "unclear", None, bad_mark)
        elif not self.annotate:
            d = decide(scores, text, plain.lower(), th, wt.score)
        elif wt.score < th.min_alignment:
            d = Decision(max(scores, key=scores.get), 0.0, "unclear", None, "low_alignment")
        else:
            d = annotation_decision(scores, th)
        if d.kind == "stress_position" and (text is None) != (d.best is None) and is_function_word(plain):
            d = Decision(d.best, d.margin, d.verdict, "function_word", d.reason)
        syllable = text[0] if text is not None else (d.best[0] if d.best is not None else None)
        return {"i": i, "word": w, "s0": s0, "s1": s1, "t0": _num(wt.t0, 3), "t1": _num(wt.t1, 3),
                "text_form": to_model_form(w),
                "text_mark": list(text) if text is not None and bad_mark is None else None,
                "best_form": marked_form(plain, d.best),
                "scores": {marked_form(plain, k): _num(v) for k, v in scores.items()},
                "margin": _num(d.margin), "verdict": d.verdict, "kind": d.kind,
                "nucleus_type": nucleus_type(plain, syllable) if syllable is not None else None,
                "nuclei": self._nuclei(wt),
                "alignment_score": _num(wt.score), "reason": d.reason}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def read_audio(path: Path) -> np.ndarray:
    """Audio as mono 16 kHz float32 (soundfile; channels averaged; other sample rates are resampled)."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = taf.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def audio_file(audio_path: str, audio_dir: Path | None) -> Path:
    """The manifest path; if it does not exist and a directory is given – directory / file name."""
    p = Path(audio_path)
    if not p.exists() and audio_dir is not None:
        return Path(audio_dir) / p.name
    return p


def already_done(output: Path) -> set[str]:
    """Ids completed successfully in the output (last line of the id without `error`; those with an error are
    retried); a truncated last line (no "\\n") is cut off so that resuming does not corrupt it."""
    if not output.exists():
        return set()
    data = output.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
        output.write_bytes(data)
    last: dict[str, bool] = {}
    for line in data.decode("utf-8").splitlines():
        try:
            e = json.loads(line)
            last[str(e["id"])] = e.get("error") is None
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue
    return {i for i, ok in last.items() if ok}


def read_input(input_path: Path):
    with open(input_path, encoding="utf-8") as f:
        for nr, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{input_path}:{nr}: invalid JSON line ({e})")
            if not isinstance(record, dict) or record.get("id") in (None, ""):
                raise SystemExit(f"{input_path}:{nr}: line has no \"id\" (every record needs a non-empty id)")
            yield record


HF_MODEL = "renaldux/lt-stress-detector"


def model_dir(model: str) -> Path:
    """A local model directory or a Hugging Face repository (downloaded to the cache)."""
    if Path(model).exists() or str(model).startswith(("/", ".", "~")):
        return Path(model).expanduser()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model))


def load_thresholds(given: Path | None, model: Path) -> tuple[Thresholds, str | None]:
    """(thresholds, file path; defaults -> None). Looked up in the model directory, then one level up."""
    if given is not None:
        return Thresholds.load(given), str(given)
    for candidate in (Path(model) / "thresholds.json", Path(model).parent / "thresholds.json"):
        if candidate.exists():
            return Thresholds.load(candidate), str(candidate)
    candidate = Path(model) / "thresholds.json"
    print(f"Warning: {candidate} not found – using default thresholds {Thresholds.default()}", file=sys.stderr)
    return Thresholds.default(), None


def process_record(checker: "Checker", record: dict, audio_dir: Path | None) -> dict:
    """One input record -> one output line. Any problem with this record (missing / unreadable / too short audio,
    an unexpected exception while checking, e.g. an invalid `candidates` key) – only its `error`, the run goes on."""
    path = audio_file(str(record.get("audio_path", "")), audio_dir)
    if not path.is_file():
        return checker.error_row(record, f"audio file not found: {path}")
    try:
        audio = read_audio(path)
    except Exception as e:
        return checker.error_row(record, f"could not read {path}: {type(e).__name__}: {e}")
    if len(audio) < MIN_SAMPLES:
        # the aligner's convolution and the feature extractor fail on a few hundred samples
        return checker.error_row(record, f"audio too short: {len(audio)} samples (< {MIN_SAMPLES})")
    try:
        return checker.check_record(record, audio)
    except Exception as e:
        print(f"Warning: {record.get('id')}: {type(e).__name__}: {e}", file=sys.stderr)
        return checker.error_row(record, f"{type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ltstress check",
                                 description="Check text stress marks against recordings: manifest -> word verdicts")
    ap.add_argument("--input", type=Path, required=True, help="JSONL: id, audio_path, text[, candidates]")
    ap.add_argument("--model", default=HF_MODEL, help=f"local directory or HF repository (default {HF_MODEL})")
    ap.add_argument("--output", type=Path, required=True,
                    help="JSONL (resumable: completed ids are skipped, records with an error are retried)")
    ap.add_argument("--thresholds", type=Path, default=None, help="default <model>/thresholds.json")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="if audio_path does not exist – look for DIR/<file name>")
    ap.add_argument("--annotate", action="store_true", help="ignore text marks: best candidate and confidence")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    model = model_dir(a.model)
    thresholds, thresholds_path = load_thresholds(a.thresholds, model)
    records = [r for r in read_input(a.input)]
    done = already_done(a.output)
    todo = [r for r in records if str(r["id"]) not in done]
    print(f"Records: {len(records)}, already done: {len(records) - len(todo)}, remaining (incl. retried errors): "
          f"{len(todo)}", file=sys.stderr)
    if not todo:
        return

    checker = Checker(Aligner(device=a.device), FrameModel(model, device=a.device),
                      thresholds, annotate=a.annotate, model_name=str(a.model),
                      thresholds_path=thresholds_path)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with open(a.output, "a", encoding="utf-8") as f:
        for n, record in enumerate(todo, 1):
            if str(record["id"]) in done:
                continue                                   # duplicate id in the input
            row = process_record(checker, record, a.audio_dir)
            errors += row["error"] is not None
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            done.add(str(record["id"]))
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} (errors: {errors})", file=sys.stderr)


if __name__ == "__main__":
    main()
