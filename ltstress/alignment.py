"""Forced alignment: word and syllable nucleus times from a CTC model (wav2vec2, letter vocabulary).

Text -> lower case, vocabulary letters only, "|" after every word (and at the end of the sentence); log_softmax ->
torchaudio forced_align -> merge_tokens. Frame duration = len(audio) / 16000 / number of frames.
  - word: from the start of its first letter's peak to the start of the following "|";
  - nucleus: from the start of its first letter to the start of the next letter (or the end of the word), with two
    corrections calibrated on LIEPA (195 recordings):
      * the nucleus start is moved earlier by ONSET_SHIFT (the vowel peak lags ~14 ms behind the vowel onset),
        but not earlier than the end of the previous letter's peak;
      * the end is pulled back if the energy at the end of the nucleus drops > TRIM_DB below its maximum
        (a plosive's peak marks the burst, before it there is a silent closure).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as taf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from ltstress.phonetics import nucleus_groups, strip_marks

MODEL = "sammy786/wav2vec2-xlsr-lithuanian"
REVISIONS = {MODEL: "184be031ec84ea1c93ef0d2394f879c22533f411"}   # pinned: labels and check timings do not drift
SAMPLE_RATE = 16_000
EN_WINDOW, EN_HOP = 320, 80        # energy curve for the nucleus end: 20 ms window, 5 ms hop
TRIM_DB = 20.0
ONSET_SHIFT = 0.015    # s; calibrated on 195 LIEPA recordings


def energy_db(audio: np.ndarray) -> np.ndarray:
    """Frame energy in dB (20 ms window, 5 ms hop); the centre of frame j is j * 5 ms + 10 ms."""
    x = audio if len(audio) >= EN_WINDOW else np.pad(audio, (0, EN_WINDOW - len(audio)))
    n = 1 + (len(x) - EN_WINDOW) // EN_HOP
    idx = np.arange(EN_WINDOW)[None, :] + EN_HOP * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx].astype(np.float64) ** 2, axis=1) + 1e-12)


def trim_end(db: np.ndarray, s: float, e: float, limit_db: float, min_duration: float) -> float:
    """The nucleus end is pulled back to the last frame whose energy is ≥ (nucleus maximum − limit_db):
    for a plosive, the next letter's CTC peak marks the burst, so otherwise the silent closure would end up in the
    nucleus."""
    c0 = EN_WINDOW / 2 / SAMPLE_RATE
    j = np.arange(len(db))
    centres = c0 + j * EN_HOP / SAMPLE_RATE
    inside = (centres >= s) & (centres < e)
    if not inside.any():
        return e
    seg = db[inside]
    last = centres[inside][np.nonzero(seg >= seg.max() - limit_db)[0][-1]]
    return float(min(e, max(s + min_duration, last + EN_HOP / SAMPLE_RATE)))


@dataclass
class WordTiming:
    word: str
    t0: float
    t1: float
    nuclei: list[tuple[float, float]]     # (start, end) in s for each nucleus_groups(word) syllable
    score: float                          # mean exp(score) of the word's letters


class AlignmentError(RuntimeError):
    pass


class Aligner:
    def __init__(self, model: str = MODEL, device: str = "cuda", trim_db: float | None = TRIM_DB,
                 onset_shift: float = ONSET_SHIFT):
        self.trim_db = trim_db
        self.onset_shift = onset_shift
        self.name = model
        self.device = device
        self.revision = REVISIONS.get(model)
        self.processor = Wav2Vec2Processor.from_pretrained(model, revision=self.revision)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = Wav2Vec2ForCTC.from_pretrained(model, revision=self.revision, dtype=dtype,
                                                    use_safetensors=False).to(device).eval()
        self._cpu = None if device.startswith("cuda") else self.model
        tok = self.processor.tokenizer
        vocab = tok.get_vocab()
        self.letters = {c: i for c, i in vocab.items() if len(c) == 1 and c.isalpha()}
        self.delimiter = vocab[tok.word_delimiter_token]
        self.pad_id = tok.pad_token_id

    # --- emissions --------------------------------------------------------
    def _input(self, audio: np.ndarray) -> torch.Tensor:
        return self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values

    @torch.inference_mode()
    def _emission_gpu(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x.to(self.device, dtype=self.model.dtype)).logits
        return torch.log_softmax(logits.float(), dim=-1).cpu()

    @torch.inference_mode()
    def _emission_cpu(self, x: torch.Tensor) -> torch.Tensor:
        if self._cpu is None:
            self._cpu = Wav2Vec2ForCTC.from_pretrained(self.name, revision=self.revision, dtype=torch.float32,
                                                       use_safetensors=False).eval()
        return torch.log_softmax(self._cpu(x.float()).logits.float(), dim=-1)

    def emission(self, audio: np.ndarray) -> torch.Tensor:
        """log_softmax (1, frames, vocabulary) as a CPU tensor; GPU OOM -> recomputed on CPU."""
        x = self._input(audio)
        if self._cpu is self.model:
            return self._emission_cpu(x)
        try:
            return self._emission_gpu(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._emission_cpu(x)

    # --- alignment --------------------------------------------------------
    def _letter_positions(self, word: str) -> list[int] | None:
        """Indices of the word's characters that are aligned; None – the word has no vocabulary letters or has a
        letter / digit missing from the vocabulary (then part of the spoken word would get no labels and would push
        its neighbours)."""
        pos = [i for i, c in enumerate(word) if c in self.letters]
        if not pos or any(c.isalnum() and c not in self.letters for c in word):
            return None
        return pos

    def align(self, audio: np.ndarray, words: list[str]) -> list[WordTiming | None]:
        words = [strip_marks(w).lower() for w in words]
        positions = [self._letter_positions(w) for w in words]
        targets: list[int] = []
        for w, pos in zip(words, positions):
            if pos is not None:
                targets += [self.letters[w[i]] for i in pos] + [self.delimiter]   # "|" after every word, and at the end
        if not targets:
            return [None] * len(words)

        emission = self.emission(audio)
        frames = emission.shape[1]
        repeats = sum(a == b for a, b in zip(targets, targets[1:]))
        if len(targets) + repeats > frames:
            raise AlignmentError(f"{len(targets)} tokens (+{repeats} repeats) > {frames} frames")
        try:
            path, scores = taf.forced_align(emission, torch.tensor([targets], dtype=torch.int32), blank=self.pad_id)
        except RuntimeError as e:
            raise AlignmentError(str(e)) from e
        spans = taf.merge_tokens(path[0], scores[0].exp(), blank=self.pad_id)
        if len(spans) != len(targets):
            raise AlignmentError(f"merged {len(spans)} tokens, expected {len(targets)}")

        dt = len(audio) / SAMPLE_RATE / frames
        db = energy_db(audio) if self.trim_db is not None else None
        out: list[WordTiming | None] = []
        k = 0
        for w, pos in zip(words, positions):
            if pos is None:
                out.append(None)
                continue
            bound = spans[k - 1].end * dt if k else 0.0    # end of the previous word's "|"
            wt = dict(zip(pos, spans[k:k + len(pos)]))     # character index -> TokenSpan
            k += len(pos) + 1
            # Word end = start of the following "|": a CTC letter is a one-frame peak, while a word-final vowel
            # continues through blank frames up to "|" (the end of the last letter would cut the final nucleus to 20 ms).
            end = spans[k - 1].start * dt
            # Letter start = start of its peak; for the first letter of a nucleus – moved earlier by ONSET_SHIFT
            # (the vowel peak lags behind the vowel onset), but not earlier than the end of the previous letter's
            # peak (at the start of a word – the end of the previous "|").
            starts = {i: wt[i].start * dt for i in pos}
            groups = nucleus_groups(w)
            for g in groups:
                before = [i for i in pos if i < g[0]]
                floor = wt[before[-1]].end * dt if before else bound
                starts[g[0]] = max(floor, min(starts[g[0]], starts[g[0]] - self.onset_shift))
            nuclei = []
            for g in groups:
                after = [i for i in pos if i > g[-1]]
                b0, b1 = starts[g[0]], (starts[after[0]] if after else end)
                if db is not None:
                    b1 = trim_end(db, b0, b1, self.trim_db, dt)
                nuclei.append((b0, b1))
            score = float(np.mean([wt[i].score for i in pos]))
            out.append(WordTiming(w, starts[pos[0]], end, nuclei, score))
        return out
