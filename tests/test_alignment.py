import numpy as np
import pytest
import torch

from ltstress.alignment import AlignmentError, energy_db, trim_end
from ltstress.phonetics import nucleus_groups, strip_marks

gpu = pytest.mark.gpu


def _words(fixture):
    return [strip_marks(w).lower() for w in fixture["text"].split()]


@gpu
def test_alignment_fixture(aligner, fixture):
    words = _words(fixture)
    r = aligner.align(fixture["audio"], words)
    assert len(r) == len(words) and all(r)
    assert all(a.t1 <= b.t0 + 1e-6 for a, b in zip(r, r[1:]))
    assert all(len(x.nuclei) == len(nucleus_groups(x.word)) for x in r)
    assert all(0 <= x.t0 < x.t1 <= len(fixture["audio"]) / 16000 for x in r)


@gpu
def test_nuclei_within_word_and_score(aligner, fixture):
    r = aligner.align(fixture["audio"], _words(fixture))
    for x in r:
        assert 0 < x.score <= 1
        assert all(x.t0 - 1e-6 <= s < e <= x.t1 + 1e-6 for s, e in x.nuclei)
        assert all(a[1] <= b[0] + 1e-6 for a, b in zip(x.nuclei, x.nuclei[1:]))


@gpu
def test_word_without_letters_none(aligner, fixture):
    r = aligner.align(fixture["audio"], ["–"] + [strip_marks(w).lower() for w in fixture["text"].split()])
    assert r[0] is None and all(r[1:])


@gpu
def test_word_with_digit_none(aligner, fixture):
    words = _words(fixture)
    r = aligner.align(fixture["audio"], words[:2] + ["3-ioji"] + words[2:])
    assert r[2] is None and all(r[:2]) and all(r[3:])


@gpu
def test_too_many_letters_error(aligner):
    with pytest.raises(AlignmentError):
        aligner.align(np.zeros(1600, dtype=np.float32), ["žodis"] * 200)


@gpu
def test_gpu_oom_recomputed_on_cpu(aligner, fixture, monkeypatch):
    words = _words(fixture)
    gpu = aligner.align(fixture["audio"], words)

    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("artificial OOM")

    monkeypatch.setattr(aligner, "_emission_gpu", oom)
    cpu = aligner.align(fixture["audio"], words)
    frame = 0.02 + 1e-3
    assert [x.word for x in cpu] == [x.word for x in gpu]
    assert all(abs(a.t0 - b.t0) <= frame and abs(a.t1 - b.t1) <= frame for a, b in zip(cpu, gpu))


@gpu
def test_final_nucleus_not_cut(aligner, fixture):
    """A word-final vowel continues up to "|" – not just a one-frame CTC peak (vokietijA, galI, karĄ, pralaimėtI)."""
    r = aligner.align(fixture["audio"], _words(fixture))
    finals = [x.nuclei[-1] for x in r if x.word[-1] in "aeiouyąęėįųū"]
    assert len(finals) == 4
    assert all(e - s >= 0.035 for s, e in finals)


def test_trim_end_cuts_silent_closure():
    """Vowel at 0.10–0.20 s, then silence (closure) up to the next letter's peak at 0.26 s -> end ~0.20 s."""
    x = np.zeros(8000, dtype=np.float32)
    x[1600:3200] = 0.3 * np.sin(np.arange(1600) * 2 * np.pi * 200 / 16000)
    x[3200:] = 1e-4 * np.random.default_rng(0).standard_normal(4800)
    db = energy_db(x)
    e = trim_end(db, 0.10, 0.26, 20.0, 0.02)
    assert 0.195 <= e <= 0.215
    assert trim_end(db, 0.10, 0.18, 20.0, 0.02) == 0.18          # loud to the end – unchanged
    assert trim_end(db, 0.21, 0.26, 20.0, 0.02) >= 0.23          # at least min_duration


def test_aligner_model_revision_pinned(monkeypatch):
    import ltstress.alignment as al
    calls = []

    class Tok:
        word_delimiter_token, pad_token_id = "|", 0

        def get_vocab(self):
            return {"<pad>": 0, "|": 1, "a": 2}

    class Proc:
        tokenizer = Tok()

    class Mod:
        def to(self, *a):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(al.Wav2Vec2Processor, "from_pretrained",
                        lambda name, **kw: calls.append((name, kw.get("revision"))) or Proc())
    monkeypatch.setattr(al.Wav2Vec2ForCTC, "from_pretrained",
                        lambda name, **kw: calls.append((name, kw.get("revision"))) or Mod())
    al.Aligner(device="cpu")
    assert calls == [(al.MODEL, "184be031ec84ea1c93ef0d2394f879c22533f411")] * 2
