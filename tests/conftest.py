import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

DATA = Path(__file__).parent / "data"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CUDA and the CTC model (skipped without CUDA)")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def aligner():
    from ltstress.alignment import Aligner
    return Aligner()


@pytest.fixture(scope="session")
def fixture():
    meta = json.loads((DATA / "liepa_fixture.json").read_text(encoding="utf-8"))
    audio, rate = sf.read(DATA / "liepa_fixture.wav", dtype="float32")
    assert rate == 16_000
    return {**meta, "audio": np.ascontiguousarray(audio, dtype=np.float32)}


# --- fake aligner and frame model for checking ----------------------------------------------
def fake_logp(T=100):
    """As in test_scoring._logp: class 0 everywhere, frames 40..50 – class 3 (the second nucleus of the word
    "dalykus" when it is the second word, after 0.6 s)."""
    p = np.full((T, 4), 0.01); p[:, 0] = 0.97
    p[40:50] = 0.01; p[40:50, 3] = 0.97
    return np.log(p).astype(np.float32)


class FakeAligner:
    """0.6 s per word, one after another, nuclei spread evenly; a word without letters -> None."""

    def __init__(self, *a, **kw):
        self.calls = []

    def align(self, audio, words):
        from ltstress.alignment import WordTiming
        from ltstress.phonetics import nucleus_groups
        self.calls.append(list(words))
        out = []
        for k, w in enumerate(words):
            if not any(c.isalpha() for c in w):
                out.append(None)
                continue
            t0, n = 0.6 * k, len(nucleus_groups(w))
            out.append(WordTiming(w, t0, t0 + 0.6, [(t0 + 0.6 * j / n, t0 + 0.6 * (j + 1) / n) for j in range(n)], 0.9))
        return out


class FailingAligner:
    def __init__(self, *a, **kw):
        pass

    def align(self, audio, words):
        from ltstress.alignment import AlignmentError
        raise AlignmentError("more letters than frames")


class FakeModel:
    frame_s = 0.02

    def __init__(self, *a, **kw):
        pass

    def logp(self, audio):
        return fake_logp()


@pytest.fixture
def fake_checker():
    from ltstress.check import Checker
    from ltstress.scoring import Thresholds
    return Checker(FakeAligner(), FakeModel(), Thresholds.default())


@pytest.fixture
def failing_align_checker():
    from ltstress.check import Checker
    from ltstress.scoring import Thresholds
    return Checker(FailingAligner(), FakeModel(), Thresholds.default())


@pytest.fixture
def cli_with_fakes(monkeypatch, tmp_path):
    """Calls ltstress.__main__.main(argv) with a fake aligner and model (without --model – a fake path)."""
    import ltstress.check as check
    from ltstress.__main__ import main
    monkeypatch.setattr(check, "Aligner", FakeAligner)
    monkeypatch.setattr(check, "FrameModel", FakeModel)

    def run(argv):
        argv = list(argv)
        if argv[0] == "check" and "--model" not in argv:
            argv += ["--model", str(tmp_path / "fake_model" / "best")]
        return main(argv)
    return run
