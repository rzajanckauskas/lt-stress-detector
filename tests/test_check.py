import json
import math

import numpy as np
import pytest
import soundfile as sf

from ltstress.check import Checker, _num, words_in_text
from ltstress.phonetics import to_model_form as M
from ltstress.scoring import Thresholds


def test_record_one_mismatch(fake_checker):
    r = fake_checker.check_record({"id": "x", "audio_path": "x.wav", "text": "Tie dalykùs"}, np.zeros(16000, np.float32))
    w = {w["word"]: w for w in r["words"]}
    assert w["dalykùs"]["verdict"] == "mismatch" and w["dalykùs"]["kind"] == "stress_position"
    assert "̃" in w["dalykùs"]["best_form"]


def test_word_without_letters_and_multiple_marks(fake_checker):
    r = fake_checker.check_record({"id": "y", "audio_path": "y.wav", "text": "Ir 5 kõkį-nórs"}, np.zeros(16000, np.float32))
    reasons = {w["word"]: w["reason"] for w in r["words"]}
    assert reasons["5"] == "no_letters" and reasons["kõkį-nórs"] == "multiple_marks"


def test_alignment_error_whole_record_unclear(failing_align_checker):
    r = failing_align_checker.check_record({"id": "z", "audio_path": "z.wav", "text": "laĩvas"}, np.zeros(160, np.float32))
    assert all(w["verdict"] == "unclear" and w["reason"] == "alignment_error" for w in r["words"])


def test_cli_resumes_and_rewrites_path(tmp_path, cli_with_fakes):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id":"a","audio_path":"/workspace/ft/x/wav/a.wav","text":"laĩvas"}\n{"id":"b","audio_path":"/missing/b.wav","text":"laĩvas"}\n')
    (tmp_path / "wav").mkdir(); sf.write(tmp_path / "wav" / "a.wav", np.zeros(16000), 16000)
    out = tmp_path / "out.jsonl"
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out), "--audio-dir", str(tmp_path / "wav")])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [e["id"] for e in rows] == ["a", "b"] and rows[0]["error"] is None and rows[1]["error"]
    # on a re-run: "a" is skipped, "b" (error) is retried; once the file exists – a new line without an error
    sf.write(tmp_path / "wav" / "b.wav", np.zeros(16000), 16000)
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out), "--audio-dir", str(tmp_path / "wav")])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [e["id"] for e in rows] == ["a", "b", "b"] and rows[2]["error"] is None and rows[2]["words"]
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out), "--audio-dir", str(tmp_path / "wav")])
    assert len(out.read_text().splitlines()) == 3


# --- extra --------------------------------------------------------------------------------------------
def test_words_in_text_hyphen_and_precomposed():
    assert words_in_text("Ir 5 kõkį-nórs, – tàs…") == ["Ir", "5", "kõkį-nórs", "tàs"]
    assert words_in_text("tàs tàs i̇̀r") == ["tàs", "tàs", "i̇̀r"]


def test_precomposed_word_text_form_in_model_form(fake_checker):
    r = fake_checker.check_record({"id": "x", "text": "Tie dalykùs"}, np.zeros(16000, np.float32))
    w = r["words"][1]
    assert (w["word"], w["text_form"], w["verdict"]) == ("dalykùs", "dalykùs", "mismatch")


def test_output_row_fields_and_strict_json(fake_checker):
    r = fake_checker.check_record({"id": "x", "audio_path": "x.wav", "text": "Tie dalykùs"}, np.zeros(16000, np.float32))
    assert r["format"] == "ltstress-check-1" and r["error"] is None and r["id"] == "x"
    assert r["thresholds"] == {"confident": 0.5, "position": 0.5, "pitch_accent": {}, "min_alignment": 0.2}
    assert r["thresholds_path"] is None
    fields = {"i", "word", "s0", "s1", "t0", "t1", "text_form", "text_mark", "best_form", "scores",
              "margin", "verdict", "kind", "nucleus_type", "nuclei", "alignment_score", "reason"}
    assert all(set(w) == fields for w in r["words"])
    w = r["words"][1]
    assert (w["i"], w["text_form"], w["t0"], w["t1"]) == (1, M("dalykùs"), 0.6, 1.2)
    assert (w["s0"], w["s1"], w["text_mark"]) == (4, 11, [2, 0])
    assert w["nuclei"] == [[0.6, 0.8], [0.8, 1.0], [1.0, 1.2]]
    assert r["words"][0]["text_mark"] is None
    assert w["nucleus_type"] == "short"          # text mark on "u"
    assert M("dalykus") in w["scores"] and w["best_form"] in w["scores"]
    json.dumps(r, allow_nan=False)


def test_checker_passes_unmarked_words_to_aligner(fake_checker):
    fake_checker.check_record({"id": "y", "text": "Ir 5 kõkį-nórs"}, np.zeros(16000, np.float32))
    assert fake_checker.aligner.calls[-1] == ["ir", "5", "kokį-nors"]


def test_input_candidates_added(fake_checker):
    r = fake_checker.check_record(
        {"id": "x", "text": "Tie dalykùs", "candidates": {"1": ["dalykũs", "dálỹkus"]}}, np.zeros(16000, np.float32))
    assert M("dalykũs") in r["words"][1]["scores"]       # variants give no circumflex on a short u
    assert M("dalykũs") not in r["words"][0]["scores"]


def test_word_without_nuclei_unclear(fake_checker):
    r = fake_checker.check_record({"id": "x", "text": "pvz laĩvas"}, np.zeros(16000, np.float32))
    w = r["words"][0]
    assert (w["verdict"], w["reason"], w["scores"]) == ("unclear", "no_nucleus", {})
    json.dumps(r, allow_nan=False)


def test_annotate_without_text_marks(fake_checker):
    c = Checker(fake_checker.aligner, fake_checker.frame_model, Thresholds.default(), annotate=True)
    r = c.check_record({"id": "x", "text": "Tie dalykus"}, np.zeros(16000, np.float32))
    w = r["words"][1]
    assert (w["verdict"], w["best_form"]) == ("match", M("dalỹkus"))
    assert w["nucleus_type"] == "long"              # syllable of the best candidate ("y")


def test_infinite_margin_null():
    assert _num(math.inf) is None and _num(-math.inf) is None and _num(0.123456) == 0.1235


def test_cli_precomposed_and_resample_and_truncated_line(tmp_path, cli_with_fakes):
    (tmp_path / "a.wav").write_bytes(b"not audio")
    sf.write(tmp_path / "c.wav", np.zeros((48000, 2)), 48000)       # stereo 48 kHz -> mono 16 kHz
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"id": "a", "audio_path": str(tmp_path / "a.wav"), "text": "laĩvas"}) + "\n"
                   + json.dumps({"id": "c", "audio_path": str(tmp_path / "c.wav"), "text": "Tie dalykùs"}) + "\n")
    out = tmp_path / "out.jsonl"
    out.write_text('{"format": "ltstress-check-1", "id": "c", "wor')   # truncated line – redone
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [e["id"] for e in rows] == ["a", "c"]
    assert rows[0]["error"] and rows[0]["words"] == []
    assert rows[1]["error"] is None and rows[1]["model"].endswith("best")
    assert rows[1]["words"][1]["verdict"] == "mismatch"


def test_cli_thresholds_next_to_model(tmp_path, cli_with_fakes):
    model = tmp_path / "run" / "best"; model.mkdir(parents=True)
    Thresholds(confident=0.5, position=99.0, pitch_accent={}, min_alignment=0.2).save(tmp_path / "run" / "thresholds.json")
    sf.write(tmp_path / "c.wav", np.zeros(16000), 16000)
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"id": "c", "audio_path": str(tmp_path / "c.wav"), "text": "Tie dalykùs"}) + "\n")
    out = tmp_path / "out.jsonl"
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out), "--model", str(model)])
    w = json.loads(out.read_text())["words"][1]
    assert (w["verdict"], w["reason"]) == ("unclear", "low_margin")     # position threshold 99


def test_main_without_command_and_unknown(capsys):
    from ltstress.__main__ import main
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit) as e:
        main(["unknown"])
    assert e.value.code != 0
    assert "check" in capsys.readouterr().err


def test_word_without_vocabulary_letters_unclear():
    from conftest import FakeAligner, FakeModel

    class NoO(FakeAligner):        # like the real aligner: a word with a letter missing from the vocabulary -> None
        def align(self, audio, words):
            out = super().align(audio, words)
            return [None if "ō" in w else t for w, t in zip(words, out)]
    c = Checker(NoO(), FakeModel(), Thresholds.default())
    r = c.check_record({"id": "x", "text": "Tōkijas dalykùs"}, np.zeros(16000, np.float32))
    assert [(w["verdict"], w["reason"]) for w in r["words"]][0] == ("unclear", "no_letters")
    assert r["words"][0]["t0"] is None and r["words"][1]["verdict"] in ("match", "mismatch", "unclear")


def test_cli_empty_and_too_short_audio_error_continues(tmp_path, cli_with_fakes):
    sf.write(tmp_path / "empty.wav", np.zeros(0), 16000)             # header only
    sf.write(tmp_path / "short.wav", np.zeros(100), 16000)
    sf.write(tmp_path / "good.wav", np.zeros(16000), 16000)
    inp = tmp_path / "in.jsonl"
    inp.write_text("".join(json.dumps({"id": i, "audio_path": str(tmp_path / f"{i}.wav"), "text": "laĩvas"}) + "\n"
                           for i in ("empty", "short", "good")))
    out = tmp_path / "out.jsonl"
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [e["id"] for e in rows] == ["empty", "short", "good"]
    assert all("audio too short" in e["error"] and e["words"] == [] for e in rows[:2])
    assert rows[2]["error"] is None and rows[2]["words"]


def test_cli_unexpected_exception_only_record_error(tmp_path, cli_with_fakes, monkeypatch):
    import ltstress.check as check
    original = check.Checker.check_record

    def sometimes_fails(self, record, audio):
        if record["id"] == "a":
            raise RuntimeError("Kernel size can't be greater than actual input size")
        return original(self, record, audio)
    monkeypatch.setattr(check.Checker, "check_record", sometimes_fails)
    sf.write(tmp_path / "x.wav", np.zeros(16000), 16000)
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        json.dumps({"id": "a", "audio_path": str(tmp_path / "x.wav"), "text": "laĩvas"}) + "\n"
        + json.dumps({"id": "k", "audio_path": str(tmp_path / "x.wav"), "text": "laĩvas", "candidates": {"x": ["laivas"]}}) + "\n"
        + json.dumps({"id": "b", "audio_path": str(tmp_path / "x.wav"), "text": "laĩvas"}) + "\n")
    out = tmp_path / "out.jsonl"
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [e["id"] for e in rows] == ["a", "k", "b"]
    assert rows[0]["error"].startswith("RuntimeError:") and rows[0]["words"] == []
    assert rows[1]["error"].startswith("ValueError:")                  # invalid candidates key
    assert rows[2]["error"] is None


def test_text_mark_not_in_variants_added_to_candidates(fake_checker):
    # "súkti": acute on a single u – variants("sukti") do not produce it; it is still scored, not no_candidate
    r = fake_checker.check_record({"id": "x", "text": "Tie súkti"}, np.zeros(16000, np.float32))
    w = r["words"][1]
    assert M("súkti") in w["scores"]
    assert w["reason"] != "no_candidate" and w["verdict"] in ("match", "mismatch", "unclear")


# --- F1: malformed text marks ---------------------------------------------------------------------
@pytest.mark.parametrize("word,reason", [("pirm̀oji", "mark_outside_nucleus"), ("aš̃", "mark_outside_nucleus"),
                                         ("dar̀", "mark_not_canonical"),
                                         ("jĩe", "mark_not_canonical")])
def test_malformed_mark_unclear_but_scored(fake_checker, word, reason):
    r = fake_checker.check_record({"id": "x", "text": f"Tie {word}"}, np.zeros(16000, np.float32))
    w = r["words"][1]
    assert (w["verdict"], w["reason"], w["kind"], w["text_mark"]) == ("unclear", reason, None, None)
    assert w["scores"] and w["best_form"] in w["scores"] and w["margin"] is not None
    json.dumps(r, allow_nan=False)


def test_word_with_u0307_normalized(fake_checker):
    # "i̇" (i + U+0307) must not shift letter indices: the mark stays on "e", not on a consonant
    r = fake_checker.check_record({"id": "x", "text": "Tie di̇dẽlis"}, np.zeros(16000, np.float32))
    w = r["words"][1]
    assert w["text_form"] == M("didẽlis") and w["text_mark"] == [1, 2]
    assert not (w["reason"] or "").startswith("mark_")


# --- F3: output fields --------------------------------------------------------------------------------
def test_word_spans_in_text(fake_checker):
    text = "„Ir 5 kõkį-nórs, – tàs…“"
    r = fake_checker.check_record({"id": "x", "text": text}, np.zeros(16000, np.float32))
    assert [text[w["s0"]:w["s1"]] for w in r["words"]] == [w["word"] for w in r["words"]]
    assert [w["word"] for w in r["words"]] == ["Ir", "5", "kõkį-nórs", "tàs"]
    assert r["words"][1]["nuclei"] is None          # word without timings


def test_foreign_candidates_ignored(fake_checker):
    cand = fake_checker._candidates("dalykus", ["nesiaukštinà", "dalykũs"])
    assert (2, 2) in cand and not any(m is not None and m[0] >= 3 for m in cand)


def test_cli_thresholds_path_in_row(tmp_path, cli_with_fakes):
    Thresholds(confident=0.5, position=0.7, pitch_accent={"ae": 0.1}, min_alignment=0.2).save(tmp_path / "th.json")
    sf.write(tmp_path / "c.wav", np.zeros(16000), 16000)
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"id": "c", "audio_path": str(tmp_path / "c.wav"), "text": "laĩvas"}) + "\n")
    out = tmp_path / "out.jsonl"
    cli_with_fakes(["check", "--input", str(inp), "--output", str(out), "--thresholds", str(tmp_path / "th.json")])
    e = json.loads(out.read_text())
    assert e["thresholds"] == {"confident": 0.5, "position": 0.7, "pitch_accent": {"ae": 0.1}, "min_alignment": 0.2}
    assert e["thresholds_path"] == str(tmp_path / "th.json")


# --- F2: input without id -----------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ['{"audio_path": "x.wav", "text": "laĩvas"}', '{"id": null, "text": "a"}',
                                 '{"id": "", "text": "a"}', '["not", "object"]'])
def test_cli_line_without_id_error_with_number(tmp_path, cli_with_fakes, bad):
    sf.write(tmp_path / "c.wav", np.zeros(16000), 16000)
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"id": "c", "audio_path": str(tmp_path / "c.wav"), "text": "laĩvas"}) + "\n" + bad + "\n")
    with pytest.raises(SystemExit) as e:
        cli_with_fakes(["check", "--input", str(inp), "--output", str(tmp_path / "out.jsonl")])
    assert f"{inp}:2:" in str(e.value) and "id" in str(e.value)
    assert not (tmp_path / "out.jsonl").exists()


# --- function words -----------------------------------------------------------------------------------
class _SingleClassModel:
    """One class dominates in all frames (0 unstressed, 1 short, 2 acute, 3 circumflex)."""
    frame_s = 0.02

    def __init__(self, cls):
        self.cls = cls

    def logp(self, audio):
        p = np.full((200, 4), 0.01); p[:, self.cls] = 0.97
        return np.log(p).astype(np.float32)


def _check(cls, text):
    from conftest import FakeAligner
    c = Checker(FakeAligner(), _SingleClassModel(cls), Thresholds.default())
    r = c.check_record({"id": "t", "text": text}, np.zeros(16000, np.float32))
    return {w["word"]: (w["verdict"], w["kind"]) for w in r["words"]}


def test_function_word_stressed_unstressed_kind_function_word():
    # text stressed, spoken unstressed: the verdict stays, the kind is function_word (stress_position otherwise)
    assert _check(0, "Taĩ laĩvas") == {"Taĩ": ("mismatch", "function_word"), "laĩvas": ("mismatch", "stress_position")}
    # text unstressed, spoken stressed
    assert _check(1, "juk kur") == {"juk": ("mismatch", "function_word"), "kur": ("mismatch", "stress_position")}


def test_function_word_other_mismatches_usual_kind():
    # both stressed, same syllable, different pitch accent -> pitch_accent; match -> None
    assert _check(3, "tái")["tái"] == ("mismatch", "pitch_accent")
    assert _check(3, "taĩ")["taĩ"] == ("match", None)
    assert _check(0, "tai")["tai"] == ("match", None)
