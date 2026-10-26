import json

from ltstress.aggregate import aggregate


def _w(form, verdict, kind, best, margin, i=0):
    return {"i": i, "word": form, "text_form": form, "best_form": best, "verdict": verdict,
            "kind": kind, "margin": margin, "scores": {}, "nucleus_type": "ae", "alignment_score": 0.9,
            "reason": None}


def test_aggregate_sorting_and_inconsistency(tmp_path):
    def w(form, verdict, kind, best, margin):
        return {"i": 0, "word": form, "text_form": form, "best_form": best, "verdict": verdict,
                "kind": kind, "margin": margin, "scores": {}, "nucleus_type": "ae", "alignment_score": 0.9, "reason": None}
    lines = [{"id": "1", "words": [w("dalykùs", "mismatch", "stress_position", "dalỹkus", 1.2)]},
             {"id": "2", "words": [w("dalykùs", "mismatch", "stress_position", "dalýkus", 0.9)]},
             {"id": "3", "words": [w("dalykùs", "match", None, "dalykùs", 0.8), w("laĩvas", "unclear", "pitch_accent", "láivas", 0.1)]}]
    path = tmp_path / "y.jsonl"; path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines))
    r = aggregate(path)["groups"]
    assert r[0]["text_form"] == "dalykùs" and r[0]["n"] == 3 and r[0]["n_mismatch"] == 2
    assert r[0]["inconsistent"] is True


def test_aggregate_group_fields_and_examples(tmp_path):
    lines = [{"id": "1", "words": [_w("Laĩvas", "mismatch", "pitch_accent", "láivas", 0.7)]},
             {"id": "2", "words": [_w("laĩvas", "mismatch", "pitch_accent", "láivas", 2.0, i=3)]},
             {"id": "3", "words": [_w("laĩvas", "unclear", "stress_position", "laivas", 0.1)]},
             {"id": "4", "error": "audio file not found", "words": []},
             {"id": "5", "words": [_w("laĩvas", "mismatch", "pitch_accent", "láivas", 1.0),
                                   _w("tàs", "mismatch", "stress_position", "tas", None, i=1)]},
             {"id": "6", "words": [_w("laĩvas", "mismatch", "pitch_accent", "láivas", 0.5)]}]
    path = tmp_path / "y.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines) + '\n{"id": "7", "wor')
    r = aggregate(path)["groups"]
    assert [g["text_form"] for g in r] == ["tàs", "laĩvas"]       # stress_position first
    g = r[1]
    assert g["word"] == "laivas" and g["n"] == 5 and g["n_mismatch"] == 4 and g["n_unclear"] == 1
    assert g["most_common_best"] == "láivas" and g["kind"] == "pitch_accent" and g["inconsistent"] is False
    assert g["examples"] == [{"id": "2", "i": 3}, {"id": "5", "i": 0}, {"id": "1", "i": 0}]
    assert r[0]["examples"] == [{"id": "5", "i": 1}]


def test_aggregate_cli_json_and_tsv(tmp_path):
    from ltstress.__main__ import main
    path = tmp_path / "y.jsonl"
    path.write_text(json.dumps({"id": "1", "words": [_w("laĩvas", "mismatch", "pitch_accent", "láivas", 0.7)]},
                               ensure_ascii=False) + "\n", encoding="utf-8")
    main(["aggregate", str(path), "--output", str(tmp_path / "y.forms.json")])
    r = json.loads((tmp_path / "y.forms.json").read_text(encoding="utf-8"))["groups"]
    assert r[0]["text_form"] == "laĩvas"
    tsv = (tmp_path / "y.forms.tsv").read_text(encoding="utf-8").splitlines()
    assert tsv[0].split("\t")[:2] == ["word", "text_form"] and "laĩvas" in tsv[1]


def test_aggregate_capital_letter_does_not_split_and_no_inconsistency(tmp_path):
    lines = [{"id": "1", "words": [_w("Tàs", "mismatch", "stress_position", "Tas", 1.0)]},
             {"id": "2", "words": [_w("tàs", "mismatch", "stress_position", "tas", 0.8)]}]
    path = tmp_path / "y.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines), encoding="utf-8")
    r = aggregate(path)["groups"]
    assert len(r) == 1 and r[0]["n"] == 2
    assert r[0]["inconsistent"] is False and r[0]["most_common_best"] == "tas"


def test_aggregate_errors_and_last_id_line(tmp_path):
    lines = [{"id": "1", "error": "audio file not found: a.wav", "words": []},
             {"id": "2", "error": "CUDA OOM", "words": []},
             {"id": "3", "error": None, "words": [_w("laĩvas", "mismatch", "pitch_accent", "láivas", 0.7)]},
             {"id": "3", "error": None, "words": [_w("laĩvas", "match", None, "laĩvas", 0.9)]},
             {"id": "2", "error": None, "words": [_w("laĩvas", "match", None, "laĩvas", 0.8)]}]   # retried – succeeded
    path = tmp_path / "y.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines), encoding="utf-8")
    r = aggregate(path)
    assert r["records"] == 3 and r["errors"] == [{"id": "1", "error": "audio file not found: a.wav"}]
    g = r["groups"][0]
    assert g["n"] == 2 and g["n_mismatch"] == 0               # "3" – only the last line


def test_aggregate_mark_groups_separate_and_first(tmp_path):
    bad = _w("aš̃", "unclear", None, "àš", 0.4)
    bad["reason"] = "mark_outside_nucleus"
    lines = [{"id": "1", "words": [_w("tàs", "mismatch", "stress_position", "tas", 1.0), bad]},
             {"id": "2", "words": [_w("laĩvas", "unclear", "stress_position", "láivas", 0.1)]}]
    path = tmp_path / "y.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines), encoding="utf-8")
    r = aggregate(path)["groups"]
    assert [(g["text_form"], g["kind"]) for g in r] == [("aš̃", "malformed_mark"), ("tàs", "stress_position"),
                                                        ("laĩvas", None)]
    assert r[0]["mark_error"] == "mark_outside_nucleus" and r[1]["mark_error"] is None


def test_aggregate_cli_errors_tsv_and_count(tmp_path, capsys):
    from ltstress.__main__ import main
    path = tmp_path / "y.jsonl"
    path.write_text(json.dumps({"id": "1", "error": "no\taudio", "words": []}, ensure_ascii=False) + "\n"
                    + json.dumps({"id": "2", "error": None, "words": [_w("laĩvas", "match", None, "laĩvas", 0.7)]},
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    main(["aggregate", str(path), "--output", str(tmp_path / "y.forms.json")])
    j = json.loads((tmp_path / "y.forms.json").read_text(encoding="utf-8"))
    assert j["format"] == "ltstress-forms-1" and j["errors"] == [{"id": "1", "error": "no\taudio"}]
    assert (tmp_path / "y.forms.errors.tsv").read_text(encoding="utf-8").splitlines() == ["id\terror", "1\tno audio"]
    assert "errors: 1" in capsys.readouterr().out


def test_aggregate_function_word_groups_last_and_counted(tmp_path, capsys):
    from ltstress.__main__ import main
    lines = [{"id": "1", "words": [_w("tai", "mismatch", "function_word", "taĩ", 2.0),
                                   _w("laĩvas", "mismatch", "pitch_accent", "láivas", 0.7, i=1),
                                   _w("tàs", "mismatch", "stress_position", "tas", 1.0, i=2),
                                   _w("dàr", "unclear", "stress_position", "dar", 0.1, i=3),
                                   _w("júk", "unclear", "function_word", "juk", 0.2, i=4)]},
             {"id": "2", "words": [_w("laĩvas", "match", None, "laĩvas", 1.0),
                                   _w("tái", "mismatch", "pitch_accent", "taĩ", 1.0, i=1)]}]
    path = tmp_path / "y.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines), encoding="utf-8")
    r = aggregate(path)
    # function word "tai" (n_mismatch / n = 1) – still after pitch_accent and after groups without a kind
    assert [(g["text_form"], g["kind"]) for g in r["groups"]] == [
        ("tàs", "stress_position"), ("tái", "pitch_accent"), ("laĩvas", "pitch_accent"), ("dàr", None),
        ("tai", "function_word"), ("júk", "function_word")]
    assert r["function_words"] == {"groups": 2, "words": 2}
    main(["aggregate", str(path), "--output", str(tmp_path / "y.forms.json")])
    assert "function-word groups: 2" in capsys.readouterr().out
    assert json.loads((tmp_path / "y.forms.json").read_text(encoding="utf-8"))["function_words"]["groups"] == 2
