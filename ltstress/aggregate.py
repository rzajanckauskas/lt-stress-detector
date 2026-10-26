"""
Summary of check output by word form: which text forms most often do not match the pronunciation.

Run: ltstress aggregate Y.jsonl [--output Y.forms.json]   (+ Y.forms.tsv and Y.forms.errors.tsv next to it)
A group is (word without marks in lower case, text form in lower case): a sentence-initial capital does not split a group.
Only the last output line of each id is used (resuming check repeats records with an error); records whose
last line has an `error` are listed separately (`errors`) – they have no words and are not part of any group.
Function-word stressed↔unstressed groups (kind "function_word") go to the end of the list and are counted separately.
Format "ltstress-forms-1" – FORMATS.md.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ltstress.phonetics import strip_marks

FORMAT = "ltstress-forms-1"
CONFIDENT = ("match", "mismatch")
TSV_FIELDS = ["word", "text_form", "n", "n_mismatch", "n_unclear", "most_common_best", "kind",
              "mark_error", "inconsistent", "examples"]


def _lines(output: Path):
    with open(output, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue                     # empty or truncated line


def _lower(s: str | None) -> str | None:
    return s.lower() if s is not None else None


def _most_common(values) -> object:
    c = Counter(v for v in values if v is not None)
    return c.most_common(1)[0][0] if c else None


def _mark_error(words: list[dict]) -> str | None:
    """The malformed text mark reason (mark_*), if the group's words have one."""
    return _most_common(w.get("reason") for w in words if (w.get("reason") or "").startswith("mark_"))


def _kind(words: list[dict], mismatches: list[dict], mark_error: str | None) -> str | None:
    """"malformed_mark" – malformed text mark; otherwise the most common kind among mismatches; if there are none –
    "function_word" if that is the most common kind among unclear words (otherwise null)."""
    if mark_error:
        return "malformed_mark"
    k = _most_common(w.get("kind") for w in mismatches)
    if k is None and _most_common(w.get("kind") for w in words if w["verdict"] == "unclear") == "function_word":
        return "function_word"
    return k


def aggregate(output: Path) -> dict:
    """{"format", "records", "errors": [{"id", "error"}], "function_words": {"groups", "words"}, "groups": [...]}.
    Groups with n, n_mismatch, n_unclear, most_common_best, kind (_kind), mark_error (reason or null),
    inconsistent (best forms of confident verdicts > 1 distinct), examples (up to 3 mismatches – if there are none,
    any – with the largest margin; null margin – last). Sorted: malformed_mark first, then stress_position, then
    n_mismatch / n; "function_word" groups – always last. `function_words`: the number of such groups and the number
    of words with kind "function_word" in the queue (mismatch + unclear)."""
    last: dict[object, dict] = {}
    for line in _lines(Path(output)):
        if isinstance(line, dict):
            last[line.get("id")] = line
    errors = [{"id": rid, "error": e["error"]} for rid, e in last.items() if e.get("error")]
    groups: dict[tuple[str, str], list[tuple[object, dict]]] = defaultdict(list)
    for line in last.values():
        for w in line.get("words") or []:
            key = (strip_marks(w["word"]).lower(), w["text_form"].lower())
            groups[key].append((line.get("id"), w))

    out = []
    for (word, text_form), items in groups.items():
        words = [w for _, w in items]
        mismatches = [(rid, w) for rid, w in items if w["verdict"] == "mismatch"]
        pool = mismatches or items
        best = sorted(pool, key=lambda e: (e[1]["margin"] is None, -(e[1]["margin"] or 0.0)))
        mark_error = _mark_error(words)
        out.append({
            "word": word, "text_form": text_form, "n": len(words), "n_mismatch": len(mismatches),
            "n_unclear": sum(w["verdict"] == "unclear" for w in words),
            "most_common_best": _most_common(_lower(w.get("best_form")) for w in words),
            "kind": _kind(words, [w for _, w in mismatches], mark_error),
            "mark_error": mark_error,
            "inconsistent": len({_lower(w.get("best_form")) for w in words
                                 if w["verdict"] in CONFIDENT}) > 1,
            "examples": [{"id": rid, "i": w["i"]} for rid, w in best[:3]],
        })
    out.sort(key=lambda g: (g["kind"] == "function_word", g["kind"] != "malformed_mark", g["kind"] != "stress_position",
                            -g["n_mismatch"] / g["n"], -g["n_mismatch"], -g["n"], g["word"], g["text_form"]))
    function_words = {"groups": sum(g["kind"] == "function_word" for g in out),
                      "words": sum(w.get("kind") == "function_word" and w["verdict"] != "match"
                                   for items in groups.values() for _, w in items)}
    return {"format": FORMAT, "records": len(last), "errors": errors, "function_words": function_words,
            "groups": out}


def _tsv_value(field: str, v) -> str:
    if field == "examples":
        return ",".join(f"{p['id']}:{p['i']}" for p in v)
    if v is None:
        return ""
    return str(v).replace("\t", " ")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ltstress aggregate", description="Summarize check output by word form")
    ap.add_argument("input", type=Path, help="ltstress check output (JSONL)")
    ap.add_argument("--output", type=Path, default=None,
                    help="default <input>.forms.json (+ .tsv and .errors.tsv next to it)")
    a = ap.parse_args(argv)
    output = a.output or a.input.with_suffix(".forms.json")
    r = aggregate(a.input)
    groups, errors = r["groups"], r["errors"]
    output.write_text(json.dumps(r, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    tsv = output.with_suffix(".tsv")
    lines = ["\t".join(TSV_FIELDS)] + ["\t".join(_tsv_value(k, g[k]) for k in TSV_FIELDS) for g in groups]
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    etsv = output.with_suffix(".errors.tsv")
    etsv.write_text("\n".join(["id\terror"] + [f"{_tsv_value('id', e['id'])}\t{_tsv_value('error', e['error'])}"
                                              for e in errors]) + "\n", encoding="utf-8")
    fw = r["function_words"]
    print(f"Records: {r['records']}, errors: {len(errors)}, groups: {len(groups)} (function-word groups: "
          f"{fw['groups']} – at the end of the list; function words in the queue: {fw['words']}) -> {output}, {tsv}, {etsv}")


if __name__ == "__main__":
    main()
