# File formats

The formats are frozen: a change in semantics means a new version (`-1` → `-2`); adding fields without changing the
meaning of existing ones keeps the same version. Stress marks are always in model form (NFC letters + separate
U+0300 short / U+0301 acute / U+0303 circumflex; never precomposed characters or U+0307). A mark `[syllable, accent]`
is the syllable index per `nucleus_groups` and the accent 0 short / 1 acute / 2 circumflex, after `canon` (on a mixed
diphthong with i/u an acute equals a grave).

## `ltstress check` input (JSONL)

`{"id", "audio_path", "text", "candidates"?: {"<word index>": ["form", ...]}}`. `id` is required and must be non-empty
(a line without it is an error reporting the line number; nothing is processed). `candidates` – additional
stress-marked forms of the same word (forms of a different word are ignored).

## `ltstress-check-1` (`ltstress check` output, JSONL)

One line per record, written line by line. When resuming into the same file, records whose last line has no `error`
are skipped, while records with an `error` are retried and get a new line: **readers take the last line of each
`id`** (as `aggregate` does).

Record fields:

| field | value |
|---|---|
| `format` | `"ltstress-check-1"` |
| `id`, `audio_path` | from the input |
| `model` | frame model path |
| `thresholds` | the thresholds used: `{"confident", "position", "pitch_accent": {nucleus type: threshold}, "min_alignment"}` |
| `thresholds_path` | thresholds file (`--thresholds`, or `<model>/thresholds.json`, or `<model>/../thresholds.json`); `null` – defaults |
| `error` | `null` or the record error (missing / unreadable / too short audio, unexpected exception); then `words` = `[]` |
| `words` | list of words (below) |

Word fields:

| field | value |
|---|---|
| `i` | word index in the sentence (matches the input `candidates` key) |
| `word` | the word as in the text; `text[s0:s1] == word` |
| `s0`, `s1` | character offsets of the word in the input `text` string (Python `str` indices, i.e. Unicode code points) |
| `t0`, `t1` | word times in the recording, s (`null` if not aligned) |
| `text_form` | `word` in model form |
| `text_mark` | the text mark `[syllable, accent]`; `null` – word without a mark, several marks, or a malformed mark (`reason` `mark_*`) |
| `best_form` | the acoustically best candidate (model form); `null` if there are no scores |
| `scores` | `{candidate form: mean log-probability}` (candidates: unstressed, `stress_variants`, the text mark, input `candidates`) |
| `margin` | margin (log-probability difference): `match` – over the second best, `mismatch`/`unclear` – best over the text mark, `mark_*` – best over the second best; `null` – a single candidate or no scores |
| `verdict` | `"match"` \| `"mismatch"` \| `"unclear"` (`mismatch` and `unclear` both go to the listening queue) |
| `kind` | `"stress_position"` \| `"pitch_accent"` \| `"function_word"` \| `null` (below) |
| `nucleus_type` | nucleus type of the stressed syllable of the text (without a mark – of the best candidate): `short`, `ae`, `long`, `mixed`, `diphthong`, `ie_uo`, `other` |
| `nuclei` | syllable nucleus times `[[start, end], ...]`, s, one per `nucleus_groups` syllable; `null` if not aligned |
| `alignment_score` | the word's CTC alignment score (0–1) |
| `reason` | why `unclear` (below); `null` otherwise |

`reason`:

| value | when | scores |
|---|---|---|
| `no_letters` | the word has no letters from the aligner vocabulary (a number, "ō") | no |
| `multiple_marks` | several stress marks ("kõkį-nórs") | no |
| `mark_outside_nucleus` | a mark on a consonant outside the syllable nucleus ("aš̃", "pirm̀oji") | yes |
| `mark_not_canonical` | a mark inside the nucleus, but not on the letter where the accenter puts that mark ("dar̀", "jĩe") | yes |
| `no_nucleus` | the word has no syllable nucleus | no |
| `alignment_error` | the whole sentence could not be aligned (all words) | no |
| `low_alignment` | the word's alignment score < `min_alignment` | yes |
| `no_candidate` | the text mark is not among the candidates | yes |
| `low_margin` | the margin is below the threshold | yes |

`kind` (only when the text mark ≠ the best candidate): `stress_position` – stress on a different syllable, or
stressed↔unstressed; `pitch_accent` – same syllable, different pitch accent; `function_word` – a stressed↔unstressed
mismatch on a function word (`ltstress.phonetics.FUNCTION_WORDS`: never stressed in LIEPA `ir, kad, tai, su, labai,
tik…` + `juk, nė, gi, jog, o` and others), i.e. exactly one of the text mark and `best_form` is unstressed. The stress
of such words is barely audible, so they go to the end of the listening queue; `verdict` is not changed
(`mismatch`/`unclear` as before). Other function-word mismatches (both stressed) get the usual
`stress_position`/`pitch_accent`. A reader that does not know `function_word` can treat it as `stress_position`.

A word with a malformed mark never gets `match`: the manifest needs fixing in that case anyway, and `best_form` is an
acoustic suggestion for listening.

## `ltstress-forms-1` (`ltstress aggregate` output, JSON)

```
{"format": "ltstress-forms-1", "records": <number of ids>, "errors": [{"id", "error"}],
 "function_words": {"groups": <groups with kind "function_word">, "words": <words with kind "function_word" in the queue>},
 "groups": [...]}
```

The last output line of each `id` is used; `errors` – records whose last line has an `error` (their words are not part
of any group). A group is `(word without marks in lower case, text_form in lower case)`:

| field | value |
|---|---|
| `word`, `text_form` | group key |
| `n`, `n_mismatch`, `n_unclear` | word counts |
| `most_common_best` | the most common `best_form` (lower case) |
| `kind` | `"malformed_mark"` – malformed text mark; otherwise the most common `kind` among `mismatch` words (`stress_position` / `pitch_accent` / `function_word` / `null`); without `mismatch` words – `"function_word"` if that is the most common `kind` among `unclear` words |
| `mark_error` | the `mark_*` reason or `null` |
| `inconsistent` | the best forms of confident verdicts (`match` / `mismatch`) differ |
| `examples` | up to 3 `{"id", "i"}`: `mismatch` words (if there are none – any) with the largest margin |

Sorted: `malformed_mark` groups first, then `stress_position`, then by `n_mismatch / n`, `n_mismatch`, `n`;
`function_word` groups always last (after all `stress_position`, `pitch_accent` and kindless groups), in the same
order among themselves. Written next to it: `*.forms.tsv` (the same group fields, `examples` as `id:i,...`) and
`*.forms.errors.tsv` (`id`, `error`).
