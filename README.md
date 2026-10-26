# lt-stress-detector

Lithuanian lexical stress and pitch accent (*priegaidė*) detection from speech.

Given a recording and its stress-marked transcript, the tool checks every word against the audio. If the stress mark in
the text does not match what the speaker said, the word is reported. The main use is cleaning text-to-speech training
data: automatic text accenters make systematic mistakes, and a human only needs to listen to the reported words. The
same tool can also annotate stress in recordings without a stress-marked transcript.

## How it works

1. **Alignment.** CTC forced alignment with
   [`sammy786/wav2vec2-xlsr-lithuanian`](https://huggingface.co/sammy786/wav2vec2-xlsr-lithuanian) gives the time span
   of every word and syllable nucleus.
2. **Frame classifier.** [`renaldux/lt-stress-detector`](https://huggingface.co/renaldux/lt-stress-detector) labels every
   20 ms frame as one of four classes:
   - unstressed;
   - stressed short;
   - acute (*tvirtapradė*);
   - circumflex (*tvirtagalė*).

   It is a w2v-BERT 2.0 model fine-tuned on LIEPA-TTS from
   [`classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier`](https://huggingface.co/classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier).
3. **Forced choice.** Every possible stress variant of the word (syllable × accent) is scored against the frame
   posteriors. If the best-matching variant differs from the text, the word is reported as a mismatch; if the decision
   is uncertain, it is reported as unclear. The decision thresholds ship with the model (`thresholds.json`).

Accuracy on a held-out speaker (stress errors were injected into correct transcripts):

| | |
|---|---|
| stress position errors caught | 99.5 % |
| pitch accent errors caught | 96.3 % |
| correct words flagged | 6.2 % |
| exact accented form without text | 93.8 % |

Short function words (`ir`, `kad`, `juk`, `nes`…) have barely audible stress. They are not scored, and mismatches on
them are reported in a separate group.

## Installation

Requires Python 3.10+. A GPU is recommended; on CPU the tool works but is slow.

```bash
python3 -m venv venv
./venv/bin/pip install -e .
```

## Usage

The input is a JSONL file with one recording per line:

```json
{"id": "s1", "audio_path": "wav/s1.wav", "text": "kad Vokietijà gãli kãrą pralaimė́ti"}
```

Stress marks are separate combining characters after the letter: U+0300 for short, U+0301 for acute and U+0303 for
circumflex. Unmarked words are treated as unstressed.

```bash
# word verdicts (the model and the thresholds are downloaded from Hugging Face)
ltstress check --input manifest.jsonl --output result.jsonl

# group mismatches by word form for listening -> result.forms.json / .forms.tsv / .forms.errors.tsv
ltstress aggregate result.jsonl

# annotation mode: the best stress variant for each word, without trusting the marks in the text
ltstress check --input manifest.jsonl --output result.jsonl --annotate
```

Each word in the output has these fields:
- `verdict`: `match`, `mismatch` or `unclear`;
- `kind`: the kind of mismatch – `stress_position`, `pitch_accent` or `function_word` (`reason` tells why a word is
  `unclear`, e.g. a malformed mark);
- `best_form`: the form that best matches the audio;
- `scores`: scores of all candidate variants.

A re-run continues where it stopped. The output formats are described in [FORMATS.md](FORMATS.md).

## Limitations

- Trained and evaluated on natural read speech. On synthetic (TTS) speech, pitch accent decisions are less reliable;
  stress position is the more robust output.
- Pitch accent verdicts should be confirmed against a dictionary or by a linguist.
- Only four LIEPA speakers were seen in training.

## Tests

```bash
./venv/bin/pip install -e ".[test]"
./venv/bin/pytest -m "not gpu"     # without a GPU
./venv/bin/pytest                  # with a GPU: also runs the alignment tests
```

## Santrauka (LT)

Lietuvių kalbos kirčio vietos ir priegaidės detektorius iš garso. Jam pateikiamas įrašas ir sukirčiuotas tekstas.
Detektorius kiekvienam žodžiui patikrina, ar teksto kirčio ženklas atitinka tai, ką ištarė diktorius, ir pažymi
nesutapimus perklausai. Taip valomi TTS mokymo duomenys nuo automatinio kirčiuoklio klaidų. Detektorius gali ir
sukirčiuoti įrašą be kirčiuoto teksto.

## Licenses

- **Code:** Apache-2.0 ([LICENSE](LICENSE)).
- **Stress model** [`renaldux/lt-stress-detector`](https://huggingface.co/renaldux/lt-stress-detector): CC BY-SA 4.0,
  the same license as its base model. The base model is described in Rupnik, Ljubešić, Porupski,
  [arXiv:2505.24571](https://arxiv.org/abs/2505.24571).
- **Alignment model:** Apache-2.0.
- **Test recording** `tests/data/liepa_fixture.wav`: from LIEPA-TTS (CC BY 4.0, Vytautas Magnus University).
