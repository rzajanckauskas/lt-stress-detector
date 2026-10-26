"""
Phonetics helpers: syllable nuclei, stress marks, the stress mark of a word (syllable, accent).

The mark and syllable rules are the same as in the same author's TTS text accenter, so the detector reads stress
marks the same way the accenter writes them.
Marks: NFC letters + separate U+0300 / U+0301 / U+0303 (never U+0307).
"""
import unicodedata

# --- shared with the TTS text accenter ---
ACCENTS = {"̀": 0, "́": 1, "̃": 2}   # short/grave, acute, tilde
ACC_NAMES = ["short", "acute", "tilde"]
COMB = {"short": "̀", "acute": "́", "tilde": "̃"}

VOW = set("aeiouyąęėįųū"); SON = set("jlmnr"); SHORT_VOW = set("aeiu")
DIPH = {"ai", "au", "ei", "ui", "ie", "uo", "oi", "ou", "eu"}
GLIDE_VOW = set("aąoųuū")


def nucleus_groups(word):
    w, groups, cur = word.lower(), [], []
    for i, c in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if c in VOW:
            if not cur:
                cur = [i]; continue
            prev = w[cur[-1]]
            glide = (len(cur) == 1 and prev == "i" and cur[0] > 0
                     and w[cur[0] - 1] not in VOW and c in GLIDE_VOW)
            if prev + c in DIPH or glide:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        elif cur and c in SON and nxt not in VOW and w[cur[-1]] in SHORT_VOW:
            cur.append(i); groups.append(cur); cur = []
        else:
            if cur: groups.append(cur)
            cur = []
    if cur: groups.append(cur)
    return groups


def mark_word(word, syl, acc):
    groups = nucleus_groups(word)
    if not groups or syl >= len(groups):
        return word
    g = groups[syl]; w = word.lower()
    vowels = [i for i in g if w[i] in VOW]
    if (len(vowels) >= 2 and w[vowels[0]] == "i" and w[vowels[1]] in GLIDE_VOW
            and (vowels[0] == 0 or w[vowels[0] - 1] not in VOW)):
        vowels = vowels[1:]
    if acc == "tilde" and len(g) > 1:
        if w[g[-1]] in SON:
            after = w[g[-1] + 1] if g[-1] + 1 < len(w) else ""
            if (len(vowels) >= 2 and w[vowels[0]] + w[vowels[1]] in ("ie", "uo")
                    and not (w[vowels[0]] == "i" and after == "t")):
                pos = vowels[1]
            else:
                pos = g[-1]
        else:
            pos = vowels[1] if len(vowels) >= 2 else g[-1]
    elif acc == "short" and len(vowels) >= 2:
        pos = vowels[1]
    else:
        pos = vowels[0] if vowels else g[0]
    return word[:pos + 1] + COMB[acc] + word[pos + 1:]


# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------
def split_marks(s):
    """Returns (NFC text without stress marks, [(position_in_NFC_text, accent_code)])."""
    out, marks = [], []
    for ch in unicodedata.normalize("NFD", s):
        if ch in ACCENTS:
            marks.append((len(out), ACCENTS[ch]))
        elif ch == "̇" and out and out[-1].lower() == "i":
            continue                      # "i̇̀" – a dictionary convention, not needed by the model
        else:
            out.append(ch)
    # recompute the positions after NFC composition
    nfd = "".join(out)
    nfc, pos_map, i = [], {}, 0
    while i < len(nfd):
        j = i + 1
        while j < len(nfd) and unicodedata.combining(nfd[j]):
            j += 1
        nfc.append(unicodedata.normalize("NFC", nfd[i:j]))
        for k in range(i, j):
            pos_map[k + 1] = len(nfc)     # mark after the k-th NFD character -> after the NFC character
        i = j
    marks = [(pos_map.get(p, len(nfc)), a) for p, a in marks]
    return "".join(nfc), marks


def to_model_form(s):
    """NFC letters + separate U+0300/0301/0303 marks (the form the stress model was trained on)."""
    plain, marks = split_marks(s)
    chars = list(plain)
    for pos, acc in sorted(marks, reverse=True):
        chars.insert(pos, COMB[ACC_NAMES[acc]])
    return "".join(chars)


def strip_marks(s):
    return split_marks(s)[0]


def stress_variants(word):
    """All possible stress variants of a word (syllable × accent) in model form; short (`) only on short vowels,
    acute/circumflex not on a single short i/u."""
    plain = strip_marks(word)
    w = plain.lower()
    out = []
    for si, g in enumerate(nucleus_groups(plain)):
        vowels = [w[i] for i in g if w[i] in VOW]
        single = len(g) == 1 and vowels
        for acc in ACC_NAMES:
            if acc == "short" and single and vowels[0] not in SHORT_VOW:
                continue
            if acc != "short" and single and vowels[0] in "iu":
                continue
            f = to_model_form(mark_word(plain, si, acc))
            if f not in out:
                out.append(f)
    return out


# --- shared with the author's stress-check tooling ---
MARKS = {"̀": 0, "́": 1, "̃": 2}


def stress_key(marked):
    """'nesiáukština' -> ('nesiaukština', (4, 1)); no stress mark -> (word, None)."""
    plain, key, n = [], None, 0          # n – number of letters in NFC form (ž, ū, ę count as one letter each)
    for ch in unicodedata.normalize("NFD", marked):
        if ch in MARKS:
            key = (n - 1, MARKS[ch])
            continue
        if not unicodedata.combining(ch):
            n += 1
        plain.append(ch)
    return unicodedata.normalize("NFC", "".join(plain)).lower(), key


VOWELS = set("aąeęėiįyouųū")


def canon(word, key):
    """An acute on a mixed diphthong with i, u (il, im, in, ir, ul, um, un, ur) is written as a grave on i/u –
    "šìrdį" = "šírdį" (LIEPA writes the acute). For comparison both spellings are normalized to the grave."""
    if key and key[1] == 1 and word[key[0]:key[0] + 1] in ("i", "u") and word[key[0] + 1:key[0] + 2] in ("l", "m", "n", "r") \
            and word[key[0] + 2:key[0] + 3] not in VOWELS:
        return (key[0], 0)
    return key


# ---------------------------------------------------------------------------
# Function words
# ---------------------------------------------------------------------------
# Words that are never stressed in the LIEPA-TTS (Regina) data.
LIEPA_UNSTRESSED = set("""ir kad ar iš ne tai su bet apie tik nuo labai dėl jau tačiau per po
už bei nei be nors prie arba""".split())
# Function words: their stress is barely audible (confirmed by listening: juk, nė), so neither a "stressed" nor an
# "unstressed" label is reliable – masked in training, excluded from evaluation, and when checking a
# stressed↔unstressed mismatch gets the kind "function_word".
FUNCTION_WORDS: frozenset[str] = frozenset(LIEPA_UNSTRESSED | {"juk", "nė", "gi", "jog", "o"}
                                           | {"nes", "kai", "negu", "pagal", "tarp", "prieš", "taip", "tad", "lyg", "pro", "ant", "iki", "pas", "virš", "ties", "būk"})


def is_function_word(word: str) -> bool:
    """Whether a word (with or without marks, in any letter case) is a function word (FUNCTION_WORDS)."""
    return strip_marks(word).lower() in FUNCTION_WORDS


# ---------------------------------------------------------------------------
# Stress mark
# ---------------------------------------------------------------------------
Mark = tuple[int, int]      # (syllable index per nucleus_groups, accent: 0 short / 1 acute / 2 circumflex)
CLASSES = ["unstressed", "short", "acute", "circumflex"]


class MultipleMarksError(ValueError):
    pass


def word_mark(form: str) -> Mark | None:
    """Stress-marked word -> (syllable, accent) after canon; no mark -> None; several marks -> MultipleMarksError."""
    if sum(ch in MARKS for ch in unicodedata.normalize("NFD", form)) > 1:
        raise MultipleMarksError(form)
    word, key = stress_key(form)
    if key is None:
        return None
    key = canon(word, key)
    for si, g in enumerate(nucleus_groups(word)):
        if key[0] in g:
            return (si, key[1])
    return None


def mark_error(form_: str) -> str | None:
    """Malformed stress mark in the text (form_ – model form with ≤ 1 mark; word_mark does not check this):
    mark outside the syllable nucleus ("aš̃", "pirm̀oji") -> "mark_outside_nucleus"; inside the nucleus but not on
    the letter where mark_word puts that mark, compared via canon ("dar̀", "jĩe") -> "mark_not_canonical";
    no mark or a well-formed one -> None."""
    word, key = stress_key(form_)
    if key is None:
        return None
    m = word_mark(form_)
    if m is None:
        return "mark_outside_nucleus"
    if canon(*stress_key(marked_form(strip_marks(form_), m))) != canon(word, key):
        return "mark_not_canonical"
    return None


def variants(word: str) -> list[Mark | None]:
    """Unstressed (None) and all possible stress variants; no duplicates, stable order."""
    out: list[Mark | None] = [None]
    for f in stress_variants(word):
        m = word_mark(f)
        if m not in out:
            out.append(m)
    return out


def marked_form(word: str, mark: Mark | None) -> str:
    """The word with a stress mark, in model form; None -> the word unchanged."""
    if mark is None:
        return word
    return to_model_form(mark_word(word, mark[0], ACC_NAMES[mark[1]]))


def nucleus_type(word: str, syllable: int) -> str:
    """Syllable nucleus type for acoustic scoring: short / ae / long / mixed / diphthong / ie_uo;
    an unclassifiable group (e.g. "iei" / "aie" merged by nucleus_groups: prieiga, paieška) -> "other"."""
    w = word.lower()
    idx = nucleus_groups(word)[syllable]
    g = "".join(w[i] for i in idx)
    if g[-1] in SON and len(g) > 1:
        return "mixed"
    if len(g) > 1 and g[0] == "i" and g[1] in GLIDE_VOW and (idx[0] == 0 or w[idx[0] - 1] not in VOW):
        g = g[1:]                              # initial glide i (as in mark_word): "iau" -> "au", "iuo" -> "uo"
        if len(g) == 1:
            return "ie_uo"                     # i + a single vowel after a consonant (ia, iu, ių)
    if g in ("ie", "uo"):
        return "ie_uo"
    if len(g) == 2:
        return "diphthong"
    if g in ("i", "u"):
        return "short"
    if g in ("a", "e"):
        return "ae"
    if g in set("yūoėąęįų"):
        return "long"
    return "other"
