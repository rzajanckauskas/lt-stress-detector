import unicodedata

import pytest

from ltstress.phonetics import (MultipleMarksError, marked_form, mark_error, nucleus_type, strip_marks, to_model_form,
                                variants, word_mark)


def test_mark_simple():
    assert word_mark("laĩvas") == (0, 2)
    assert word_mark("plaukìmo") == (1, 0)
    assert word_mark("júostos") == (0, 1)
    assert word_mark("ir") is None


def test_mark_mixed_diphthong_normalized():
    assert word_mark("šìrdį") == word_mark("šírdį") == (0, 0)


def test_mark_position_differs():
    assert word_mark("dalỹkus") == (1, 2)
    assert word_mark("dalykùs") == (2, 0)


def test_multiple_marks():
    with pytest.raises(MultipleMarksError):
        word_mark("kõkį-nórs")


def test_variants_include_none_and_text_form():
    v = variants("dalykus")
    assert v[0] is None and (1, 2) in v and (2, 0) in v and len(v) == len(set(v))


def test_marked_form_separate_marks():
    f = marked_form("aukštas", (0, 2))
    assert "̃" in f and unicodedata.is_normalized("NFC", strip_marks(f))


def test_nucleus_type():
    assert nucleus_type("aukštas", 0) == "diphthong"
    assert nucleus_type("vilkas", 0) == "mixed"
    assert nucleus_type("dieną", 0) == "ie_uo"
    assert nucleus_type("namas", 0) == "ae"
    assert nucleus_type("būti", 0) == "long"
    assert nucleus_type("būti", 1) == "short"


def test_nucleus_type_glide_i():
    assert nucleus_type("kiaulė", 0) == "diphthong"
    assert nucleus_type("kiaušinis", 0) == "diphthong"
    assert nucleus_type("šiuo", 0) == "ie_uo"
    assert nucleus_type("ryšių", 1) == "ie_uo"


def test_nucleus_type_unknown_group_other():
    """nucleus_groups merges "iei" / "aie" into one group – type "other", not an error."""
    assert nucleus_type("prieiga", 0) == "other"
    assert nucleus_type("paieška", 0) == "other"


# --- F1: malformed text marks ---------------------------------------------------------------------
def test_mark_outside_nucleus_none():
    # mark on a consonant outside the nucleus: word_mark (used for LIEPA preparation) stays None
    assert word_mark(to_model_form("aš̃")) is None
    assert word_mark(to_model_form("pirm̀oji")) is None


@pytest.mark.parametrize("word", ["aš̃", "pirm̀oji", "jis̃", "k̀ur", "buṽo"])
def test_mark_error_outside_nucleus(word):
    assert mark_error(to_model_form(word)) == "mark_outside_nucleus"


@pytest.mark.parametrize("word", ["dar̀", "jĩe", "kàip", "mań", "prĩncas"])
def test_mark_error_not_canonical(word):
    assert word_mark(to_model_form(word)) is not None
    assert mark_error(to_model_form(word)) == "mark_not_canonical"


@pytest.mark.parametrize("word", ["laĩvas", "šìrdį", "šírdį", "dalykùs", "Jiẽ", "kaĩp", "ir", "dàr", "júostos"])
def test_mark_error_well_formed(word):
    assert mark_error(to_model_form(word)) is None


# --- function words ---------------------------------------------------------------------------------
def test_function_words_liepa_set_and_extra():
    from ltstress.phonetics import FUNCTION_WORDS
    liepa = set("ir kad ar iš ne tai su bet apie tik nuo labai dėl jau tačiau per po už bei nei be nors prie arba".split())
    assert isinstance(FUNCTION_WORDS, frozenset) and FUNCTION_WORDS == liepa | {"juk", "nė", "gi", "jog", "o"} | {"nes", "kai", "negu", "pagal", "tarp", "prieš", "taip", "tad", "lyg", "pro", "ant", "iki", "pas", "virš", "ties", "būk"}


@pytest.mark.parametrize("word,expected", [("Taĩ", True), ("júk", True), ("NĖ", True), ("nė̃", True), ("sù", True),
                                           ("o", True), ("jūsų", False), ("taip", True), ("laĩvas", False),
                                           ("iš-tikrųjų", False)])
def test_is_function_word(word, expected):
    from ltstress.phonetics import is_function_word
    assert is_function_word(word) is expected
