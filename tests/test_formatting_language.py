"""Language-routing guards for inverse text normalisation.

The English rules are word-based and must not touch another language, and the
space-token pipeline cannot represent languages written without spaces. These
tests pin the routing, the localised digit rules, the CJK cleanup, and - in the
corruption regression - prove the gate itself is what stops an English scale
word from rewriting French.
"""

from __future__ import annotations

import pytest

from config import Config
from formatting import (
    CURRENCY_SYMBOLS,
    DECIMAL_WORDS,
    DIGIT_LANGUAGES,
    NO_SPACE_LANGUAGES,
    PERCENT_PHRASES,
    PUNCTUATION_BY_LANGUAGE,
    normalise,
)


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")


# (language, spoken source, expected written form)
DIGIT_CASES = [
    # French
    ("fr", "le total est 12 pour cent", "le total est 12 %"),
    ("fr", "3 virgule 14", "3,14"),
    ("fr", "25 euros", "25 €"),
    ("fr", "point d'interrogation", "?"),
    ("fr", "bonjour virgule monde", "bonjour, monde"),
    # Spanish
    ("es", "12 por ciento", "12%"),
    ("es", "3 coma 14", "3,14"),
    ("es", "25 dólares", "25 $"),
    # Indonesian
    ("id", "12 persen", "12%"),
    ("id", "3 koma 14", "3,14"),
    ("id", "25000 rupiah", "25000 Rp"),
    # Vietnamese
    ("vi", "12 phần trăm", "12%"),
    ("vi", "3 phẩy 14", "3,14"),
    ("vi", "25 đồng", "25 ₫"),
]


@pytest.mark.parametrize("language,source,expected", DIGIT_CASES)
def test_digit_language_rules(language, source, expected):
    assert normalise(source, language=language) == expected


@pytest.mark.parametrize("language,source,expected", DIGIT_CASES)
def test_digit_language_normalisation_is_idempotent(language, source, expected):
    assert normalise(expected, language=language) == expected


def test_french_scale_words_are_not_corrupted_by_the_english_word_tables():
    """The language gate, not luck, is what protects French.

    The specified phrase "deux millions d'euros" happens to survive the English
    parser because its scale table knows only the singular "million", so a
    control on that string alone could not detect a removed gate. The singular
    "deux million d'euros" is the form the English parser DOES rewrite, so it is
    the discriminator: French must leave both alone while English corrupts the
    singular one. Remove the gate and the French assertions fail.
    """
    plural = "deux millions d'euros"
    singular = "deux million d'euros"

    # The specified regression, with no digits inserted.
    assert normalise(plural, language="fr") == plural
    assert "1,000,000" not in normalise(plural, language="fr")

    # The discriminating form, also untouched in French.
    assert normalise(singular, language="fr") == singular
    assert not any(char.isdigit() for char in normalise(singular, language="fr"))

    # Negative control: the English pipeline corrupts the singular scale word.
    corrupted = normalise(singular, language="en")
    assert corrupted != singular
    assert "1,000,000" in corrupted


CJK_CASES = [
    ("zh", "你好 ， 世界", "你好，世界"),
    ("zh", "你好, 世界", "你好，世界"),
    ("zh", "你好! 世界", "你好！世界"),
    ("zh", "今天天气很好。", "今天天气很好。"),
    ("ja", "こんにちは 。", "こんにちは。"),
    ("ja", "こんにちは、 世界", "こんにちは、世界"),
]


@pytest.mark.parametrize("language,source,expected", CJK_CASES)
def test_no_space_language_cleanup(language, source, expected):
    assert normalise(source, language=language) == expected


@pytest.mark.parametrize("language,source,expected", CJK_CASES)
def test_no_space_language_normalisation_is_idempotent(language, source, expected):
    assert normalise(expected, language=language) == expected


def test_no_space_branch_does_not_run_english_rules():
    """Number parsing and currency are word-based and unsafe without spaces."""
    assert normalise("five percent", language="zh") == "five percent"
    assert normalise("five percent", language="ja") == "five percent"


def test_thai_only_has_whitespace_tidied():
    """Thai separates phrases, not words, so nothing word-based is safe."""
    sentence = "สวัสดี ครับ ผมชื่อ"
    assert normalise(sentence, language="th") == sentence
    assert normalise("สวัสดี   ครับ", language="th") == "สวัสดี ครับ"


def test_unknown_language_is_whitespace_tidy_only():
    assert normalise("  hello   world  ", language="de") == "hello world"
    assert normalise("five percent", language="de") == "five percent"


IDEMPOTENT = [
    ("fr", "le total est 12 %"),
    ("fr", "3,14"),
    ("fr", "25 €"),
    ("fr", "bonjour, monde"),
    ("es", "12%"),
    ("id", "25000 Rp"),
    ("vi", "25 ₫"),
    ("zh", "你好，世界"),
    ("ja", "こんにちは。"),
    ("th", "สวัสดี ครับ"),
    ("de", "hello world"),
    ("en", "5%"),
    ("en", "the end."),
]


@pytest.mark.parametrize("language,text", IDEMPOTENT)
def test_idempotency_across_languages(language, text):
    assert normalise(text, language=language) == text


def test_english_legacy_defaults_still_hold_without_arguments():
    assert normalise("five percent") == "5%"
    assert normalise("the end period") == "the end."


def test_config_language_is_used_when_no_explicit_language(config):
    config.set("language", "fr")
    assert normalise("25 euros", config) == "25 €"


def test_explicit_language_overrides_config(config):
    config.set("language", "fr")
    assert normalise("25 euros", config, language="en") == "25 euros"


def test_auto_and_empty_config_language_fall_back_to_english(config):
    config.set("language", "auto")
    assert normalise("five percent", config) == "5%"
    config.set("language", "")
    assert normalise("five percent", config) == "5%"


def test_language_codes_are_reduced_to_the_base(config):
    assert normalise("five percent", language="en-US") == "5%"
    assert normalise("25 euros", language="fr-FR") == "25 €"


def test_literal_escape_hatch_works_for_every_language():
    for language in ("fr", "es", "id", "vi", "zh", "ja", "th", "de"):
        assert normalise("literal five percent", language=language) == "five percent"


def test_public_constant_sets_are_wired_up():
    assert NO_SPACE_LANGUAGES == frozenset({"zh", "ja", "th", "yue"})
    assert DIGIT_LANGUAGES == frozenset({"fr", "es", "id", "vi"})
    assert set(DECIMAL_WORDS) == set(DIGIT_LANGUAGES)
    assert set(PERCENT_PHRASES) == set(DIGIT_LANGUAGES)
    assert set(CURRENCY_SYMBOLS) == set(DIGIT_LANGUAGES)
    assert set(PUNCTUATION_BY_LANGUAGE) == set(DIGIT_LANGUAGES)
