"""Language guards for the whisper prompt builder.

An English prose prompt biases a non-English transcription toward English, so a
non-English prompt carries no English prose and no English " and " connective.
The tail ordering and the token budget must behave the same either way.
"""

from __future__ import annotations

import pytest

from config import Config
from vocabulary import build_hotwords, hotwords_for, terms_from_config

PROSE_MARKERS = ("The speaker may mention", "while dictating")


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")


def test_non_english_prompt_has_no_english_prose():
    result = build_hotwords(["Zenor", "Vint"], language="ja")

    assert "The speaker may mention" not in result
    assert "while dictating" not in result
    assert " and " not in result
    assert "Zenor" in result
    assert "Vint" in result
    # Bare comma-separated list, most important term last.
    assert result == "Zenor, Vint"


def test_english_default_is_unchanged_and_equals_explicit_en():
    default = build_hotwords(["Zenor", "Vint"])
    explicit = build_hotwords(["Zenor", "Vint"], language="en")

    assert default == explicit
    for marker in PROSE_MARKERS:
        assert marker in default
    assert " and " in default


def test_falsy_or_unknown_language_keeps_the_english_prose(config):
    default = build_hotwords(["Zenor", "Vint"])
    assert build_hotwords(["Zenor", "Vint"], language="") == default
    assert build_hotwords(["Zenor", "Vint"], language=None) == default


def test_non_english_tail_ordering_is_preserved():
    """The shortest, most confusable term must still take the tail."""
    result = build_hotwords(["Zylophane", "Qat"], language="ja")

    assert result.split(", ")[-1] == "Qat", result
    assert "Zylophane" in result


def test_non_english_token_budget_is_respected():
    long_terms = [f"Zylophane{index}" for index in range(14)]
    result = build_hotwords(long_terms + ["Qat"], max_tokens=30, language="ja")

    assert result
    assert (len(result) + 3) // 4 <= 30
    assert result.split(", ")[-1] == "Qat", result
    assert len(result.split(", ")) < len(long_terms) + 1


def test_non_english_empty_rules_match_english():
    assert build_hotwords([], language="ja") == ""
    assert build_hotwords(["Zenor"], max_tokens=0, language="ja") == ""


def test_hotwords_for_forwards_the_language(config):
    result = hotwords_for(config, language="ja")

    assert result == build_hotwords(terms_from_config(config), language="ja")
    assert "The speaker may mention" not in result
