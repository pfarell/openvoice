"""Guards for inverse text normalisation.

Two properties matter and both are tested here: the rules fire on clearly spoken
forms, and they do NOT fire on prose that merely contains a number word. The
danger cases are the ones that would corrupt text if the parser were greedy.
"""

from __future__ import annotations

import pytest

from config import Config
from formatting import normalise


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")


CASES = [
    ("twelve point five millimetres", "12.5 mm"),
    ("five percent", "5%"),
    ("hello world exclamation mark", "hello world!"),
    ("the end period", "the end."),
    ("first item comma second item", "first item, second item"),
    ("twenty five thousand", "25,000"),
    ("nine thousand", "9000"),
    ("two million", "2,000,000"),
    ("three hundred and sixty", "360"),
    ("minus five", "-5"),
    ("negative three point two", "-3.2"),
    ("twenty first century", "21st century"),
    ("three quarters of a litre", "3/4 L"),
    ("five plus three", "5+3"),
    ("five squared", "5\u00b2"),
    ("twelve millimetres", "12 mm"),
    ("five kilograms", "5 kg"),
    ("sixty hertz", "60 Hz"),
    ("three point five gigabytes", "3.5 GB"),
    ("we talked for three hours", "we talked for 3 h"),
]

DANGER = [
    "I have two cats",
    "one of the best",
    "two of them",
    "first of all",
    "second opinion",
    "third time lucky",
    "someone often says none of it matters",
    "handsome bone tone",
    "the tone of the story",
    "we went to the shop",
    "a quarter of the way",
    "three quarters of the way",
    "she is a ten out of ten",
]


@pytest.mark.parametrize("source,expected", CASES)
def test_spoken_forms_are_written_out(source, expected):
    assert normalise(source) == expected


@pytest.mark.parametrize("source", DANGER)
def test_prose_with_number_words_is_left_alone(source):
    assert normalise(source) == source


@pytest.mark.parametrize("source,expected", CASES)
def test_normalisation_is_idempotent(source, expected):
    assert normalise(expected) == expected


def test_negative_control_a_greedy_parser_would_change_the_danger_cases():
    """Proves the danger-case assertions discriminate.

    Replace every number word with its digit unconditionally - the naive
    implementation - and check that it corrupts at least three of the phrases
    that must survive. Without this, "leaves prose alone" could pass simply
    because nothing was implemented.
    """
    digits = {
        "two": "2", "one": "1", "first": "1st", "second": "2nd", "third": "3rd",
        "ten": "10", "quarter": "1/4",
    }

    def greedy(text: str) -> str:
        return " ".join(digits.get(word.strip(".,"), word) for word in text.split())

    changed = [phrase for phrase in DANGER if greedy(phrase) != phrase]
    assert len(changed) >= 3, (
        f"the greedy control changed only {len(changed)} danger phrases, so the "
        "'must not change' assertions cannot be detecting anything"
    )


def test_escape_hatch_bypasses_everything():
    assert normalise("literal five percent") == "five percent"


def test_escape_hatch_is_documented_as_one_shot():
    """The escape is deliberately one-shot: normalising its output again would
    apply the rules. This test pins that behaviour instead of pretending the
    function is idempotent for escaped input."""
    once = normalise("literal five percent")
    assert once == "five percent"
    assert normalise(once) == "5%"


def test_spoken_punctuation_can_be_switched_off(config):
    config.set("spoken_punctuation", False)
    assert normalise("the end period", config) == "the end period"
    assert normalise("five percent", config) == "5%"


def test_empty_and_whitespace_are_safe():
    assert normalise("") == ""
    assert normalise("   ") == "   " or normalise("   ").strip() == ""


def test_plain_sentence_is_untouched():
    text = "send the invoice to the accounting department tomorrow morning"
    assert normalise(text) == text
