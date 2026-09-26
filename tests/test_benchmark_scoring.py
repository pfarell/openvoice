"""Guards for the benchmark scorer.

The scorer decides whether the app is improving, so a wrong scorer silently
invalidates every measurement. It must fold formatting differences and still
count real wording errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from benchmark import canonical, word_error_rate  # noqa: E402

FORMATTING_EQUIVALENT = [
    ("five thousand dollars", "$5,000"),
    ("twelve point five percent", "12.5%"),
    ("the range is twenty to thirty units", "the range is 20 to 30 units"),
    ("The meeting is tomorrow", "The meeting is tomorrow."),
]

REAL_ERRORS = [
    ("send the invoice", "bend the invoice", 1 / 3),
    ("they are going to their office", "they're going to there office", 1 / 6),
    ("send the invoice to the accounting department tomorrow morning",
     "and the invoice to the accounting department is more important", None),
]


@pytest.mark.parametrize("reference,hypothesis", FORMATTING_EQUIVALENT)
def test_formatting_differences_score_as_perfect(reference, hypothesis):
    rate, _ = word_error_rate(reference, hypothesis)
    assert rate == 0.0, f"{reference!r} vs {hypothesis!r} scored {rate:.2f}"


@pytest.mark.parametrize("reference,hypothesis,expected", REAL_ERRORS)
def test_real_wording_errors_are_counted(reference, hypothesis, expected):
    rate, edits = word_error_rate(reference, hypothesis)
    assert rate > 0.0
    if expected is not None:
        assert rate == pytest.approx(expected, abs=0.02)
    assert edits, "a non-zero rate must come with the edits that produced it"


def test_spoken_punctuation_scores_perfect_after_the_itn_pass():
    """The wording metric does not know about spoken punctuation words - that is
    what formatting.normalise is for, and the real scorer applies it to both sides
    before comparing. This pins that behaviour."""
    from formatting import normalise

    spoken = "she said hello comma then left period"
    written = "she said hello, then left."
    rate, _ = word_error_rate(normalise(spoken), written)
    assert rate == 0.0, f"after ITN the two differ by {rate:.2f}"


def test_spoken_punctuation_without_the_itn_pass_is_expected_to_differ():
    """The control for the test above: without normalisation the same pair does
    NOT match, so the assertion above is actually testing the ITN pass."""
    spoken = "she said hello comma then left period"
    written = "she said hello, then left."
    rate, _ = word_error_rate(spoken, written)
    assert rate > 0.2


def test_negative_control_the_scorer_is_not_always_zero():
    """Proves the perfect-score assertions can fail: identical text must score 0
    and unrelated text must not."""
    assert word_error_rate("same words here", "same words here")[0] == 0.0
    assert word_error_rate("same words here", "totally different thing")[0] > 0.5


def test_canonical_does_not_weld_words_together():
    """Regression: the number pattern once swallowed its trailing space, so
    "five thousand dollars" became "5000dollars"."""
    assert "5000dollars" not in canonical("five thousand dollars")
    assert canonical("five thousand dollars") == "5000"


def test_canonical_handles_a_number_run_between_other_words():
    assert canonical("about twenty five thousand units") == "about 25000 units"
