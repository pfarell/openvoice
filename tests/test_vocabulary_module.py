"""Tests for the personal-vocabulary module.

These guard the three levers that make proper nouns survive dictation: prompt
shaping (``build_hotwords``), candidate mining (``suggest_terms``), and learning
from the user's own corrections (``CorrectionLearner``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Config
from vocabulary import (
    DEFAULT_TERMS,
    CorrectionLearner,
    build_hotwords,
    hotwords_for,
    suggest_terms,
    terms_from_config,
)


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")



def _terms_in(prompt: str) -> list[str]:
    """The terms of a hotwords prompt, in order, without the prose framing."""
    body = prompt
    for marker in ("The speaker may mention ", " while dictating"):
        if marker in body:
            body = body.split(marker, 1)[1] if marker.startswith("The") else body.split(marker, 1)[0]
    body = body.strip().rstrip(".")
    body = body.replace(" and ", ", ")
    return [t.strip() for t in body.split(",") if t.strip()]


def test_hotwords_put_the_most_important_term_last():
    result = build_hotwords(["Vint", "Ani", "Zenor"])
    terms = _terms_in(result)
    assert terms[-1] == "Ani", result
    assert terms.index("Vint") < terms.index("Ani")


def test_hotwords_give_the_tail_to_a_short_word_like_term():
    """The ordering was measured wrong once and this pins the fix.

    On a real dictation, a short project name (position 4 of 11) came back as an
    ordinary word and another (position 5) as a different one, while the long
    distinctive names at the tail survived. Short word-like terms are the ones
    the language model substitutes, so they get the tail, not the front.
    """
    result = build_hotwords(["Zylophane", "Qat"])
    assert _terms_in(result)[-1] == "Qat", result
    assert "Zylophane" in result


def test_hotwords_do_not_let_internal_caps_outrank_a_confusable_short_term():
    """Internal capitals mark a term the model is unlikely to mutate, so they must
    not push a short confusable term away from the tail."""
    result = build_hotwords(["ZebrO", "Qat"])
    assert _terms_in(result)[-1] == "Qat", result


def test_hotwords_rank_a_shorter_term_above_a_longer_one():
    """Inverted on measurement: see the docstring on vocabulary._importance."""
    result = build_hotwords(["Qattal", "Qat"])
    assert _terms_in(result)[-1] == "Qat", result
    assert "Qattal" in result


def test_hotwords_respect_the_token_budget_and_drop_low_priority():
    """Trimming happens from the FRONT, so the tail - which is the part that
    carries weight - must survive intact."""
    long_terms = [f"Zylophane{index}" for index in range(14)]
    result = build_hotwords(long_terms + ["Qat"], max_tokens=30)

    assert result
    assert (len(result) + 3) // 4 <= 30
    assert _terms_in(result)[-1] == "Qat", result
    assert len(result.split(", ")) < len(long_terms) + 1


def test_hotwords_empty_input_is_a_safe_empty_string():
    assert build_hotwords([]) == ""
    assert build_hotwords(["Zenor"], max_tokens=0) == ""


def test_terms_from_config_empty_returns_the_builtin_list(config):
    config.set("custom_vocabulary", "")
    assert terms_from_config(config) == DEFAULT_TERMS


def test_terms_from_config_parses_commas_newlines_and_dedupes(config):
    config.set("custom_vocabulary", "Alpha, beta gamma\nDelta, alpha")
    assert terms_from_config(config) == ["Alpha", "beta gamma", "Delta"]


def test_hotwords_for_uses_the_configured_terms(config):
    config.set("custom_vocabulary", "Kleptor, Zylophane")
    result = hotwords_for(config)
    assert result == build_hotwords(terms_from_config(config))
    assert "Zylophane" in result
    assert "Kleptor" in result


def _build_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "Zenor").mkdir(parents=True)
    (root / "Zenor" / "core.py").write_text("", encoding="utf-8")
    (root / "Kestrel").mkdir()
    (root / "Kestrel" / "shapes.py").write_text("", encoding="utf-8")
    (root / "Wallet_Tracker").mkdir()
    (root / "Wallet_Tracker" / "main.py").write_text("", encoding="utf-8")
    (root / "vertexflow-office").mkdir()
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "node_modules" / "InternalOnlyTool").mkdir()
    (root / ".git" / "objects").mkdir(parents=True)
    (root / ".git" / "GitInternalName.py").write_text("", encoding="utf-8")
    (root / "ab").mkdir()
    (root / "1234").mkdir()
    (root / "data").mkdir()
    return root


def test_suggest_terms_finds_names_and_skips_noise(tmp_path):
    terms = suggest_terms(_build_tree(tmp_path))

    assert "Zenor" in terms
    assert "Kestrel" in terms
    assert "Wallet Tracker" in terms
    assert "Vertexflow" in terms

    assert "InternalOnlyTool" not in terms
    assert "GitInternalName" not in terms
    assert "Left Pad" not in terms
    assert "Ab" not in terms
    assert "1234" not in terms
    assert all(len(term) >= 4 for term in terms)


def test_correction_records_a_term_only_after_the_threshold(config):
    learner = CorrectionLearner(config)
    learner.observe("Vynt Career", "Vint Career")
    assert "Vint" not in learner.learned_terms()

    graduated = learner.observe("Vynt Career", "Vint Career")
    assert "Vint" in learner.learned_terms()
    assert "Vint" in graduated
    assert learner.replacements().get("Vynt") == "Vint"


def test_correction_apply_fixes_a_later_transcript(config):
    learner = CorrectionLearner(config)
    for _ in range(2):
        learner.observe("Vynt Career", "Vint Career")
    assert learner.apply("I met Vynt Career today") == "I met Vint Career today"


def test_correction_apply_leaves_unrelated_text_alone(config):
    learner = CorrectionLearner(config)
    for _ in range(2):
        learner.observe("Vynt Career", "Vint Career")
    untouched = "a completely unrelated sentence"
    assert learner.apply(untouched) == untouched


def test_correction_persists_across_instances(config):
    first = CorrectionLearner(config)
    for _ in range(2):
        first.observe("Vynt Career", "Vint Career")

    second = CorrectionLearner(config)
    assert "Vint" in second.learned_terms()
    assert second.apply("Vynt Career") == "Vint Career"


def test_correction_tolerates_a_corrupt_store(config):
    (Path(config.path).parent / "learned_terms.json").write_text("{not json", encoding="utf-8")
    learner = CorrectionLearner(config)
    assert learner.learned_terms() == []
    assert learner.replacements() == {}

    for _ in range(2):
        learner.observe("Vynt Career", "Vint Career")
    assert "Vint" in learner.learned_terms()


def test_forget_removes_a_term_and_its_replacement(config):
    learner = CorrectionLearner(config)
    for _ in range(2):
        learner.observe("Vynt Career", "Vint Career")
    assert "Vint" in learner.learned_terms()

    learner.forget("vint")
    assert "Vint" not in learner.learned_terms()
    assert learner.replacements() == {}
    assert learner.apply("Vynt Career") == "Vynt Career"


def test_negative_control_similarity_gate_discriminates(config):
    """Proves the near-miss gate rejects unrelated pairs, not the threshold.

    Both pairs are observed three times, above the default threshold of two, so
    a low count cannot be what keeps "Vantry" out. The comparison is
    case-sensitive: "ready" -> "Vantry" scores 0.36 and shares no leading
    character, so the gate rejects it, while "NovaGatx" -> "NovaGate"
    shares four leading characters, so the same ``observe`` path accepts it.
    """
    learner = CorrectionLearner(config)
    for _ in range(3):
        learner.observe("ready", "Vantry")
        learner.observe("NovaGatx", "NovaGate")

    assert "Vantry" not in learner.learned_terms()
    assert "ready" not in learner.replacements()

    assert "NovaGate" in learner.learned_terms()
    assert learner.replacements().get("NovaGatx") == "NovaGate"
