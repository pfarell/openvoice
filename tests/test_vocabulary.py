"""Guards for vocabulary biasing and dictation history.

The vocabulary behaviour is the one that fixed a real complaint: proper nouns
were being mangled into ordinary words until the term list was passed as
whisper hotwords.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

import history
from config import Config
from transcribe import DEFAULT_VOCABULARY, vocabulary_for


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")


def test_no_setting_means_no_biasing(config):
    config.set("custom_vocabulary", "")
    assert vocabulary_for(config) == ""


def test_whitespace_only_setting_means_no_biasing(config):
    config.set("custom_vocabulary", "   \n  ")
    assert vocabulary_for(config) == ""


def test_user_terms_win(config):
    config.set("custom_vocabulary", "NovaGate, Zenor")
    assert vocabulary_for(config) == "NovaGate, Zenor"


def test_negative_control_the_empty_default_is_deliberate(config):
    """Proves the empty default above is a choice, not an accident: the shipped
    list holds no vendor vocabulary, and only the user's own terms bias."""
    assert DEFAULT_VOCABULARY.strip() == ""
    config.set("custom_vocabulary", "NovaGate")
    assert vocabulary_for(config) != ""


def test_history_round_trip(config):
    config.set("save_history", True)
    audio = (0.25 * np.sin(np.arange(16000) / 50.0)).astype(np.float32)
    path = history.save(audio, "hello world", config)
    assert path is not None and path.exists()
    assert path.with_suffix(".txt").read_text(encoding="utf-8") == "hello world"
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 16000


def test_history_is_not_written_when_disabled(config):
    config.set("save_history", False)
    audio = np.zeros(1600, dtype=np.float32)
    assert history.save(audio, "text", config) is None
    assert history.entries(config) == []


def test_history_ignores_empty_audio(config):
    assert history.save(np.empty(0, dtype=np.float32), "text", config) is None


def test_history_prunes_to_the_limit(config):
    config.set("save_history", True)
    config.set("history_limit", 3)
    audio = np.zeros(800, dtype=np.float32)
    for index in range(6):
        path = history.save(audio, f"entry {index}", config)
        assert path is not None
    kept = history.entries(config)
    assert len(kept) == 3, f"expected 3 entries, found {len(kept)}"
    for path in kept:
        assert path.with_suffix(".txt").exists(), "text sidecar must be pruned with its wav"
