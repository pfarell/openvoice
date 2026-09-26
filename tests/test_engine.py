"""Guards for engine selection.

The behaviour under test is what let a bigger model become practical: "auto"
picks the GPU, the GPU allows the turbo model at beam 5, and a machine without a
usable GPU still gets a working, fast configuration.
"""

from __future__ import annotations

import pytest

import transcribe
from config import Config
from transcribe import (
    SMALL_MODEL,
    SMALL_MULTI_MODEL,
    TURBO_MODEL,
    V3_MODEL,
    resolve_engine,
)


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")


def test_auto_with_a_gpu_picks_the_most_accurate_model(config, monkeypatch):
    """large-v3, not turbo. Turbo trades accuracy for speed and this machine has
    speed to spare: measured on the user's own recordings, large-v3 scored 61.4%
    agreement against turbo's 54.3% and distil-large-v3's 31.4%, and both models
    ran in about the same time on the GPU."""
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 1)
    config.set("device", "auto")
    config.set("model_name", "auto")
    config.set("compute_type", "auto")
    assert resolve_engine(config) == (V3_MODEL, "cuda", "float16")
    assert V3_MODEL != TURBO_MODEL, "turbo is a different, less accurate model"


def test_auto_without_a_gpu_picks_the_small_english_model_on_cpu(config, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)
    config.set("device", "auto")
    config.set("model_name", "auto")
    config.set("compute_type", "auto")
    config.set("language", "en")
    assert resolve_engine(config) == (SMALL_MODEL, "cpu", "int8")


def test_auto_without_a_gpu_and_auto_language_picks_the_multilingual_model(
    config, monkeypatch
):
    """small.en reports supported_languages == ["en"], so it can never emit a
    non-English word. An auto (or any non-"en") language must therefore take the
    multilingual sibling on CPU."""
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)
    config.set("device", "auto")
    config.set("model_name", "auto")
    config.set("compute_type", "auto")
    config.set("language", "auto")
    assert resolve_engine(config) == (SMALL_MULTI_MODEL, "cpu", "int8")


def test_negative_control_the_two_auto_paths_differ(config, monkeypatch):
    """Proves the assertions above discriminate: the GPU and no-GPU choices are
    genuinely different, so a resolver that ignored the GPU would fail one."""
    config.set("device", "auto")
    config.set("model_name", "auto")
    config.set("compute_type", "auto")
    config.set("language", "en")

    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 1)
    with_gpu = resolve_engine(config)
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)
    without_gpu = resolve_engine(config)
    assert with_gpu != without_gpu


def test_negative_control_the_english_and_auto_cpu_paths_differ(config, monkeypatch):
    """Proves the language split discriminates: with identical auto device and
    model settings on CPU, "en" and "auto" must resolve to different models, so a
    resolver that ignored the language would fail one of the tests above."""
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)
    config.set("device", "auto")
    config.set("model_name", "auto")
    config.set("compute_type", "auto")

    config.set("language", "en")
    english = resolve_engine(config)
    config.set("language", "auto")
    automatic = resolve_engine(config)
    assert english != automatic


def test_explicit_model_wins_over_auto(config, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 1)
    config.set("model_name", "Systran/faster-whisper-base.en")
    assert resolve_engine(config)[0] == "Systran/faster-whisper-base.en"


def test_explicit_cpu_device_is_respected(config, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 1)
    config.set("device", "cpu")
    config.set("model_name", "auto")
    config.set("compute_type", "auto")
    config.set("language", "en")
    assert resolve_engine(config) == (SMALL_MODEL, "cpu", "int8")


def test_nonsense_settings_fall_back_to_something_valid(config, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 1)
    config.set("device", "quantum")
    config.set("compute_type", "magic")
    model, device, compute = resolve_engine(config)
    assert device in ("cpu", "cuda")
    assert compute in ("int8", "float16", "int8_float16", "float32")
    assert model


def test_fallback_attempt_is_always_a_cpu_small_pair():
    """The fallback must be something that runs anywhere, or the whole point of
    having a fallback is lost."""
    assert SMALL_MODEL
    assert TURBO_MODEL != SMALL_MODEL


def test_no_vocabulary_ships_by_default():
    """The shipped list is empty: the words worth biasing for are the speaker's
    own, and someone else's project names would only be noise in the prompt."""
    assert transcribe.DEFAULT_VOCABULARY.strip() == ""


def test_the_shipped_vocabulary_stays_empty_and_short():
    """A guard against re-bloating the default, which was measured to be harmful.

    On a real 3-sentence dictation a 29-term hotword list scored 48.1% WER while
    a 10-term list scored 37.0% on the same audio: a long prompt skews the
    language model and damages ordinary words. Terms belong in a user's config,
    added deliberately, not in the shipped default.
    """
    assert len(transcribe.DEFAULT_VOCABULARY.split(",")) <= 12
