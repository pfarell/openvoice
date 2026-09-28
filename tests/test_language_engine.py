"""Guards for the multilingual engine: language choice and the hallucination gate.

Every test injects a fake, duck-typed model onto a Transcriber, so no model is
loaded and no network is touched. The fake records the kwargs whisper would have
received, which is what makes the language decision observable rather than
inferred. Phrase folding is O(len(text)); the rest of the per-call work is O(1).
"""

from __future__ import annotations

import numpy as np
import pytest

import transcribe
from config import Config
from transcribe import Transcriber, normalise_language_code

SAMPLE_RATE = 16000


class FakeSegment:
    """The slice of a faster-whisper segment the transcriber actually reads."""

    def __init__(
        self,
        text,
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        compression_ratio=1.0,
        temperature=0.0,
    ):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio
        self.temperature = temperature


class FakeInfo:
    """The slice of the transcribe info object the fallback path reads."""

    def __init__(self, language="", language_probability=None):
        self.language = language
        self.language_probability = language_probability
        self.all_language_probs = (
            [(language, language_probability)] if language else []
        )


class FakeModel:
    """A duck-typed stand-in for WhisperModel.

    Only ``detect_language``, ``transcribe`` and ``supported_languages`` are read.
    The transcribe kwargs are captured verbatim so each test can assert the
    language and multilingual flag the engine chose.
    """

    def __init__(
        self,
        text="hello world",
        detect=("zh", 0.9),
        info=None,
        supported=("en", "zh", "fr", "ja", "th", "vi", "id"),
    ):
        self._text = text
        self._detect = detect
        self._info = info if info is not None else FakeInfo()
        self.supported_languages = list(supported)
        self.detect_calls = 0
        self.transcribe_kwargs: list[dict] = []

    def detect_language(
        self,
        audio=None,
        language_detection_segments=1,
        language_detection_threshold=0.5,
    ):
        self.detect_calls += 1
        code, probability = self._detect
        return (code, probability, [(code, probability)] if code else [])

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs.append(kwargs)
        return iter([FakeSegment(self._text)]), self._info


@pytest.fixture()
def config(tmp_path):
    return Config(tmp_path / "config.json")


@pytest.fixture()
def make_transcriber(config, monkeypatch):
    """Build a Transcriber whose model is already 'loaded', so load() is a no-op.

    The GPU probe is pinned to CPU so the resolved key is deterministic and no
    real model or CUDA runtime is touched.
    """
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)

    def build(model):
        engine = Transcriber(None, config)
        engine._model = model
        engine._model_key = engine._key()
        return engine

    return build


def _audio(seconds: float = 1.0) -> np.ndarray:
    """A non-silent, long-enough clip that passes the pre-checks."""
    t = np.linspace(0.0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def test_normalise_language_code_accepts_and_rejects():
    assert normalise_language_code("auto") is None
    assert normalise_language_code("") is None
    assert normalise_language_code(None) is None
    assert normalise_language_code("none") is None
    assert normalise_language_code("EN-us") == "en"
    assert normalise_language_code(" zh ") == "zh"
    assert normalise_language_code("zh_Hans") == "zh"


def test_forced_language_skips_detection_and_records_the_code(make_transcriber, config):
    config.set("language", "zh")
    model = FakeModel()
    engine = make_transcriber(model)

    result = engine.transcribe_audio(_audio())

    assert model.detect_calls == 0, "a forced language must not trigger detection"
    assert model.transcribe_kwargs[0]["language"] == "zh"
    assert result.language == "zh"
    assert result.language_detected is False
    assert result.language_probability is None


def test_auto_uses_detection_and_marks_it_detected(make_transcriber, config):
    config.set("language", "auto")
    model = FakeModel(detect=("zh", 0.9))
    engine = make_transcriber(model)

    result = engine.transcribe_audio(_audio())

    assert model.detect_calls == 1
    assert model.transcribe_kwargs[0]["language"] == "zh"
    assert result.language == "zh"
    assert result.language_detected is True
    assert result.language_probability == pytest.approx(0.9)


def test_auto_failed_detection_falls_back_to_whisper_info(make_transcriber, config):
    config.set("language", "auto")
    model = FakeModel(detect=("", 0.0), info=FakeInfo("fr", 0.7))
    engine = make_transcriber(model)

    result = engine.transcribe_audio(_audio())

    assert model.detect_calls == 1
    assert model.transcribe_kwargs[0]["language"] is None
    assert result.language == "fr"
    assert result.language_probability == pytest.approx(0.7)
    assert result.language_detected is False


def test_unsupported_configured_language_degrades_to_auto(make_transcriber, config):
    config.set("language", "zh")
    model = FakeModel(detect=("en", 0.8), supported=("en",))
    engine = make_transcriber(model)

    result = engine.transcribe_audio(_audio())

    assert model.detect_calls == 1
    assert model.transcribe_kwargs[0]["language"] == "en"
    assert result.language_detected is True


def test_detect_per_segment_flag_reaches_whisper(make_transcriber, config):
    config.set("language", "auto")
    model = FakeModel(detect=("en", 0.9))
    engine = make_transcriber(model)
    engine.transcribe_audio(_audio())
    assert model.transcribe_kwargs[0]["multilingual"] is False

    config.set("detect_per_segment", True)
    model = FakeModel(detect=("en", 0.9))
    engine = make_transcriber(model)
    engine.transcribe_audio(_audio())
    assert model.transcribe_kwargs[0]["multilingual"] is True


def test_detect_language_is_fail_open_without_a_detector(make_transcriber, config):
    engine = make_transcriber(object())
    assert engine.detect_language(_audio()) == ("", 0.0)


def test_detect_language_is_fail_open_when_detection_raises(make_transcriber, config):
    class Boom:
        def detect_language(self, **kwargs):
            raise RuntimeError("boom")

    engine = make_transcriber(Boom())
    assert engine.detect_language(_audio()) == ("", 0.0)


def test_engine_name_is_empty_before_load(config, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)
    engine = Transcriber(None, config)
    assert engine.engine_name == ""


def test_engine_name_reports_the_loaded_model(make_transcriber, config):
    engine = make_transcriber(FakeModel())
    assert engine.engine_name == transcribe.SMALL_MULTI_MODEL


def test_fold_phrase_keeps_cjk_and_strips_punctuation_and_space():
    assert transcribe.fold_phrase("谢谢观看。") == "谢谢观看"
    assert transcribe.fold_phrase("Thank you, for watching!") == "thankyouforwatching"
    assert transcribe.fold_phrase("  本集完  ") == "本集完"


@pytest.mark.parametrize(
    "phrase",
    [
        "谢谢观看。",
        "谢谢观看",
        "謝謝觀看",
        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    ],
)
def test_multilingual_hallucination_phrases_are_rejected(
    make_transcriber, config, phrase
):
    config.set("language", "auto")
    engine = make_transcriber(FakeModel(text=phrase))

    result = engine.transcribe_audio(_audio())

    assert not result.accepted
    assert result.reject_reason
    assert "filler phrase" in result.reject_reason


def test_english_hallucination_is_still_rejected(make_transcriber, config):
    config.set("language", "auto")
    engine = make_transcriber(FakeModel(text="Thank you for watching!"))

    result = engine.transcribe_audio(_audio())

    assert not result.accepted
    assert result.reject_reason
    assert "filler phrase" in result.reject_reason


def test_an_ordinary_chinese_sentence_is_accepted(make_transcriber, config):
    config.set("language", "auto")
    engine = make_transcriber(FakeModel(text="今天天气很好，我们出去散步吧。"))

    result = engine.transcribe_audio(_audio())

    assert result.accepted, result.reject_reason


def test_an_ordinary_english_sentence_is_accepted(make_transcriber, config):
    config.set("language", "auto")
    engine = make_transcriber(FakeModel(text="The meeting is tomorrow afternoon."))

    result = engine.transcribe_audio(_audio())

    assert result.accepted, result.reject_reason


def test_negative_control_the_gate_is_exact_not_a_substring(make_transcriber, config):
    """Prove the rejections above discriminate rather than substring-matching.

    "谢谢观看吧" contains the stored phrase "谢谢观看" as a prefix, so a substring
    check would reject it. It is not itself a stored phrase, so the exact match
    must accept it. If this ever starts rejecting, the gate has become a
    substring match and legitimate sentences are at risk.
    """
    config.set("language", "auto")
    engine = make_transcriber(FakeModel(text="谢谢观看吧"))

    result = engine.transcribe_audio(_audio())

    assert result.accepted, result.reject_reason
