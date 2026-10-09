"""Guards for engine loading when the runtime can build but cannot compute.

CTranslate2 loads its CUDA runtime DLLs lazily, on the first encode rather than
on construction, so a successful ``WhisperModel(...)`` proves nothing about
whether the engine can actually run. These tests pin the defence: a candidate
that cannot encode is rejected during ``load()`` and the fallback is used, and a
model that fails mid-run is dropped so the next call reloads through the
fallback list instead of hanging forever in "transcribing".
"""

from __future__ import annotations

import numpy as np
import pytest

import transcribe
from config import Config
from transcribe import SMALL_MODEL, V3_MODEL, Transcriber

SAMPLE_RATE = 16000


class FlakyRuntimeModel:
    """A duck-typed WhisperModel that fails exactly where the real one did.

    Construction succeeds, like CTranslate2 with a missing cublas DLL;
    ``transcribe`` raises for the names in ``broken``.
    """

    broken: set = set()
    built: list = []

    def __init__(
        self,
        model_name,
        device="cpu",
        compute_type="int8",
        local_files_only=False,
    ):
        self.model_name = model_name
        type(self).built.append(model_name)

    def transcribe(self, audio, **kwargs):
        if self.model_name in type(self).broken:
            raise RuntimeError(
                "Library cublas64_12.dll is not found or cannot be loaded"
            )
        return iter(()), None


@pytest.fixture()
def config(tmp_path):
    config = Config(tmp_path / "config.json")
    config.set("language", "en")
    return config


@pytest.fixture()
def fake_runtime(monkeypatch):
    """Serve a fake WhisperModel and a GPU, so no real model or DLL is touched."""
    import faster_whisper

    FlakyRuntimeModel.broken = set()
    FlakyRuntimeModel.built = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FlakyRuntimeModel)
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 1)
    return FlakyRuntimeModel


def test_a_model_that_cannot_encode_is_rejected_at_load(config, fake_runtime):
    fake_runtime.broken = {V3_MODEL}
    engine = Transcriber(None, config)
    engine.load()
    assert engine.engine_name == SMALL_MODEL
    assert engine._beam == 1
    assert fake_runtime.built == [V3_MODEL, SMALL_MODEL]


def test_negative_control_a_working_engine_is_kept(config, fake_runtime):
    """Proves the probe discriminates: a healthy engine is not replaced."""
    engine = Transcriber(None, config)
    engine.load()
    assert engine.engine_name == V3_MODEL
    assert fake_runtime.built == [V3_MODEL]


def test_a_failed_transcription_drops_the_model_for_a_clean_reload(
    config, fake_runtime
):
    engine = Transcriber(None, config)
    engine.load()
    fake_runtime.broken = {V3_MODEL}
    audio = np.full(SAMPLE_RATE, 2e-3, dtype=np.float32)
    with pytest.raises(RuntimeError):
        engine.transcribe_audio(audio)
    assert engine._model is None
    assert engine._model_key is None
