"""Guards for long-dictation chunking and the fast conditioning path.

Whisper decodes in 30 s windows. Without timestamps the window step is fixed,
so a long dictation was cut mid-word and whole sentences were dropped (11.5%
WER on clean 40-70 s dictations in tools/long_bench.py, 3.6% once cut at
pauses). These tests pin the cut placement, the wiring into the decoder and
the scipy filter's equivalence with the pure-Python one. No model is loaded.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio
import transcribe
from config import Config
from transcribe import CHUNK_SECONDS, Transcriber, silence_cuts

SR = 16000


def _speech(seconds: float, pauses=(), seed: int = 0, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
    for start, end in pauses:
        span = int(end * SR) - int(start * SR)
        x[int(start * SR) : int(end * SR)] = noise * rng.standard_normal(span)
    return x


def _pieces(points):
    return list(zip(points[0::2], points[1::2]))


def test_short_clip_is_not_split():
    assert silence_cuts(_speech(CHUNK_SECONDS - 0.5)) is None
    assert silence_cuts(_speech(CHUNK_SECONDS)) is None


def test_long_clip_is_cut_inside_its_pauses():
    points = silence_cuts(_speech(65.0, pauses=[(19.8, 20.3), (42.8, 43.3)]))
    pieces = _pieces(points)
    assert pieces[0][0] == 0.0 and pieces[-1][1] == pytest.approx(65.0)
    for (_, end), (start, _) in zip(pieces, pieces[1:]):
        assert end == start
    cuts = [end for _, end in pieces[:-1]]
    assert len(cuts) == 2
    assert 19.8 <= cuts[0] <= 20.3
    assert 42.8 <= cuts[1] <= 43.3
    assert all(end - start <= CHUNK_SECONDS for start, end in pieces)


def test_pauseless_clip_gets_balanced_pieces_without_slivers():
    pieces = _pieces(silence_cuts(_speech(80.0)))
    lengths = [end - start for start, end in pieces]
    assert len(pieces) == 3
    assert max(lengths) <= CHUNK_SECONDS
    assert min(lengths) >= 20.0


def _words(count: int) -> str:
    return " ".join(f"w{index}" for index in range(count))


class _Segment:
    avg_logprob = -0.1
    no_speech_prob = 0.01
    compression_ratio = 1.0
    temperature = 0.0

    def __init__(self, text: str = "hello world", start: float = 0.0):
        self.text = text
        self.start = start


class _Info:
    language = "en"
    language_probability = 1.0
    all_language_probs = [("en", 1.0)]


class _Model:
    """One segment per clip, as faster-whisper gives without timestamps."""

    supported_languages = ["en"]

    def __init__(self, words=None, retry_words=None):
        self.kwargs = []
        self.words = words
        self.retry_words = retry_words

    def detect_language(self, audio=None, **_):
        return ("en", 1.0, [("en", 1.0)])

    def transcribe(self, audio, **kwargs):
        self.kwargs.append(kwargs)
        clips = kwargs.get("clip_timestamps")
        if not isinstance(clips, list):
            return iter([_Segment()]), _Info()
        starts = clips[0::2]
        if len(starts) == 1 and self.retry_words is not None:
            return iter([_Segment(_words(self.retry_words), starts[0])]), _Info()
        counts = self.words or [40] * len(starts)
        return iter([_Segment(_words(n), start) for n, start in zip(counts, starts)]), _Info()


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "_cuda_device_count", lambda: 0)
    config = Config(tmp_path / "config.json")
    config.set("language", "en")
    built = Transcriber(None, config)
    built._model = _Model()
    built._model_key = built._key()
    return built


def test_long_dictation_is_decoded_in_pause_aligned_clips(engine):
    engine.transcribe_audio(_speech(65.0, pauses=[(19.8, 20.3), (42.8, 43.3)]))
    clips = engine._model.kwargs[-1]["clip_timestamps"]
    assert isinstance(clips, list) and len(clips) == 6


def test_short_dictation_and_vad_mode_decode_the_whole_clip(engine):
    engine.transcribe_audio(_speech(5.0))
    assert engine._model.kwargs[-1]["clip_timestamps"] == "0"
    engine.config.set("vad_filter", True)
    engine.transcribe_audio(_speech(65.0, pauses=[(19.8, 20.3)]))
    assert engine._model.kwargs[-1]["clip_timestamps"] == "0"
    assert engine._model.kwargs[-1]["vad_filter"] is True


def test_truncated_pieces_flags_only_the_sparse_piece():
    clip = _speech(52.0)
    cuts = silence_cuts(clip)
    assert len(cuts) == 4
    assert transcribe.truncated_pieces(clip, cuts, [60, 30]) == [1]
    assert transcribe.truncated_pieces(clip, cuts, [60, 52]) == []
    assert transcribe.truncated_pieces(clip, cuts, [30, 60]) == [0]


def test_voiced_frames_separates_speech_from_pauses():
    clip = _speech(10.0, pauses=[(4.0, 6.0)], noise=0.001)
    mask = transcribe.voiced_frames(transcribe._smoothed_energy(clip))
    assert mask.size == 1000
    assert mask[:350].all() and mask[650:].all()
    assert not mask[420:580].any()
    assert transcribe.voiced_frames(transcribe._smoothed_energy(_speech(10.0))).all()


def test_truncated_piece_is_redecoded_alone_without_hotwords(engine):
    engine.config.set("custom_vocabulary", "TripwiseGO")
    engine._model = _Model(words=[60, 8], retry_words=55)
    result = engine.transcribe_audio(_speech(52.0))
    first, retry = engine._model.kwargs[-2:]
    pieces = _pieces(first["clip_timestamps"])
    assert first["hotwords"]
    assert retry["clip_timestamps"] == list(pieces[1])
    assert retry["hotwords"] is None
    assert len(result.text.split()) == 60 + 55


def test_retry_that_recovers_nothing_is_discarded(engine):
    engine._model = _Model(words=[60, 8], retry_words=9)
    result = engine.transcribe_audio(_speech(52.0))
    assert len(engine._model.kwargs) == 2
    assert len(result.text.split()) == 60 + 8


def test_even_pieces_are_decoded_once(engine):
    engine._model = _Model(words=[60, 58], retry_words=200)
    result = engine.transcribe_audio(_speech(52.0))
    assert len(engine._model.kwargs) == 1
    assert len(result.text.split()) == 118


@pytest.mark.skipif(audio._scipy_lfilter is None, reason="scipy not installed")
def test_scipy_highpass_matches_the_python_loop(monkeypatch):
    rng = np.random.default_rng(3)
    x = (0.2 * rng.standard_normal(44100) + 0.05).astype(np.float32)
    fast = audio.highpass_rumble(x, 44100)
    monkeypatch.setattr(audio, "_scipy_lfilter", None)
    slow = audio.highpass_rumble(x, 44100)
    assert np.max(np.abs(fast - slow)) < 1e-6


def test_condition_capture_resamples_and_removes_dc():
    t = np.arange(44100 * 2) / 44100
    captured = (0.1 * np.sin(2 * np.pi * 300 * t) + 0.05).astype(np.float32)
    out = audio.condition_capture(captured, 44100, adaptive=False, normalize=False)
    assert out.dtype == np.float32
    assert abs(out.size - 2 * SR) <= SR // 10
    assert abs(float(np.mean(out))) < 0.005
