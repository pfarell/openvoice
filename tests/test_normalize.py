"""Level normalisation tests.

The behaviour under test is the one that fixed a real defect: this machine's
microphone delivers speech around -45 dBFS RMS, and at that level the recogniser
dropped the opening word of a sentence. Normalising recovered it.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio import normalize_level


def rms_db(audio: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 1e-12))


def peak_db(audio: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.max(np.abs(audio))), 1e-12))


def speech_like(rms_target_db: float = -45.0, seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(seconds * 16000)
    envelope = np.hanning(n)
    noise = rng.normal(0.0, 1.0, n) * envelope
    return (noise * (10 ** (rms_target_db / 20.0)) / max(float(np.std(noise)), 1e-12)).astype(
        np.float32
    )


def test_quiet_speech_is_raised_to_the_target():
    clip = speech_like(-45.0)
    assert rms_db(clip) < -40.0, "fixture must start quiet"
    out = normalize_level(clip)
    assert -21.5 < rms_db(out) < -18.5, f"expected about -20 dBFS, got {rms_db(out):.1f}"


def test_negative_control_un_normalised_clip_stays_quiet():
    """Proves the assertion above is capable of failing: without normalisation the
    same fixture is nowhere near the target."""
    clip = speech_like(-45.0)
    assert rms_db(clip) < -40.0
    assert not (-21.5 < rms_db(clip) < -18.5)


def test_loud_clip_is_not_amplified_past_the_peak_ceiling():
    clip = speech_like(-12.0)
    out = normalize_level(clip)
    assert peak_db(out) <= -1.0 + 0.01, f"peak {peak_db(out):.2f} dBFS exceeds the ceiling"
    assert rms_db(out) <= rms_db(clip) + 0.01, "a loud clip must not be made louder"


def test_silence_is_left_alone():
    silence = np.zeros(16000, dtype=np.float32)
    out = normalize_level(silence)
    assert out.size == silence.size
    assert np.all(out == 0.0), "silence must not be amplified into the recogniser"


def test_gain_is_capped():
    extremely_quiet = speech_like(-85.0)
    out = normalize_level(extremely_quiet, max_gain_db=30.0)
    gained = 20.0 * np.log10(
        max(float(np.sqrt(np.mean(np.square(out, dtype=np.float64)))), 1e-12)
    ) - rms_db(extremely_quiet)
    assert gained <= 30.0 + 0.01, f"gain {gained:.1f} dB exceeded the cap"


def test_output_is_finite_and_in_range():
    for level in (-45.0, -30.0, -12.0, -3.0):
        out = normalize_level(speech_like(level))
        assert np.all(np.isfinite(out))
        assert float(np.max(np.abs(out))) <= 1.0


def test_empty_input_returns_empty():
    out = normalize_level(np.empty(0, dtype=np.float32))
    assert out.size == 0
