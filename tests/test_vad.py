"""Silence-trimming tests for ``audio.trim_silence``.

Every expectation is a behavioural claim about the trimming policy, not a
smoke check. The loud/quiet/DC fixtures, the in-test threshold mutation, and
``tests/negative_control_vad.py`` together make the length and energy
assertions able to fail. Synthetic signals are all built from
``np.random.default_rng(0)`` so they are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

import audio

SAMPLE_RATE = 16000
FRAME = 320
HOP = 160
PAD_MS = 150.0


def _rng() -> np.random.Generator:
    """Deterministic generator for every synthetic signal in this module."""
    return np.random.default_rng(0)


def _sine(seconds: float, freq: float = 220.0, dbfs: float = -20.0) -> np.ndarray:
    """A mono float32 sine whose RMS is ``dbfs`` relative to full scale."""
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    amplitude = 10.0 ** (dbfs / 20.0) * np.sqrt(2.0)
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds: float, amplitude: float = 1e-4) -> np.ndarray:
    """A low-level noise floor well below -40 dBFS but above -120 dBFS."""
    n = int(round(seconds * SAMPLE_RATE))
    return _rng().uniform(-amplitude, amplitude, size=n).astype(np.float32)


def case2_signal() -> tuple[np.ndarray, int, float]:
    """Build case 2: 300 ms noise, 1 s -20 dBFS tone, 700 ms noise.

    Returns the concatenated signal, the tone sample count, and the tone's
    exact energy, so the mutation harness can reuse them unchanged.
    """
    tone = _sine(1.0)
    signal = np.concatenate([_noise(0.3), tone, _noise(0.7)]).astype(np.float32)
    tone_energy = float(np.sum(np.square(tone, dtype=np.float64)))
    return signal, tone.size, tone_energy


def case2_holds(result: np.ndarray, signal: np.ndarray, tone_n: int, tone_energy: float) -> bool:
    """Whether a trimmed result satisfies every case-2 requirement.

    True only when the result is shorter than the input, longer than the tone,
    and carries within 20% of the tone's energy. Shared verbatim with
    ``tests/negative_control_vad.py`` so the mutation harness exercises the
    exact predicate this module asserts.
    """
    if not isinstance(result, np.ndarray):
        return False
    if result.size >= signal.size:
        return False
    if result.size <= tone_n:
        return False
    if tone_energy <= 0.0:
        return False
    result_energy = float(np.sum(np.square(result, dtype=np.float64)))
    return abs(result_energy - tone_energy) / tone_energy <= 0.20


def test_digital_silence_returns_empty() -> None:
    """Two seconds of digital silence trim to nothing."""
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    result = audio.trim_silence(silence)
    assert result.size == 0
    assert result.dtype == np.float32


def test_speech_in_noise_is_trimmed_and_preserved() -> None:
    """The tone is kept, both noise margins are cut, its energy survives."""
    signal, tone_n, tone_energy = case2_signal()
    result = audio.trim_silence(
        signal, sample_rate=SAMPLE_RATE, threshold_db=-40.0, pad_ms=PAD_MS
    )
    assert case2_holds(result, signal, tone_n, tone_energy)
    assert result.flags["C_CONTIGUOUS"]


def test_all_tone_keeps_full_length() -> None:
    """A clip that is loud throughout is only padded, never shortened."""
    tone = _sine(2.0)
    result = audio.trim_silence(tone)
    assert result.dtype == np.float32
    assert result.size == pytest.approx(tone.size, abs=0.250 * SAMPLE_RATE)


def test_quiet_tail_is_removed() -> None:
    """A -55 dBFS tail after real speech is sliced away, not returned."""
    speech = _sine(1.0)
    quiet_start = speech.size
    signal = np.concatenate([speech, _sine(2.0, dbfs=-55.0)]).astype(np.float32)
    result = audio.trim_silence(signal)
    pad = int(round(PAD_MS * SAMPLE_RATE / 1000.0))
    assert result.size < signal.size
    assert result.size <= quiet_start + pad + FRAME


def test_dc_offset_is_treated_as_loud() -> None:
    """A constant 0.5 offset measures as loud, so nearly everything is kept."""
    dc = np.full(SAMPLE_RATE * 2, 0.5, dtype=np.float32)
    result = audio.trim_silence(dc)
    assert result.dtype == np.float32
    assert result.size >= dc.size - 2 * HOP


def test_empty_input_returns_empty_without_raising() -> None:
    """Empty in, empty contiguous float32 out, no exception."""
    result = audio.trim_silence(np.empty(0, dtype=np.float32))
    assert result.size == 0
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]


def test_negative_control_low_threshold_keeps_full_length() -> None:
    """The case-2 "shorter than the input" claim is falsifiable.

    Dropping ``threshold_db`` to -120.0 puts the roughly -85 dBFS noise floor
    above the threshold too, so ``trim_silence`` must return essentially the
    whole clip. If it could not, the case-2 length assertion would be vacuous
    rather than discriminating.
    """
    signal, _tone_n, _tone_energy = case2_signal()
    result = audio.trim_silence(signal, sample_rate=SAMPLE_RATE, threshold_db=-120.0)
    assert result.size >= signal.size - 2 * HOP
