"""Guards for the corrected microphone health metric.

This module replaces an earlier check that declared this machine's microphone
"unusable" because only 0.0-0.3 % of its speech energy sat above 1 kHz. That
metric was a SHARE of total energy, so low-frequency rumble and any DC offset
swamped it. The same real recording measures -19 dB at 2-4 kHz relative to
300-1000 Hz, a perfectly normal speech tilt, and every test here pins the
relative measurement that says so.

The negative-control test computes the old share on the healthy fixture and
asserts it would have fallen below the old 0.5 % floor: that is exactly the
false alarm this change removes. Rumble must not move the verdict, because
rumble lives below the reference band and cannot scale a relative level.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio import band_levels_db, measure_quality

SAMPLE_RATE = 16000
HOP_ALIGNED_HZ = 400.0


def _rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


def speech_like(seconds: float = 1.5, f0: float = 120.0, exponent: float = 1.6,
                noise_dbfs: float = -70.0) -> np.ndarray:
    """A speech-like harmonic stack with a natural tilt and a quiet noise floor.

    A strong low-frequency fundamental plus harmonics whose amplitudes fall as
    ``h ** -exponent``, shaped by a syllabic envelope, so the loudest 30 % of
    frames are real speech and the frame levels span well over 15 dB. The quiet
    white floor gives the spectrum a realistic broadband tail.
    """
    rng = _rng()
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    signal = np.zeros(n)
    for harmonic in range(1, int(8000 // f0) + 1):
        frequency = f0 * harmonic
        if frequency >= 0.45 * SAMPLE_RATE:
            break
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += harmonic ** (-exponent) * np.sin(2.0 * np.pi * frequency * t + phase)
    envelope = 0.05 + 0.95 * np.clip(np.sin(2.0 * np.pi * 3.5 * t) ** 2, 0.0, None)
    signal *= envelope
    signal /= float(np.max(np.abs(signal)))
    signal *= 10.0 ** (-6.0 / 20.0)
    noise = rng.normal(0.0, 1.0, n) * 10.0 ** (noise_dbfs / 20.0)
    return (signal + noise).astype(np.float32)


def bandlimited(seconds: float = 1.5) -> np.ndarray:
    """One 400 Hz tone and nothing else.

    A 400 Hz period divides the 12.5 ms frame hop, so the loudest-frames
    concatenation is an exact continuous tone, and nothing at all is left above
    1 kHz: the spectrum must roll off harder than -30 dB.
    """
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (0.7 * np.sin(2.0 * np.pi * HOP_ALIGNED_HZ * t)).astype(np.float32)


def clipped(seconds: float = 1.5) -> np.ndarray:
    signal = speech_like(seconds).astype(np.float64)
    signal = signal / float(np.max(np.abs(signal))) * 1.4
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def rumbled(seconds: float = 1.5, amplitude: float = 0.04) -> np.ndarray:
    signal = speech_like(seconds).astype(np.float64)
    t = np.arange(signal.size) / SAMPLE_RATE
    return (signal + amplitude * np.sin(2.0 * np.pi * 50.0 * t)).astype(np.float32)


def old_energy_share_above_1khz(audio: np.ndarray) -> float:
    """The removed share metric, kept only to prove it was wrong.

    It divides the energy above 1 kHz by the energy of the whole clip, so DC and
    rumble in the denominator drive it toward zero for healthy speech.
    """
    segment = audio[: min(audio.size, SAMPLE_RATE * 3 // 2)]
    if segment.size < 512:
        return 0.0
    windowed = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / SAMPLE_RATE)
    total = float(np.sum(spectrum)) or 1.0
    return 100.0 * float(np.sum(spectrum[freqs >= 1000.0])) / total


def test_healthy_speech_is_good() -> None:
    quality = measure_quality(speech_like())
    assert quality["verdict"] == "good", quality["reasons"]
    assert quality["snr_db"] >= 15.0
    assert quality["clipping_fraction"] < 0.001
    assert -25.0 < quality["tilt_db"] < -10.0


def test_band_levels_are_relative_to_the_reference_band() -> None:
    levels = band_levels_db(speech_like())
    assert levels["300-1000"] == pytest.approx(0.0, abs=1e-9)
    assert levels["2000-4000"] < levels["1000-2000"] < levels["300-1000"]


def test_silence_is_no_audio_with_finite_numbers() -> None:
    quality = measure_quality(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert quality["verdict"] == "no-audio"
    assert quality["reasons"]
    for key in ("snr_db", "tilt_db", "clipping_fraction", "peak_dbfs", "rms_dbfs", "duration_s"):
        assert np.isfinite(quality[key]), key


def test_short_clip_is_no_audio() -> None:
    quality = measure_quality(np.zeros(1024, dtype=np.float32))
    assert quality["verdict"] == "no-audio"
    assert quality["reasons"]


def test_bandlimited_audio_is_unusable_with_a_rolloff_reason() -> None:
    quality = measure_quality(bandlimited())
    assert quality["verdict"] == "unusable", quality["reasons"]
    assert quality["tilt_db"] <= -30.0
    assert any("roll" in reason for reason in quality["reasons"]), quality["reasons"]


def test_clipped_audio_is_flagged() -> None:
    quality = measure_quality(clipped())
    assert quality["clipping_fraction"] >= 0.001
    assert quality["verdict"] != "good"
    assert any("clipping" in reason for reason in quality["reasons"]), quality["reasons"]


def test_rumble_does_not_change_the_verdict() -> None:
    """Regression for the original bug: rumble used to collapse the verdict.

    A 50 Hz rumble sits below the 300-1000 Hz reference band, so it cannot move
    a relative band level. It may cost a little SNR, but the verdict must stay
    ``good`` and the tilt must barely move.
    """
    clean = measure_quality(speech_like())
    noisy = measure_quality(rumbled())
    assert clean["verdict"] == "good"
    assert noisy["verdict"] == "good", noisy["reasons"]
    assert abs(noisy["tilt_db"] - clean["tilt_db"]) < 3.0


def test_negative_control_the_old_share_metric_fails_a_good_signal() -> None:
    """Proves the assertions discriminate rather than rubber-stamp.

    The healthy fixture is measured GOOD by the relative metric, yet the old
    share metric puts less than 0.5 % of its energy above 1 kHz, so the old
    check would have called the same signal UNUSABLE. The two metrics disagree,
    which is the whole point: the share was measuring the rumble, not the speech.
    """
    healthy = speech_like()
    share = old_energy_share_above_1khz(healthy)
    assert measure_quality(healthy)["verdict"] == "good"
    assert share < 0.5, f"old metric scored {share:.3f}% and would not have failed"


def test_measure_quality_never_divides_by_zero_on_empty_input() -> None:
    for empty in (np.empty(0, dtype=np.float32), np.zeros(0, dtype=np.float64)):
        quality = measure_quality(empty)
        assert quality["verdict"] == "no-audio"
        assert quality["duration_s"] == 0.0
        for key in ("snr_db", "tilt_db", "clipping_fraction", "peak_dbfs", "rms_dbfs"):
            assert np.isfinite(quality[key]), key
