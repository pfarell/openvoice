"""Microphone capture, health measurement and silence trimming for OpenVoice.

``trim_silence`` is a pure function so the trimming policy can be tested
without a microphone. ``Recorder`` owns one sounddevice input stream, forwards
a throttled level to the bus, and never lets a callback exception kill the
stream. Only ``bus`` and ``config`` are imported, and only for typing, so this
module stays importable without Qt.

Spectral measurement policy
---------------------------
Band levels are measured in dB *relative to the 300-1000 Hz band* over the
loudest 30 % of frames. A share of total energy - ``energy in band / total
energy`` - must never be used here. A microphone's room rumble and any DC
offset dominate the total, so the share collapses toward zero for perfectly
healthy speech and reports a fault that does not exist. Measured on this
machine, a real dictation recording scored 0.2 % of its energy above 1 kHz
while its 2-4 kHz band sat at a normal -19 dB. A share-based verdict declared
that healthy microphone UNUSABLE and sent the user to change Windows settings
for a problem that was not there. Relative band levels and a roll-off check
cannot be fooled that way: rumble lives below the reference band.

No denoising
------------
Do not add speech enhancement anywhere in this path. MetricGAN+ raised word
error rate in all 40 configurations tested for Whisper, and Whisper's mel front
end is trained on unprocessed audio, so cleaning the signal first makes
recognition worse. The only processing allowed is the 80 Hz high-pass in
:func:`highpass_rumble`, which removes the DC and rumble the mel front end would
otherwise see, plus linear resampling.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from bus import Bus
    from config import Config

SAMPLE_RATE = 16000
FRAME_MS = 20.0
HOP_MS = 10.0
MIN_LEVEL_DB = -60.0
EMIT_INTERVAL_S = 0.016
ENERGY_FLOOR = 1e-9
#: Seconds of always-on audio kept ahead of a recording. The Intel array's first
#: words arrive late (see ``Recorder``); buffering the moments before the keypress
#: recovers the lead-in and the hotkey-to-capture latency.
PREROLL_SECONDS = 1.5

BAND_EDGES = (
    (80, 300),
    (300, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
)
REFERENCE_BAND = "300-1000"
TILT_BAND = "2000-4000"
ANALYSIS_FRAME_MS = 25.0
ANALYSIS_HOP_MS = 12.5
SPEECH_FRAME_FRACTION = 0.30
CLIP_LEVEL = 0.99
NO_AUDIO_PEAK_DB = -60.0
MIN_AUDIO_SECONDS = 0.3
GOOD_SNR_DB = 15.0
MARGINAL_SNR_DB = 10.0
TILT_LIMIT_DB = -30.0
TILT_NORMAL_DB = -25.0
CLIP_FRACTION_LIMIT = 0.001
HIGHPASS_HZ = 80.0
_TINY_POWER = 1e-30
_AMPLITUDE_FLOOR = 1e-12

try:
    from scipy.signal import lfilter as _scipy_lfilter
    from scipy.signal import resample_poly as _scipy_resample_poly
except Exception:
    _scipy_lfilter = None
    _scipy_resample_poly = None


def _amplitude_from_db(db: float) -> float:
    """Convert an amplitude dBFS value to a linear amplitude.

    Time: O(1). Space: O(1).
    """
    return float(10.0 ** (db / 20.0))


def adaptive_threshold(
    frame_dbs: np.ndarray,
    absolute_db: float = -40.0,
    margin_db: float = 10.0,
    min_gate_db: float = -60.0,
    min_spread_db: float = 6.0,
) -> float:
    """Derive a speech threshold from the clip's own noise floor.

    One absolute threshold cannot serve both a low-gain laptop array and a loud
    headset. Measured on this machine, the Intel microphone array sits around
    -76 dBFS RMS with peaks near -54 dBFS, so ordinary speech can land close to
    -40 dBFS and be discarded as silence.

    The 20th percentile of the frame levels estimates the noise floor; the
    threshold sits ``margin_db`` above it and is clamped to
    ``[min_gate_db, absolute_db]``. The clamp means this can only ever be MORE
    permissive than the conservative absolute default, never more aggressive, so
    it cannot start trimming speech the fixed threshold would have kept. A clip
    with too little level spread has no usable floor and keeps ``absolute_db``.

    Time: O(f) for f frames (percentile via partition). Space: O(f).
    """
    if frame_dbs.size == 0:
        return absolute_db
    floor = float(np.percentile(frame_dbs, 20))
    ceiling = float(np.percentile(frame_dbs, 95))
    if ceiling - floor < min_spread_db:
        return absolute_db
    return min(max(floor + margin_db, min_gate_db), absolute_db)


def _framed(
    audio: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a mono clip into overlapping frames and their dBFS levels.

    Returns the ``(frames, frame_dbfs)`` pair, where ``frames`` is shaped
    ``(f, frame)`` and empty when the clip is shorter than one frame. Each
    frame's level is ``20*log10(max(rms, 1e-12))``.

    Time: O(n) for the frame gather and the per-frame RMS.
    Space: O(n) for the frames and their levels.
    """
    n = audio.size
    frame = int(round(sample_rate * frame_ms / 1000.0))
    hop = int(round(sample_rate * hop_ms / 1000.0))
    if frame <= 0 or hop <= 0 or n < frame:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    starts = np.arange(0, n - frame + 1, hop)
    frames = audio[starts[:, None] + np.arange(frame)[None, :]]
    rms = np.sqrt(np.maximum(np.mean(frames * frames, axis=1), 0.0))
    dbs = 20.0 * np.log10(np.maximum(rms, _AMPLITUDE_FLOOR))
    return frames, dbs


def band_levels_db(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Band levels in dB relative to the 300-1000 Hz band.

    The clip is framed at 25 ms with a 50 % overlap; the loudest 30 % of frames
    are concatenated so only speech, not the pauses, sets the spectrum. That
    span is Hann-windowed and transformed, and the power in each of
    :data:`BAND_EDGES` is summed. Every band is reported relative to the
    300-1000 Hz band, which is therefore 0.0 dB.

    This relative form is deliberate: see the module docstring for why a share
    of total energy must never be used. Clips shorter than one frame, and
    digital silence, return every band at 0.0 without raising.

    Time: O(n log n) - O(f log f) to order the frames and O(m log m) for the FFT
    over the concatenated speech span, where n is the sample count and f the
    frame count. Space: O(n).
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    zeroed = {f"{low}-{high}": 0.0 for low, high in BAND_EDGES}
    frames, dbs = _framed(arr, sample_rate, ANALYSIS_FRAME_MS, ANALYSIS_HOP_MS)
    if frames.shape[0] == 0:
        return zeroed
    keep = max(1, int(round(SPEECH_FRAME_FRACTION * frames.shape[0])))
    loudest = np.sort(np.argsort(dbs)[-keep:])
    speech = frames[loudest].reshape(-1)
    if speech.size < 8:
        return zeroed
    windowed = speech * np.hanning(speech.size)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(speech.size, 1.0 / sample_rate)
    levels = {
        f"{low}-{high}": 10.0
        * np.log10(float(np.sum(power[(freqs >= low) & (freqs < high)])) + _TINY_POWER)
        for low, high in BAND_EDGES
    }
    reference = levels[REFERENCE_BAND]
    return {name: level - reference for name, level in levels.items()}


def measure_quality(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Measure microphone health and return a verdict with its reasons.

    The returned keys are ``snr_db`` (P90 minus P10 of the 20 ms frame levels),
    ``tilt_db`` (the 2-4 kHz band relative to 300-1000 Hz; about -15 to -25 dB is
    normal speech, not a fault), ``clipping_fraction``, ``peak_dbfs``,
    ``rms_dbfs``, ``duration_s``, ``verdict`` and ``reasons``.

    The verdict follows exactly these rules, in order:

    * ``no-audio`` when the clip is shorter than 0.3 s or peaks below -60 dBFS.
    * ``good`` when ``snr_db >= 15``, ``clipping_fraction < 0.001`` and
      ``tilt_db > -30``.
    * ``marginal`` when ``snr_db >= 10``.
    * ``unusable`` otherwise.

    ``reasons`` is a list of short strings that name every measured fault, so a
    caller can print why. Empty input is handled without dividing by zero: it
    reports ``no-audio`` with finite numbers.

    Time: O(n log n), dominated by :func:`band_levels_db`. Space: O(n).
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = arr.size
    duration = n / float(sample_rate) if sample_rate > 0 else 0.0
    peak = float(np.max(np.abs(arr))) if n else 0.0
    rms = float(np.sqrt(np.mean(np.square(arr)))) if n else 0.0
    peak_dbfs = 20.0 * math.log10(max(peak, _AMPLITUDE_FLOOR))
    rms_dbfs = 20.0 * math.log10(max(rms, _AMPLITUDE_FLOOR))
    clipping = float(np.mean(np.abs(arr) >= CLIP_LEVEL)) if n else 0.0
    _frames, frame_dbs = _framed(arr, sample_rate, FRAME_MS, HOP_MS)
    snr_db = float(np.percentile(frame_dbs, 90) - np.percentile(frame_dbs, 10)) if frame_dbs.size else 0.0
    tilt_db = float(band_levels_db(arr, sample_rate)[TILT_BAND])

    if duration < MIN_AUDIO_SECONDS:
        verdict = "no-audio"
        reasons = [f"clip is {duration:.2f} s long, under the {MIN_AUDIO_SECONDS:.1f} s minimum"]
    elif peak_dbfs < NO_AUDIO_PEAK_DB:
        verdict = "no-audio"
        reasons = [f"peak {peak_dbfs:.0f} dBFS is below the {NO_AUDIO_PEAK_DB:.0f} dBFS floor"]
    else:
        reasons = []
        if clipping >= CLIP_FRACTION_LIMIT:
            reasons.append(f"clipping on {clipping * 100.0:.2f}% of samples")
        if tilt_db <= TILT_LIMIT_DB:
            reasons.append(
                f"spectrum rolls off faster than {TILT_LIMIT_DB:.0f} dB at 2-4 kHz, "
                "which means band-limiting"
            )
        if snr_db >= GOOD_SNR_DB and clipping < CLIP_FRACTION_LIMIT and tilt_db > TILT_LIMIT_DB:
            verdict = "good"
            reasons.append("speech clears the noise floor with a natural spectral tilt")
        elif snr_db >= MARGINAL_SNR_DB:
            verdict = "marginal"
            if tilt_db <= TILT_NORMAL_DB and tilt_db > TILT_LIMIT_DB:
                reasons.append(
                    f"2-4 kHz sits at {tilt_db:.0f} dB, below the usual -15 to -25 dB, "
                    "so the top end is thin"
                )
            if not reasons:
                reasons.append(f"signal-to-noise is {snr_db:.0f} dB, usable but not clean")
        else:
            verdict = "unusable"
            reasons.append(f"signal-to-noise is only {snr_db:.0f} dB, so speech barely clears the noise")

    return {
        "snr_db": snr_db,
        "tilt_db": tilt_db,
        "clipping_fraction": clipping,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "duration_s": duration,
        "verdict": verdict,
        "reasons": reasons,
    }


def highpass_rumble(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, cutoff_hz: float = HIGHPASS_HZ
) -> np.ndarray:
    """Remove DC and room rumble with a 2nd-order Butterworth high-pass.

    The captured signal reaches Whisper's mel front end, which is trained on
    clean audio and would otherwise be handed the DC offset and the low rumble a
    laptop array picks up. The filter is a biquad (Q = 1/sqrt(2)) built from the
    standard bilinear coefficients and run in direct form I. DC is removed
    exactly, 50 Hz room rumble is attenuated about 9 dB, and the response is
    -3 dB at ``cutoff_hz``. ``scipy.signal.lfilter`` runs the same recurrence
    in C when scipy is importable; the Python loop is the fallback.

    Time: O(n) in the sample count - one arithmetic pass. Space: O(n) for the
    returned array.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = arr.size
    if n == 0 or sample_rate <= 2.0 * cutoff_hz:
        return arr.copy()
    omega = 2.0 * math.pi * cutoff_hz / sample_rate
    cos_w = math.cos(omega)
    alpha = math.sin(omega) / (2.0 * math.sqrt(0.5))
    b0 = (1.0 + cos_w) / 2.0
    b1 = -(1.0 + cos_w)
    b2 = (1.0 + cos_w) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    if _scipy_lfilter is not None:
        out = _scipy_lfilter([b0, b1, b2], [1.0, a1, a2], arr.astype(np.float64))
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
    out = np.empty(n, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        xi = float(arr[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi
        out[i] = yi
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _resample_poly(audio: np.ndarray, up: int, down: int) -> np.ndarray:
    """Band-limited rational resampling with a numpy polyphase filter bank.

    A windowed-sinc prototype low-pass with ``24 * max(up, down) + 1`` taps is
    decomposed into ``up`` phases, each output sample is the dot product of one
    input neighbourhood with its phase, and every ``down``-th output is kept.
    The prototype is scaled by ``up`` to undo the zero-stuffing gain loss, so a
    passband tone keeps its amplitude and everything above the target Nyquist is
    attenuated rather than aliased.

    Time: O(m * P), where m is the output length and P the taps per phase, i.e.
    O(m) for a fixed rational ratio. Space: O(m).
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0 or (up == 1 and down == 1):
        return x
    divisor = math.gcd(int(up), int(down))
    up //= divisor
    down //= divisor
    max_rate = max(up, down)
    taps = 2 * 12 * max_rate + 1
    cutoff = 1.0 / max_rate
    positions = np.arange(taps) - (taps - 1) / 2.0
    prototype = cutoff * np.sinc(cutoff * positions) * np.hamming(taps)
    prototype *= up
    phases = (taps + up - 1) // up
    n_out = (n * up + down - 1) // down
    out = np.zeros(n_out, dtype=np.float64)
    if n_out == 0:
        return out
    j = np.arange(n_out)
    grid = j * down
    phase = grid % up
    base = grid // up
    for k in range(phases):
        index = base - k
        hidx = phase + k * up
        ok = (index >= 0) & (index < n) & (hidx < taps)
        out[ok] += x[index[ok]] * prototype[hidx[ok]]
    return out


def resample_to(
    audio: np.ndarray, sample_rate: int, target_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Resample mono audio to ``target_rate`` as float32 in -1..1.

    PortAudio's default resampler is not band-limited, so asking a 48 kHz
    endpoint for 16 kHz can lose the top of the speech band. Capturing at the
    endpoint's native rate and resampling here keeps it. ``scipy.signal``'s
    ``resample_poly`` is used when scipy is importable, otherwise the numpy
    polyphase resampler above. Matching rates return a copy unchanged.

    Time: O(m * P) for the numpy path, O(m log m) for the scipy path. Space:
    O(m) for the returned array.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sample_rate <= 0 or sample_rate == target_rate:
        return arr.copy()
    divisor = math.gcd(int(sample_rate), int(target_rate))
    up = target_rate // divisor
    down = sample_rate // divisor
    if _scipy_resample_poly is not None:
        out = _scipy_resample_poly(arr, up, down)
    else:
        out = _resample_poly(arr, up, down)
    return np.clip(np.asarray(out, dtype=np.float32).reshape(-1), -1.0, 1.0)


def normalize_level(
    audio: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain_db: float = 30.0,
    peak_ceiling_dbfs: float = -1.0,
) -> np.ndarray:
    """Apply makeup gain so the recogniser sees a consistent signal level.

    Measured on this machine: the Intel array delivers speech at about -45 dBFS
    RMS. Whisper's front end is trained on far louder audio and drops the
    opening word at that level — "send the invoice…" was transcribed as "and the
    invoice…" until the clip was normalised, which recovered it exactly.

    The gain is the smaller of what the RMS target asks for, what the peak
    ceiling allows, and ``max_gain_db``. It can therefore attenuate a loud clip
    and will never amplify near-silence into the recogniser's face.

    Time: O(n). Space: O(n).
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if peak < 1e-6 or rms < 1e-7:
        return audio
    gain_db = min(
        target_dbfs - 20.0 * math.log10(rms),
        peak_ceiling_dbfs - 20.0 * math.log10(peak),
        max_gain_db,
    )
    gain = 10.0 ** (gain_db / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32, copy=False)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -40.0,
    pad_ms: float = 150.0,
    adaptive: bool = True,
) -> np.ndarray:
    """Trim leading and trailing silence from a mono float32 clip.

    The clip is framed at 20 ms with a 10 ms hop and each frame's RMS is
    measured in dBFS as ``20*log10(max(rms, 1e-9))``. The kept span runs from
    the first to the last frame above ``threshold_db``, extended by ``pad_ms``
    on both sides and clamped to the clip. Digital silence, a clip whose peak
    is below -60 dBFS, and a clip shorter than one 20 ms frame all return an
    empty float32 array; empty input returns empty without raising.

    With ``adaptive`` set (the default) the threshold is derived from the clip's
    own noise floor by :func:`adaptive_threshold`, which can only lower it; see
    that function for why a fixed level fails on a low-gain microphone.

    Time: O(n) in the sample count - one cumulative sum plus one vectorised
    pass over the frames, where frames = (n - frame) // hop + 1.
    Space: O(n) for the float64 prefix-sum and square arrays, plus the result.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if float(np.max(np.abs(arr))) < _amplitude_from_db(MIN_LEVEL_DB):
        return np.empty(0, dtype=np.float32)

    frame = int(round(sample_rate * FRAME_MS / 1000.0))
    hop = int(round(sample_rate * HOP_MS / 1000.0))
    if frame <= 0 or hop <= 0 or n < frame:
        return np.empty(0, dtype=np.float32)

    squares = np.square(arr, dtype=np.float64)
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(squares)))
    starts = np.arange(0, n - frame + 1, hop)
    mean_squares = (prefix[starts + frame] - prefix[starts]) / frame
    rms = np.sqrt(np.maximum(mean_squares, 0.0))
    dbs = 20.0 * np.log10(np.maximum(rms, ENERGY_FLOOR))

    loud = np.flatnonzero(dbs > (adaptive_threshold(dbs, threshold_db) if adaptive else threshold_db))
    if loud.size == 0:
        return np.empty(0, dtype=np.float32)

    pad = int(round(pad_ms * sample_rate / 1000.0))
    first = int(starts[loud[0]])
    last = int(starts[loud[-1]]) + frame
    start = max(0, first - pad)
    end = min(n, last + pad)
    if end <= start:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(arr[start:end], dtype=np.float32)


def salvage_quiet(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Recover a clip that adaptive trimming emptied but that still holds signal.

    The Intel Smart Sound array delivers the first word or two of an utterance
    quietly (its noise-suppression gate is still opening), so the adaptive
    threshold can decide every frame is silence and ``trim_silence`` returns
    empty - the dictation is then dropped as "no speech". When the raw clip's
    peak is at or above the true-silence floor, this re-trims with that fixed
    floor instead of the adaptive one, so quiet-but-real speech survives. A clip
    that is genuinely silent (peak below -60 dBFS) still returns empty, and
    Whisper's own no-speech gate rejects anything this lets through.

    Time: O(n). Space: O(n).
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)
    if float(np.max(np.abs(arr))) < _amplitude_from_db(MIN_LEVEL_DB):
        return np.empty(0, dtype=np.float32)
    return trim_silence(
        arr, sample_rate=sample_rate, threshold_db=MIN_LEVEL_DB, adaptive=False
    )


def condition_capture(
    captured: np.ndarray, capture_rate: int, adaptive: bool = True, normalize: bool = True
) -> np.ndarray:
    """Turn a raw capture into the 16 kHz clip the recogniser is given.

    The single conditioning chain shared by :meth:`Recorder.stop` and the
    benchmarks, so a benchmark measures exactly what a dictation goes through:
    rumble high-pass, resample, silence trim (quiet-clip salvage when the trim
    finds nothing), then level normalisation. Time and space O(n).
    """
    filtered = highpass_rumble(captured, capture_rate)
    resampled = resample_to(filtered, capture_rate, SAMPLE_RATE)
    trimmed = trim_silence(resampled, sample_rate=SAMPLE_RATE, adaptive=adaptive)
    if trimmed.size == 0:
        trimmed = salvage_quiet(resampled, sample_rate=SAMPLE_RATE)
    return normalize_level(trimmed) if normalize else trimmed


def dbfs_to_level(db: float, floor_db: float = MIN_LEVEL_DB) -> float:
    """Map a dBFS value onto 0.0..1.0 with clamping.

    ``floor_db`` maps to 0.0 and 0 dBFS maps to 1.0. Non-finite input and
    values outside the range clamp to 0.0 or 1.0.

    Time: O(1). Space: O(1).
    """
    if not math.isfinite(db):
        return 0.0
    if floor_db >= 0.0:
        return 1.0 if db >= 0.0 else 0.0
    level = (db - floor_db) / (0.0 - floor_db)
    return float(min(max(level, 0.0), 1.0))


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for every sounddevice device with input channels.

    Time: O(d) in the device count. Space: O(d) for the returned list.
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    found: list[tuple[int, str]] = []
    try:
        for index, device in enumerate(devices):
            if int(device["max_input_channels"]) > 0:
                found.append((index, str(device["name"])))
    except Exception:
        return found
    return found


class Recorder:
    """Push-to-talk microphone recorder over one sounddevice input stream.

    Each PortAudio callback appends one copied block. ``stop`` concatenates once,
    high-passes to drop DC and rumble, resamples to 16 kHz, then trims. The
    stream is opened at the device's native rate when ``capture_native_rate`` is
    true, because PortAudio's default resampler is not band-limited and asking a
    48 kHz endpoint for 16 kHz can lose the top of the speech band.

    The callback body is wrapped so a failure can never kill the stream, and any
    stream start/stop failure resets state and reports on the bus rather than
    leaving ``is_recording`` stuck true.

    Time: O(1) per callback, O(total samples) for stop plus filtering and
    trimming. Space: O(total samples) for the buffer and the returned array.
    """

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, bus: "Bus", config: "Config") -> None:
        self._bus = bus
        self._config = config
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._capture_rate = SAMPLE_RATE
        self._buffer: list[np.ndarray] = []
        self._samples = 0
        self._recording = False
        self._last_emit = 0.0
        self._preroll: deque[np.ndarray] = deque()
        self._preroll_samples = 0

    @property
    def is_recording(self) -> bool:
        """Whether a stream is currently open and buffering."""
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        """Seconds buffered so far, or 0.0 when idle."""
        if not self._recording:
            return 0.0
        with self._lock:
            return self._samples / float(self._capture_rate)

    def prime(self) -> None:
        """Open the stream and leave it open without buffering, when configured.

        The Intel Smart Sound array on this machine gates its output when no
        speech is present: idle capture sits near -84 dBFS and the first one to
        two seconds of an utterance arrive quietly (down to about -58 dBFS)
        before the gate opens to a normal -33 dBFS. Keeping the stream open
        removes the stream-open cost, but it does NOT remove that gate - an aged
        warm stream still ramps. The pre-roll buffer and
        :func:`salvage_quiet` exist to stop that quiet onset being dropped. Set
        ``keep_mic_warm`` to false to close the stream between utterances.
        """
        if self._config.get("keep_mic_warm"):
            self._ensure_stream()

    def start(self) -> None:
        """Begin buffering; opens the stream if it is not already open.

        The buffer is seeded with the pre-roll already captured, so the moments
        before the keypress - and the latency between keypress and this call -
        are not lost.
        """
        if self._recording:
            return
        with self._lock:
            self._buffer = list(self._preroll)
            self._samples = 0
            self._last_emit = 0.0
        self._ensure_stream()
        if self._stream is None:
            return
        with self._lock:
            self._recording = True

    def _resolve_capture_rate(self) -> int:
        """Pick the device's native rate when ``capture_native_rate`` is set.

        The key defaults to true and is read defensively because it may be absent
        from an older config file. A device whose reported rate is unusable, or a
        query failure, falls back to 16 kHz.

        Time: O(1). Space: O(1).
        """
        enabled = self._config.get("capture_native_rate")
        if enabled is None:
            enabled = True
        if not enabled:
            return SAMPLE_RATE
        device = self._config.get("mic_device")
        try:
            if device is None:
                device = sd.default.device[0]
            native = int(round(float(sd.query_devices(device)["default_samplerate"])))
        except Exception:
            return SAMPLE_RATE
        if native <= 2 * HIGHPASS_HZ:
            return SAMPLE_RATE
        return native

    def _ensure_stream(self) -> None:
        """Open the input stream exactly once; report failure on the bus."""
        if self._stream is not None:
            return
        self._capture_rate = self._resolve_capture_rate()
        try:
            stream = sd.InputStream(
                samplerate=self._capture_rate,
                channels=1,
                dtype="float32",
                device=self._config.get("mic_device"),
                callback=self._callback,
                blocksize=0,
            )
            self._stream = stream
            stream.start()
        except Exception as exc:
            self._fail(exc)

    def stop(self) -> np.ndarray:
        """Stop buffering and return the trimmed recording.

        Returns an empty float32 array when not recording, and resets state. The
        stream stays open when ``keep_mic_warm`` is set.
        """
        if not self._recording:
            return np.empty(0, dtype=np.float32)
        with self._lock:
            chunks = self._buffer
            self._buffer = []
            self._samples = 0
            self._recording = False
        if not self._config.get("keep_mic_warm"):
            self._close_stream()
        if not chunks:
            return np.empty(0, dtype=np.float32)
        return condition_capture(
            np.concatenate(chunks).reshape(-1),
            self._capture_rate,
            adaptive=bool(self._config.get("adaptive_vad")),
            normalize=bool(self._config.get("normalize_audio")),
        )

    def cancel(self) -> None:
        """Discard everything captured; resets state."""
        with self._lock:
            self._buffer = []
            self._samples = 0
            self._recording = False
        if not self._config.get("keep_mic_warm"):
            self._close_stream()

    def reopen(self) -> None:
        """Drop and reopen the stream so a changed microphone setting takes effect."""
        self.cancel()
        self._close_stream()
        self.prime()

    def _close_stream(self) -> None:
        """Stop and close the current stream, swallowing PortAudio errors."""
        stream = self._stream
        self._stream = None
        with self._lock:
            self._preroll.clear()
            self._preroll_samples = 0
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _fail(self, exc: BaseException) -> None:
        """Report a stream failure on the bus and reset to idle."""
        print(f"[audio] recorder failed: {exc}", file=sys.stderr)
        self._close_stream()
        with self._lock:
            self._buffer = []
            self._samples = 0
            self._recording = False
        try:
            self._bus.transcription_error.emit(str(exc))
        except Exception:
            pass

    def _append_preroll_locked(self, block: np.ndarray) -> None:
        """Keep the last ``PREROLL_SECONDS`` of audio; caller holds the lock."""
        self._preroll.append(block)
        self._preroll_samples += int(block.shape[0])
        limit = int(PREROLL_SECONDS * self._capture_rate)
        while self._preroll_samples > limit and len(self._preroll) > 1:
            self._preroll_samples -= int(self._preroll.popleft().shape[0])

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        """Buffer one block and emit a throttled level; never raises."""
        try:
            block = indata.copy()
            np.nan_to_num(block, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
            np.clip(block, -1.0, 1.0, out=block)
            recording = self._recording
            with self._lock:
                self._append_preroll_locked(block)
                if recording:
                    self._buffer.append(block)
                    self._samples += int(block.shape[0])
            if not recording:
                return
            rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
            level = dbfs_to_level(20.0 * math.log10(max(rms, ENERGY_FLOOR)))
            now = time.perf_counter()
            with self._lock:
                due = now - self._last_emit >= EMIT_INTERVAL_S
                if due:
                    self._last_emit = now
            if due:
                self._bus.audio_level.emit(level)
        except Exception:
            pass
