"""Microphone health check and calibration for OpenVoice.

Records a few seconds from the selected input, reports the relative band levels,
the SNR, the spectral tilt and the clipping fraction, and says plainly whether
that microphone can drive dictation. Run it before deciding the app is broken.

    python tools/mic_check.py --seconds 4
    python tools/mic_check.py --device 1 --seconds 6
    python tools/mic_check.py --list

The band levels are measured relative to the 300-1000 Hz band on the loudest
30 % of frames. They are never a share of total energy: low-frequency rumble
swamps a share and makes a healthy microphone look broken. A 2-4 kHz band around
-20 dB relative to 300-1000 Hz is normal speech, not a fault.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audio  # noqa: E402

LEVEL_ADVICE = f"""
If the level is too low, it is almost always one of these, in this order:

  1. Windows might be blocking access
     Settings > Privacy & security > Microphone
     Turn on "Microphone access" and "Let desktop apps access your microphone".

  2. The input volume is turned down or muted
     Settings > System > Sound > Input > select the device > Volume
     Also right-click the speaker icon > Sound settings > More sound settings >
     Recording > your microphone > Properties > Levels (set to 100 %, unmute).

  3. A hardware or keyboard mute key is engaged
     Many laptops mute the microphone array with a dedicated key or an Fn
     combination, and show an LED or an on-screen mic-off icon.

  4. The wrong device is selected
     Run `python tools/mic_check.py --list`, then pin the right one in the tray
     menu under Microphone, or with `--device N`.

A 2-4 kHz level of roughly -20 dB relative to 300-1000 Hz is normal speech, not
a fault. Do not change any audio setting for a tilt in that range.
"""

ENHANCEMENT_ADVICE = """
This input is clipping or losing its top end, which is worth fixing:

  1. Windows 11
     Settings > System > Sound > Input > select the device
       - "Audio enhancements"  -> Off
       - "Voice Focus"         -> Off   (AI noise cancellation)

  2. Classic sound control panel
     Win+R -> mmsys.cpl -> Recording -> your microphone -> Properties
       - "Enhancements" tab -> Disable all
       - "Advanced" tab     -> uncheck "Enable audio enhancements"
       - "Levels" tab       -> Microphone 100 %, Microphone Boost 0 dB

Turn those off only because the measurement below shows clipping or a roll-off
worse than -30 dB, not because of a normal spectral tilt.
"""


def amplitude_db(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-12))


def advice_for(quality: dict) -> str:
    clipping = quality["clipping_fraction"] >= audio.CLIP_FRACTION_LIMIT
    rolled_off = quality["tilt_db"] <= audio.TILT_LIMIT_DB
    if clipping or rolled_off:
        return LEVEL_ADVICE + "\n" + ENHANCEMENT_ADVICE
    return LEVEL_ADVICE


def report(arr: np.ndarray, sample_rate: int = audio.SAMPLE_RATE) -> bool:
    quality = audio.measure_quality(arr, sample_rate)
    bands = audio.band_levels_db(arr, sample_rate)

    print(f"  duration            : {quality['duration_s']:.2f} s")
    print(f"  peak                : {quality['peak_dbfs']:7.1f} dBFS")
    print(f"  rms                 : {quality['rms_dbfs']:7.1f} dBFS")
    print("  band levels (dB rel. 300-1000 Hz):")
    for name, level in bands.items():
        print(f"      {name:>10} Hz : {level:7.1f} dB")
    print(f"  snr (P90-P10)       : {quality['snr_db']:7.1f} dB")
    print(
        f"  tilt (2-4 kHz)      : {quality['tilt_db']:7.1f} dB"
        "   (about -15 to -25 dB is normal speech)"
    )
    print(f"  clipping            : {quality['clipping_fraction'] * 100.0:7.3f} %")
    print()
    print(f"  VERDICT: {quality['verdict'].upper()}")
    for reason in quality["reasons"]:
        print(f"    - {reason}")
    print()

    if quality["verdict"] != "good":
        print(advice_for(quality))
    return quality["verdict"] == "good"


def capture(seconds: float, device) -> tuple[np.ndarray, int]:
    import sounddevice as sd

    rate = audio.SAMPLE_RATE
    try:
        native = sd.query_devices(device)["default_samplerate"] if device is not None else None
        if native is None:
            native = sd.query_devices(sd.default.device[0])["default_samplerate"]
        native = int(round(float(native)))
        if native > 2 * audio.HIGHPASS_HZ:
            rate = native
    except Exception:
        rate = audio.SAMPLE_RATE

    captured = sd.rec(
        int(seconds * rate),
        samplerate=rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    samples = captured.reshape(-1)
    filtered = audio.highpass_rumble(samples, rate)
    return audio.resample_to(filtered, rate, audio.SAMPLE_RATE), audio.SAMPLE_RATE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--file",
        default=None,
        help="analyse a saved WAV instead of recording (for testing this check itself)",
    )
    args = parser.parse_args()

    if args.file:
        from faster_whisper.audio import decode_audio

        print(f"analysing {args.file}")
        print()
        return 0 if report(decode_audio(args.file, sampling_rate=audio.SAMPLE_RATE)) else 1

    devices = audio.list_input_devices()

    if args.list:
        print("input devices:")
        for index, name in devices:
            default = ""
            try:
                import sounddevice as sd

                if index == sd.default.device[0]:
                    default = "  <- system default"
            except Exception:
                pass
            print(f"  {index:>3}  {name}{default}")
        return 0

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    import sounddevice as sd

    if device is None:
        try:
            device = sd.default.device[0]
        except Exception:
            device = None
    name = "system default"
    try:
        name = sd.query_devices(device)["name"]
    except Exception:
        pass

    print(f"recording {args.seconds:.1f} s from [{device}] {name}")
    print(">>> SPEAK NOW, in your normal dictation voice <<<")
    print()

    try:
        samples, sample_rate = capture(args.seconds, device)
    except Exception as exc:
        print(f"  could not open that device: {exc}")
        print()
        print("  Try `--list` and pick another index.")
        return 1

    return 0 if report(samples, sample_rate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
