"""Say something and see exactly what the pipeline heard. This is the check to run
with your own voice; the automated end-to-end test uses a synthetic voice and cannot
tell you how the app behaves on your speech, your room and your microphone.

    python tools/test_my_voice.py
    python tools/test_my_voice.py --seconds 6 --device 1 --insert
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bus import Bus  # noqa: E402
from config import Config  # noqa: E402
from transcribe import Transcriber  # noqa: E402


def db(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-12))


def tone(freq: int, ms: int, enabled: bool) -> None:
    """Audible cue, so the person speaking knows when the microphone is live
    without needing to watch a terminal they may not be looking at."""
    if not enabled:
        return
    try:
        import winsound

        winsound.Beep(freq, ms)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--insert", action="store_true", help="paste the text like the app does")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--no-beep", action="store_true", help="silence the audible cues")
    args = parser.parse_args()

    import sounddevice as sd

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if device is None:
        device = sd.default.device[0]
    try:
        name = sd.query_devices(device)["name"]
    except Exception:
        name = "system default"

    config = Config()
    if args.model:
        config.set("model_name", args.model)

    print(f"microphone: [{device}] {name}")
    print(f"model     : {config.get('model_name')} (beam {config.get('beam_size')})")
    print()
    print("  warming the microphone…", flush=True)
    try:
        stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", device=device, blocksize=0
        )
        stream.start()
    except Exception as exc:
        print(f"\ncould not open that device: {exc}")
        print("run `python tools/mic_check.py --list` and pass --device N")
        return 1

    beeps = not args.no_beep
    for remaining in range(int(max(args.countdown, 0)), 0, -1):
        print(f"  starting in {remaining}…", flush=True)
        tone(520, 80, beeps)
        time.sleep(1.0)
    for _ in range(2):
        tone(1000, 130, beeps)
        time.sleep(0.18)
    print("  >>> SPEAK NOW <<<", flush=True)

    started = time.perf_counter()
    chunks = []
    deadline = started + args.seconds
    try:
        while time.perf_counter() < deadline:
            data, _overflowed = stream.read(1600)
            chunks.append(data.copy())
    except Exception as exc:
        print(f"\ncapture failed: {exc}")
        return 1
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
    tone(380, 260, beeps)
    wall = time.perf_counter() - started
    captured = (
        np.concatenate(chunks).reshape(-1).copy()
        if chunks
        else np.empty(0, dtype=np.float32)
    )
    raw = captured

    samples_dir = ROOT / "tools" / "voice_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = samples_dir / f"voice-{stamp}.wav"
    try:
        import wave

        with wave.open(str(raw_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((np.clip(raw, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        print(f"  saved raw : {raw_path}")
    except Exception as exc:
        print(f"  could not save the raw capture: {exc}")

    nonfinite = int(np.count_nonzero(~np.isfinite(raw)))
    if nonfinite:
        print(f"\n  WARNING: the driver returned {nonfinite} corrupt samples "
              f"(non-finite). They were zeroed; this is the driver, not the app.")
    np.nan_to_num(raw, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(raw, -1.0, 1.0, out=raw)

    print(f"  captured  : {raw.size / 16000:.2f} s   peak {db(float(np.max(np.abs(raw)))):.1f} dBFS"
          f"   rms {db(float(np.sqrt(np.mean(raw ** 2)))):.1f} dBFS")

    import audio as audio_module

    bucket = 8000
    profile = []
    for start in range(0, raw.size, bucket):
        chunk = raw[start:start + bucket]
        if chunk.size == 0:
            continue
        profile.append(db(float(np.sqrt(np.mean(chunk ** 2)))))
    threshold = audio_module.adaptive_threshold(
        np.array(
            [
                db(float(np.sqrt(np.mean(raw[i:i + 320] ** 2))))
                for i in range(0, max(raw.size - 320, 1), 160)
            ]
        )
        if raw.size > 320
        else np.zeros(1)
    )
    print(f"  profile   : 0.5 s buckets, threshold {threshold:.1f} dBFS")
    for index, level in enumerate(profile):
        bar = "#" * int(max(level + 80.0, 0.0) / 2.0)
        mark = "  <-- speech" if level > threshold else ""
        print(f"    {index * 0.5:4.1f}s {level:7.1f} dBFS |{bar}{mark}")

    trimmed = audio_module.trim_silence(raw, adaptive=bool(config.get("adaptive_vad")))
    print(f"  after VAD : {trimmed.size / 16000:.2f} s")
    if trimmed.size == 0:
        print("\n  Nothing above the silence threshold — the recogniser got no audio.")
        print("  Run `python tools/mic_check.py --seconds 5` to see why.")
        return 1

    bus = Bus()
    transcriber = Transcriber(bus, config)
    print("\n  loading the model…", flush=True)
    load_start = time.perf_counter()
    transcriber.load()
    print(f"  model loaded in {time.perf_counter() - load_start:.2f} s")

    decode_start = time.perf_counter()
    text = transcriber.transcribe(trimmed)
    decode = time.perf_counter() - decode_start

    print()
    print(f"  heard     : {text or '<nothing>'}")
    print(f"  decode    : {decode:.2f} s for {trimmed.size / 16000:.2f} s audio "
          f"(RTF {decode / (trimmed.size / 16000):.2f}), wall clock {wall:.2f} s")

    if not text:
        print("\n  The audio was loud enough but the model produced no words.")
        return 1

    if args.insert:
        from injector import insert_text

        ok = insert_text(
            text,
            auto_enter=bool(config.get("auto_enter")),
            enter_delay_ms=int(config.get("auto_enter_delay_ms") or 120),
            trailing_space=bool(config.get("trailing_space")),
        )
        print(f"  inserted  : {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
