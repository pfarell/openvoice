"""Hands-free accuracy and latency benchmark for real dictation conditions.

No recording is needed. Cached LibriSpeech clips (``lang_bench.py --fetch
--languages en``) are turned into the two kinds of utterance the app actually
sees: short single sentences and long 40-70 s dictations made of several
sentences joined by natural pauses. Each utterance is rendered twice at the
laptop array's native 44.1 kHz: ``clean`` (healthy level, no noise) and
``laptop`` (quiet level, background noise, mains hum, DC offset, a wake-gate
attenuated onset and a hotkey click) and ``far`` (talking from 50-60 cm: 8 dB
quieter than ``laptop``, room reverb, and the quiet room the saved dictations
show, about 18 dB below the speech). Both go through the app's real
conditioning chain (:func:`audio.condition_capture`) and recogniser
(:meth:`transcribe.Transcriber.transcribe_audio`) with the user's own language
and vocabulary settings, read from their config without writing it.

Every random choice comes from one seeded generator, so two runs on different
code see identical audio and their JSON reports are directly comparable.

Input size: ~30 clips, ~40 utterances x 2 conditions. Time is dominated by the
recogniser; scoring is O(r * h) per utterance in reference and hypothesis words.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from lang_bench import sample_dir, score  # noqa: E402

CAPTURE_RATE = 44100
RUNS_DIR = TOOLS / "runs"
FAR_SPEECH_DBFS = -52.0
FAR_NOISE_DBFS = -70.0
FAR_RT60 = 0.45
FAR_DRR_DB = 0.0
CONDITIONS = ("clean", "laptop", "far")
COPIED_SETTINGS = ("language", "custom_vocabulary", "initial_prompt", "model_name", "beam_size")


def load_clips() -> list[tuple[np.ndarray, str]]:
    from faster_whisper.audio import decode_audio

    clips = []
    for wav in sorted(sample_dir("en").glob("*.wav")):
        reference = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        audio = np.asarray(decode_audio(str(wav), sampling_rate=16000), dtype=np.float32)
        clips.append((audio, reference))
    return clips


def build_utterances(clips, rng, long_count: int) -> list[dict]:
    """Short singles plus ``long_count`` 40-70 s multi-sentence dictations."""
    items = [
        {"kind": "short", "parts": [audio], "gaps": [], "reference": ref}
        for audio, ref in clips
        if audio.size / 16000 <= 25.0
    ]
    for _ in range(long_count):
        target = rng.uniform(40.0, 70.0)
        order = rng.permutation(len(clips))
        parts, gaps, refs, total = [], [], [], 0.0
        for index in order:
            audio, ref = clips[int(index)]
            if parts:
                gap = rng.uniform(0.3, 0.8)
                gaps.append(gap)
                total += gap
            parts.append(audio)
            refs.append(ref)
            total += audio.size / 16000
            if total >= target:
                break
        items.append({"kind": "long", "parts": parts, "gaps": gaps, "reference": " ".join(refs)})
    return items


def room_response(rng, rt60: float = FAR_RT60, drr_db: float = FAR_DRR_DB) -> np.ndarray:
    """Direct path plus an exponentially decaying diffuse tail (60 dB down at ``rt60``)."""
    n = int(rt60 * CAPTURE_RATE)
    t = np.arange(n) / CAPTURE_RATE
    tail = rng.standard_normal(n) * np.exp(-np.log(1000.0) * t / rt60)
    tail[: int(0.003 * CAPTURE_RATE)] = 0.0
    tail *= 10 ** (-drr_db / 20) / np.sqrt(np.sum(tail**2))
    tail[0] = 1.0
    return tail


def render(item: dict, condition: str, rng) -> np.ndarray:
    """The 44.1 kHz capture the microphone would deliver for ``item``."""
    pieces = [np.zeros(int(0.4 * 16000), dtype=np.float32)]
    for index, part in enumerate(item["parts"]):
        if index:
            pieces.append(np.zeros(int(item["gaps"][index - 1] * 16000), dtype=np.float32))
        pieces.append(part)
    pieces.append(np.zeros(int(0.6 * 16000), dtype=np.float32))
    speech16 = np.concatenate(pieces).astype(np.float64)
    n_out = int(round(speech16.size * CAPTURE_RATE / 16000))
    t_out = np.arange(n_out) / CAPTURE_RATE
    speech = np.interp(t_out, np.arange(speech16.size) / 16000, speech16)
    active = speech[np.abs(speech) > 1e-4]
    rms = float(np.sqrt(np.mean(active**2))) if active.size else 1.0
    if condition == "clean":
        return (speech * (10 ** (-24 / 20) / rms)).astype(np.float32)
    level = 10 ** (-44 / 20)
    noise_db = -12.0
    if condition == "far":
        voiced = np.abs(speech) > 1e-4
        speech = fftconvolve(speech, room_response(rng))[:n_out]
        rms = float(np.sqrt(np.mean(speech[voiced] ** 2))) if voiced.any() else 1.0
        speech *= 10 ** (FAR_SPEECH_DBFS / 20) / rms
        noise_db = FAR_NOISE_DBFS + 44.0
    else:
        speech *= level / rms
    onset = int(0.4 * CAPTURE_RATE)
    gate = np.ones(n_out)
    gate[onset : onset + int(0.35 * CAPTURE_RATE)] = np.linspace(0.08, 1.0, int(0.35 * CAPTURE_RATE))
    speech *= gate
    spectrum = np.fft.rfft(rng.standard_normal(n_out))
    freqs = np.fft.rfftfreq(n_out, 1 / CAPTURE_RATE)
    spectrum /= np.sqrt(np.maximum(freqs, 20.0))
    noise = np.fft.irfft(spectrum, n_out)
    noise *= level * 10 ** (noise_db / 20) / np.sqrt(np.mean(noise**2))
    hum = 0.3 * level * 10 ** ((noise_db + 12.0) / 20) * np.sin(2 * np.pi * 50.0 * t_out)
    click = np.zeros(n_out)
    start = int(0.05 * CAPTURE_RATE)
    length = int(0.004 * CAPTURE_RATE)
    click[start : start + length] = 0.6 * np.exp(-np.arange(length) / (length / 5)) * rng.choice(
        [-1.0, 1.0], size=length
    )
    return np.clip(speech + noise + hum + click + 0.01, -1.0, 1.0).astype(np.float32)


def make_config(tmp: Path, model: str | None):
    from config import Config, default_config_path

    live = default_config_path()
    target = tmp / "config.json"
    if live.exists():
        shutil.copyfile(live, target)
    config = Config(target)
    live_values = {key: config.get(key) for key in COPIED_SETTINGS}
    defaults = Config(tmp / "defaults.json")
    for key, value in live_values.items():
        defaults.set_volatile(key, value)
    if model:
        defaults.set_volatile("model_name", model)
    return defaults


def run(label: str, long_count: int, seed: int, model: str | None, limit: int | None,
        conditions=CONDITIONS) -> int:
    import audio as audio_module
    import transcribe
    from bus import Bus

    clips = load_clips()
    if not clips:
        print("no cached English clips; run tools/lang_bench.py --fetch --languages en", file=sys.stderr)
        return 2
    rng = np.random.default_rng(seed)
    items = build_utterances(clips, rng, long_count)
    if limit:
        items = items[:limit]
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp), model)
        engine = transcribe.resolve_engine(config)
        print(f"engine {engine}  language={config.get('language')!r}  utterances={len(items)}")
        transcriber = transcribe.Transcriber(Bus(), config)
        transcriber.load()
        transcriber.transcribe_audio(np.zeros(16000, dtype=np.float32) + 1e-3)
        rows = []
        for condition in conditions:
            render_rng = np.random.default_rng(seed + 1)
            for number, item in enumerate(items):
                captured = render(item, condition, render_rng)
                started = time.perf_counter()
                processed = audio_module.condition_capture(captured, CAPTURE_RATE)
                conditioned = time.perf_counter()
                result = transcriber.transcribe_audio(processed)
                decoded = time.perf_counter()
                if result.accepted:
                    _, edits, words, _ = score(item["reference"], result.text, "en")
                else:
                    _, _, words, _ = score(item["reference"], "", "en")
                    edits = words
                rows.append(
                    {
                        "condition": condition,
                        "kind": item["kind"],
                        "index": number,
                        "seconds": round(captured.size / CAPTURE_RATE, 2),
                        "words": words,
                        "edits": edits,
                        "rejected": None if result.accepted else result.reject_reason,
                        "condition_s": round(conditioned - started, 3),
                        "decode_s": round(decoded - conditioned, 3),
                        "hypothesis": result.text,
                        "reference": item["reference"],
                    }
                )
                tag = "REJECTED " + str(result.reject_reason) if not result.accepted else ""
                print(
                    f"{condition:6} {item['kind']:5} {number:2d} {rows[-1]['seconds']:5.1f}s"
                    f"  WER={edits / max(words, 1) * 100:5.1f}%  cond={rows[-1]['condition_s']:.2f}s"
                    f"  dec={rows[-1]['decode_s']:.2f}s {tag}"
                )
    summary = summarise(rows)
    print()
    for key, value in summary.items():
        print(
            f"{key:14} WER={value['wer'] * 100:5.2f}%  ({value['edits']}/{value['words']})"
            f"  rejected={value['rejected']}  cond={value['condition_s']:.2f}s"
            f"  decode={value['decode_s']:.2f}s  RTF={value['rtf']:.3f}"
        )
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"long_bench_{label}.json"
    out.write_text(json.dumps({"seed": seed, "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def summarise(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        for key in (row["condition"], f"{row['condition']}/{row['kind']}", "all"):
            groups.setdefault(key, []).append(row)
    summary = {}
    for key in sorted(groups):
        group = groups[key]
        words = sum(r["words"] for r in group)
        edits = sum(r["edits"] for r in group)
        seconds = sum(r["seconds"] for r in group)
        summary[key] = {
            "wer": edits / max(words, 1),
            "edits": edits,
            "words": words,
            "rejected": sum(1 for r in group if r["rejected"]),
            "condition_s": sum(r["condition_s"] for r in group),
            "decode_s": sum(r["decode_s"] for r in group),
            "rtf": sum(r["condition_s"] + r["decode_s"] for r in group) / max(seconds, 1e-9),
        }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="run")
    parser.add_argument("--long", type=int, default=8, dest="long_count")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args(argv)
    return run(args.label, args.long_count, args.seed, args.model, args.limit, args.conditions)


if __name__ == "__main__":
    raise SystemExit(main())
