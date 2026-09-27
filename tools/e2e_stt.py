"""End-to-end check: Windows SAPI speaks known sentences, the real pipeline
transcribes them, and word error rate is measured.

This is a discriminating check: a broken transcriber, a broken audio path or a
broken resampler all move the WER off zero. Windows SAPI is a local, free
component of Windows — no network, no API key.

Usage:
    python tools/e2e_stt.py [--beam 1] [--model small.en] [--keep]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bus import Bus  # noqa: E402
from config import Config  # noqa: E402
from transcribe import Transcriber  # noqa: E402

SENTENCES = [
    "Please schedule the team meeting for Thursday afternoon.",
    "The quick brown fox jumps over the lazy dog.",
    "I need to refactor the authentication module before the release.",
    "Add Zenor to the distribution list for this report.",
    "The measurement came back at twelve point five millimetres.",
    "Send the invoice to the accounting department tomorrow morning.",
    "Let us go over the design document one more time.",
]

DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
DOT_PLACEHOLDER = "QDOTMARK"

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "point": ".",
}

SPELLING = {
    "millimetres": "millimeters", "millimetre": "millimeter",
    "centimetres": "centimeters", "centimetre": "centimeter",
    "metres": "meters", "metre": "meter", "colour": "color",
    "litres": "liters", "litre": "liter", "grey": "gray",
}

ABBREVIATIONS = {
    "mm": "millimeters", "cm": "centimeters", "km": "kilometers",
    "kg": "kilograms", "ml": "milliliters",
}


def _is_number(token: str) -> bool:
    return token == "." or token.replace(".", "").isdigit()


def sapi_wav(text: str, path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{path}'); "
        f"$s.Speak('{text}'); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
    )


def normalise(text: str) -> list[str]:
    """Fold punctuation, number words, abbreviations and spelling variants so WER
    measures wording rather than formatting. These folds are declared, not hidden:
    the verbatim hypothesis is printed for every sentence and verbatim-exact
    sentences are counted separately."""
    text = text.lower().replace(DOT_PLACEHOLDER.lower(), "")
    text = DECIMAL_POINT.sub(DOT_PLACEHOLDER, text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace(DOT_PLACEHOLDER, ".")

    folded = [
        NUMBER_WORDS.get(w, SPELLING.get(w, ABBREVIATIONS.get(w, w)))
        for w in text.split()
    ]
    merged: list[str] = []
    for token in folded:
        if _is_number(token) and merged and _is_number(merged[-1]):
            merged[-1] += token
        else:
            merged.append(token)
    return [t for t in merged if t != "."]


def wer(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, O(r*h) time and O(h) space for r,h words."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        current = [i]
        for j, hw in enumerate(hyp, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (rw != hw),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--keep", action="store_true", help="keep the generated WAVs")
    parser.add_argument(
        "--vocab",
        default="",
        help='custom vocabulary fed as the whisper initial prompt, e.g. --vocab "Zenor"',
    )
    args = parser.parse_args()

    from faster_whisper.audio import decode_audio

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        config = Config(tmp_path / "config.json")
        config.set("language", "en")
        config.set("model_name", args.model)
        config.set("beam_size", args.beam)
        config.set("custom_vocabulary", args.vocab)
        bus = Bus()
        transcriber = Transcriber(bus, config)

        print(
            f"model={args.model} beam={args.beam} vocab={args.vocab!r} "
            f"sentences={len(SENTENCES)}"
        )
        print("-" * 78)

        load_start = time.perf_counter()
        transcriber.load()
        print(f"model load: {time.perf_counter() - load_start:.2f} s")

        scores: list[float] = []
        verbatim: list[int] = []
        audio_seconds = 0.0
        decode_seconds = 0.0

        for i, sentence in enumerate(SENTENCES, 1):
            wav = tmp_path / f"s{i:02d}.wav"
            sapi_wav(sentence, wav)
            audio = decode_audio(str(wav), sampling_rate=16000)
            seconds = float(len(audio)) / 16000.0

            start = time.perf_counter()
            text = transcriber.transcribe(audio)
            elapsed = time.perf_counter() - start

            score = wer(sentence, text)
            scores.append(score)
            if text.strip() == sentence.strip():
                verbatim.append(i)
            audio_seconds += seconds
            decode_seconds += elapsed

            print(f"[{i}] {seconds:4.1f}s audio in {elapsed:5.2f}s  WER {score * 100:5.1f}%")
            print(f"    ref: {sentence}")
            print(f"    got: {text or '<empty>'}")
            if args.keep:
                kept = ROOT / "tools" / "e2e_samples"
                kept.mkdir(parents=True, exist_ok=True)
                (kept / wav.name).write_bytes(wav.read_bytes())

        mean = sum(scores) / len(scores)
        exact = sum(1 for s in scores if s == 0.0)
        rtf = decode_seconds / audio_seconds if audio_seconds else 0.0
        print("-" * 78)
        print(f"mean WER (wording)  : {mean * 100:.2f}%")
        print(f"exact (wording)     : {exact}/{len(scores)}")
        print(
            f"exact (verbatim)    : {len(verbatim)}/{len(scores)}"
            f"   sentence numbers {verbatim}"
        )
        print(f"audio / decode      : {audio_seconds:.1f}s / {decode_seconds:.1f}s  (RTF {rtf:.2f})")
        print(f"VERDICT             : {'PASS' if exact == len(scores) else 'FAIL'}")
        return 0 if exact == len(scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
