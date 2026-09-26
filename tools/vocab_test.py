"""Measure which biasing method actually fixes proper nouns.

Windows SAPI speaks sentences containing this machine's real project names; the
real transcriber decodes them under several biasing configurations; the score is
how often the proper noun survives. Runs fully offline.

    python tools/vocab_test.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bus import Bus  # noqa: E402
from config import Config  # noqa: E402
from transcribe import Transcriber  # noqa: E402

TERMS = ["NovaGate", "CadQuery", "Luau", "opencode", "Zenor", "WisprFlow"]

SENTENCES = [
    ("NovaGate", "Open NovaGate and check the dashboard."),
    ("CadQuery", "I rebuilt the bracket in CadQuery this morning."),
    ("Luau", "The Luau script handles the shop interface."),
    ("opencode", "Start opencode and load the project."),
    ("Zenor", "Send the report to Zenor when it is ready."),
    ("WisprFlow", "WisprFlow uses a small floating indicator."),
]

VOCAB_PROMPT = ", ".join(TERMS)


def sapi(text: str, path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True
    )


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def term_present(term: str, text: str, threshold: float = 0.75) -> tuple[bool, str | None]:
    """True when the term (or something close enough to it) appears in the text."""
    if term.lower() in text.lower():
        return True, term
    words = text.replace(".", " ").replace(",", " ").split()
    best, best_score = None, 0.0
    for word in words:
        score = similarity(term, word)
        if score > best_score:
            best, best_score = word, score
    return (best_score >= threshold), best


def main() -> int:
    from faster_whisper.audio import decode_audio

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = Config(tmp_path / "config.json")
        config.set("beam_size", 1)
        bus = Bus()

        clips = []
        for term, sentence in SENTENCES:
            wav = tmp_path / f"{term}.wav"
            sapi(sentence, wav)
            clips.append((term, sentence, decode_audio(str(wav), sampling_rate=16000)))

        configurations = [
            ("baseline (no biasing)", {}),
            ("initial_prompt", {"initial_prompt": VOCAB_PROMPT}),
            ("hotwords", {"hotwords": VOCAB_PROMPT}),
            ("hotwords + initial_prompt", {"hotwords": VOCAB_PROMPT, "initial_prompt": VOCAB_PROMPT}),
        ]

        results: dict[str, list[tuple[str, str, bool]]] = {}
        for label, extra in configurations:
            transcriber = Transcriber(bus, config)
            transcriber.load()
            rows = []
            for term, sentence, audio in clips:
                segments, _ = transcriber._model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    **extra,
                )
                text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
                ok, heard = term_present(term, text)
                rows.append((term, text, ok))
            results[label] = rows
            hits = sum(1 for _, _, ok in rows if ok)
            print(f"{label:<26} {hits}/{len(rows)} terms correct")

        print()
        print("detail")
        print("-" * 92)
        for label, rows in results.items():
            print(f"\n{label}")
            for term, text, ok in rows:
                mark = "ok " if ok else "MISS"
                print(f"  [{mark}] want {term:<12} got {text!r}")

        best = max(results, key=lambda k: sum(1 for _, _, ok in results[k] if ok))
        print()
        print(f"best configuration: {best}")

        print("\nfuzzy-correction thresholds on what the best config heard")
        print("-" * 92)
        all_pairs = []
        for label, rows in results.items():
            for term, text, ok in rows:
                if ok and term.lower() in text.lower():
                    continue
                for word in text.replace(".", " ").replace(",", " ").split():
                    score = similarity(term, word)
                    if score > 0.5:
                        all_pairs.append((term, word, score))
        if all_pairs:
            for term, word, score in sorted(all_pairs, key=lambda p: -p[2]):
                print(f"  {term:<12} vs {word:<16} similarity {score:.3f}")
        else:
            print("  every term was already exact under at least one configuration")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
