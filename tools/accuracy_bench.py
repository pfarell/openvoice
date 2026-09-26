"""Rank recogniser configurations on the thing that actually fails: proper nouns.

Synthetic SAPI speech, hard project names, several configurations, scored by how
many names survive. Absolute numbers will not match a real voice; the point is to
rank configurations against each other on a repeatable task.

    python tools/accuracy_bench.py
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

import transcribe  # noqa: E402

TERMS_SENTENCES = [
    ("Vint", "Open Vint and check the budget screen."),
    ("Kestrel", "The Kestrel app remembers your books."),
    ("Quill", "Quill is the assistant on this machine."),
    ("Zenor", "Zenor handles the memory layer."),
    ("NovaGate", "Open NovaGate and check the dashboard."),
    ("Zephyr", "Send the report to Zephyr when it is ready."),
    ("CadQuery", "I rebuilt the bracket in CadQuery this morning."),
    ("Luau", "The Luau script handles the shop interface."),
    ("opencode", "Start opencode and load the project."),
]

HOTWORDS = (
    "Vint, Kestrel, Quill, Zenor, NovaGate, Zephyr, opencode, CadQuery, "
    "Luau, WisprFlow, FreeCAD, KiCad, Wokwi, ESP32, Ollama, Supabase, Vercel, "
    "Roblox, Minecraft"
)

CONFIGS = [
    ("small.en cpu int8 beam1", "Systran/faster-whisper-small.en", "cpu", "int8", 1),
    ("small.en cuda fp16 beam1", "Systran/faster-whisper-small.en", "cuda", "float16", 1),
    ("turbo    cuda fp16 beam1", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "cuda", "float16", 1),
    ("turbo    cuda fp16 beam5", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "cuda", "float16", 5),
    ("turbo    cpu  int8 beam1", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "cpu", "int8", 1),
]


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


def term_ok(term: str, text: str, threshold: float = 0.8) -> bool:
    if term.lower() in text.lower():
        return True
    best = max(
        (similarity(term, w) for w in text.replace(",", " ").replace(".", " ").split()),
        default=0.0,
    )
    return best >= threshold


def main() -> int:
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    transcribe.enable_cuda_dlls()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clips = []
        for term, sentence in TERMS_SENTENCES:
            wav = tmp_path / f"{term}.wav"
            sapi(sentence, wav)
            clips.append((term, decode_audio(str(wav), sampling_rate=16000)))

        print(f"{len(clips)} sentences, one hard proper noun each")
        print(f"hotwords: {'yes' if HOTWORDS else 'no'} ({len(HOTWORDS.split(','))} terms)")
        print()

        rows = []
        for label, repo, device, compute, beam in CONFIGS:
            try:
                model = WhisperModel(
                    repo, device=device, compute_type=compute, local_files_only=True
                )
            except Exception as exc:
                print(f"{label:<26} LOAD FAILED: {str(exc)[:70]}")
                continue

            hits = 0
            misses = []
            audio_seconds = 0.0
            decode_seconds = 0.0
            for term, audio in clips:
                start = time.perf_counter()
                segments, _ = model.transcribe(
                    audio,
                    language="en",
                    beam_size=beam,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    hotwords=HOTWORDS,
                )
                text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
                decode_seconds += time.perf_counter() - start
                audio_seconds += audio.size / 16000
                if term_ok(term, text):
                    hits += 1
                else:
                    misses.append(f"{term}->{text!r}")

            rtf = decode_seconds / audio_seconds if audio_seconds else 0.0
            rows.append((label, hits, len(clips), rtf))
            print(f"{label:<26} {hits}/{len(clips)} names  RTF {rtf:5.2f}")
            for miss in misses:
                print(f"      {miss}")

        print()
        print("ranking by names correct, then by speed")
        for label, hits, total, rtf in sorted(rows, key=lambda r: (-r[1], r[3])):
            print(f"  {label:<26} {hits}/{total}  RTF {rtf:.2f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
