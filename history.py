"""Keep the last few dictations, audio included, so a wrong transcript can be
diagnosed from the real recording instead of guessed at.

Each entry is a WAV of the exact audio that was sent to the recogniser plus a
text file with what came back, so the two can be compared directly. When the
caller supplies metadata (the detected language, the engine that produced the
text) it is written beside them as a JSON sidecar with a creation timestamp.
Entries are pruned to ``history_limit``, newest kept; a pruned entry takes its
text and metadata files with it.

Cost: O(limit) on prune, once per dictation; the write itself is O(samples).
"""

from __future__ import annotations

import json
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

DIR_NAME = "history"
SAMPLE_RATE = 16000


def history_dir(config) -> Path:
    return Path(config.path).parent / DIR_NAME


def save(audio: np.ndarray, text: str, config, meta: dict | None = None) -> Path | None:
    """Store one dictation. Returns the WAV path, or None when storing is off
    or the entry could not be written.

    ``meta`` is optional. When it is a non-empty dict it is written next to the
    WAV as ``<stamp>.json`` with a ``"created"`` ISO timestamp added. A metadata
    write failure is reported and never loses the WAV.
    """
    if not config.get("save_history"):
        return None
    if audio is None or audio.size == 0:
        return None

    directory = history_dir(config)
    try:
        directory.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = 0
        while (directory / f"{stamp}.wav").exists():
            suffix += 1
            stamp = f"{stamp}_{suffix}"
        wav_path = directory / f"{stamp}.wav"

        samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes((samples * 32767).astype("<i2").tobytes())

        wav_path.with_suffix(".txt").write_text(text or "", encoding="utf-8")
        if meta:
            _write_meta(wav_path, meta)
        prune(config)
        return wav_path
    except Exception as exc:
        print(f"[history] could not store the dictation: {exc}", file=sys.stderr, flush=True)
        return None


def _write_meta(wav_path: Path, meta: dict) -> None:
    """Write the JSON sidecar for a stored dictation. Never raises.

    The sidecar only aids later diagnosis, so a failure here is reported to
    stderr and the WAV is left in place rather than being reported as lost.
    """
    payload = dict(meta)
    payload["created"] = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        wav_path.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(f"[history] could not write metadata sidecar: {exc}", file=sys.stderr)


def entries(config) -> list[Path]:
    directory = history_dir(config)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)


def prune(config) -> None:
    try:
        limit = int(config.get("history_limit") or 20)
    except Exception:
        limit = 20
    for stale in entries(config)[limit:]:
        for path in (stale, stale.with_suffix(".txt"), stale.with_suffix(".json")):
            try:
                path.unlink()
            except Exception:
                pass
