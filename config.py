"""JSON-backed settings for OpenVoice.

Input size: one settings dict, O(10) keys. Every operation is O(keys) time and
O(keys) space apart from the disk write, which is a single atomic replace.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

APP_DIR_NAME = "OpenVoice"
#: The name this app shipped under before the rebrand. An existing legacy
#: directory keeps being used so an upgrade never orphans the user's settings.
LEGACY_APP_DIR_NAME = "OpenVoice"
#: The name this app shipped under before the rebrand. An existing legacy
#: directory keeps being used so an upgrade never orphans the user's settings.
LEGACY_APP_DIR_NAME = "RockyDictate"

DEFAULTS: dict = {
    "mode": "both",
    "hold_key": "alt_r",
    "toggle_key": "f9",
    "cancel_key": "esc",
    "auto_enter": False,
    "auto_enter_delay_ms": 120,
    "trailing_space": False,
    "model_name": "auto",
    "compute_type": "auto",
    "device": "auto",
    "beam_size": 5,
    "vad_filter": False,
    "spoken_punctuation": True,
    "learn_threshold": 2,
    "condition_on_previous": False,
    "adaptive_vad": True,
    "keep_mic_warm": True,
    "normalize_audio": True,
    "mic_device": None,
    "custom_vocabulary": "",
    "save_history": True,
    # 20 was too few: a benchmark run needs its recordings to still exist
    # afterwards, and a second run silently pruned the first one's audio, which
    # made the two runs impossible to compare.
    "history_limit": 200,
    "mirror_input": False,
    "pill_enabled": True,
    "run_at_login": False,
    "max_record_seconds": 300,
    "language": "auto",
    "detect_per_segment": False,
}


def default_config_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    current = Path(base) / APP_DIR_NAME / "config.json"
    legacy = Path(base) / LEGACY_APP_DIR_NAME / "config.json"
    if legacy.exists() and not current.exists():
        return legacy
    return current


class Config:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_config_path()
        self._data: dict = dict(DEFAULTS)
        self._listeners: list = []
        existed = self.path.exists()
        self.load()
        if not existed:
            try:
                self.save()
            except Exception as exc:
                print(f"[config] could not write defaults: {exc}", file=sys.stderr)

    def on_change(self, fn) -> None:
        self._listeners.append(fn)

    def _notify(self, key: str) -> None:
        for fn in tuple(self._listeners):
            try:
                fn(key)
            except Exception as exc:
                print(f"[config] listener failed for {key!r}: {exc}", file=sys.stderr)

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._quarantine(exc)
            return
        if not isinstance(raw, dict):
            self._quarantine(ValueError(f"top level is {type(raw).__name__}, not object"))
            return
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in raw.items() if not k.startswith("_")})
        self._data = merged

    def _quarantine(self, exc: Exception) -> None:
        backup = self.path.with_name(f"{self.path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            self.path.replace(backup)
            note = f"moved to {backup.name}"
        except Exception as move_exc:
            note = f"could not back up: {move_exc}"
        print(f"[config] unreadable ({exc}); {note}; using defaults", file=sys.stderr)
        self._data = dict(DEFAULTS)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, key: str, default=None):
        if key in self._data:
            return self._data[key]
        if default is not None:
            return default
        return DEFAULTS.get(key)

    def set(self, key: str, value) -> None:
        if self._data.get(key) == value:
            return
        self._data[key] = value
        self.save()
        self._notify(key)

    def set_volatile(self, key: str, value) -> None:
        """Change a value for this process only, without touching the file.

        Tools that pin a setting for a deterministic run (the English benchmark,
        for example) must not rewrite the user's own configuration as a side
        effect. No save, no listeners; O(1) time and space.
        """
        self._data[key] = value

    def update(self, values: dict) -> None:
        changed = [k for k, v in values.items() if self._data.get(k) != v]
        if not changed:
            return
        self._data.update(values)
        self.save()
        for key in changed:
            self._notify(key)

    def reset(self) -> None:
        self._data = dict(DEFAULTS)
        self.save()
        self._notify("*")

    def as_dict(self) -> dict:
        return dict(self._data)
