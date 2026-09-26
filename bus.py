"""The single cross-module channel.

Every module talks over this bus; no module imports another module's internals.
One Bus instance is created in app.py and passed by constructor injection.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

STATES = ("idle", "recording", "transcribing", "error")


class Bus(QObject):
    state_changed = Signal(str)
    audio_level = Signal(float)
    recording_started = Signal()
    recording_stopped = Signal()
    transcription_done = Signal(str, str)
    transcription_error = Signal(str)
    transcription_rejected = Signal(str)
    config_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}; expected one of {STATES}")
        if state == self._state:
            return
        self._state = state
        self.state_changed.emit(state)
