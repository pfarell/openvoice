"""Global hotkey handling for OpenVoice.

Two layers, deliberately split so the decision logic can be tested with no
keyboard, no OS hooks and no Qt event loop:

* :class:`HotkeyStateMachine` - pure Python. Feed it stable key names and it
  returns ``"start"`` / ``"stop"`` / ``"cancel"`` / ``None``.
* :class:`HotkeyManager` - a :class:`~PySide6.QtCore.QObject` that owns the
  pynput listener, translates real key events into names, drives the state
  machine and emits a single Qt signal ``action(str)``.

The manager never touches the bus state; ``app.py`` owns bus transitions.

Complexity
----------
``HotkeyStateMachine`` keeps one ``set`` of currently-held key names plus three
strings. The held set is bounded by the number of physically held keys (a small
constant in practice). :meth:`HotkeyStateMachine.on_press` and
:meth:`HotkeyStateMachine.on_release` are O(1) time and O(1) amortised space per
event; total space is O(h) for ``h`` simultaneously held keys.

Selective suppression limitation
--------------------------------
Cancelling with the cancel key (Esc by default) must stop the recording *and*
not reach the focused app, otherwise Esc closes a dialog you are dictating into.
That is done in the Windows low-level hook via ``win32_event_filter``, and only
for the cancel key while a recording is active - every other key always passes
through untouched. Because pynput does not dispatch a suppressed event to
``on_press``/``on_release``, the filter itself triggers the cancel through the
same state machine before suppressing. If the platform is not Windows, or the
listener API rejects the hook option, the manager degrades to *no suppression*
with a single stderr note; cancel still works through the normal key path, it
just also reaches the focused app. This is a Windows-only feature by design.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QObject, Signal

# Windows low-level hook message ids (winuser.h).
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_SYSKEYDOWN = 0x0104
_WM_SYSKEYUP = 0x0105
_PRESS_MESSAGES = (_WM_KEYDOWN, _WM_SYSKEYDOWN)
_RELEASE_MESSAGES = (_WM_KEYUP, _WM_SYSKEYUP)

#: pynput ``Key`` names that need a stable, cross-platform canonical spelling.
#: Windows reports AltGr as a right-Alt variant, so the two collapse to one name.
#: pynput spells the left shift ``Key.shift`` (``Key.shift_l`` is the same member),
#: so it is normalised to the explicit ``"shift_l"`` the config uses.
_NORMALISE = {
    "alt_gr": "alt_r",
    "cmd": "win",
    "cmd_r": "win",
    "shift": "shift_l",
}

#: Config keys whose change means the state machine must be rebuilt.
_WATCHED_CONFIG_KEYS = frozenset({"mode", "hold_key", "toggle_key", "cancel_key", "*"})


def key_name_from_pynput(key) -> str | None:
    """Map a pynput key to a stable lowercase name.

    Characters pass through lowercased (``'a'``, ``'1'``, ``'/'``); special
    keys use their ``Key`` member name (``Key.f9`` -> ``"f9"``,
    ``Key.enter`` -> ``"enter"``). ``Key.alt_gr`` normalises to ``"alt_r"``
    (Windows reports AltGr as a right-Alt variant), ``Key.cmd`` /
    ``Key.cmd_r`` normalise to ``"win"``, and pynput's ``Key.shift`` (the same
    member as ``Key.shift_l``) normalises to ``"shift_l"``. Anything without a
    name returns ``None``. Never raises.

    This function is duck-typed - it never imports pynput - so it can be unit
    tested with lightweight stand-ins as well as real ``Key``/``KeyCode``.
    """
    try:
        char = getattr(key, "char", None)
        if isinstance(char, str) and len(char) == 1 and char.isprintable():
            return char.lower()

        name = getattr(key, "name", None)
        if isinstance(name, str) and name:
            name = name.lower()
            return _NORMALISE.get(name, name)
    except Exception:  # pragma: no cover - defensive: unnameable objects
        return None
    return None


class HotkeyStateMachine:
    """Pure decision core for hold/toggle/cancel recording control.

    No pynput, no Qt, no I/O. See the module docstring for complexity.

    The auto-repeat guard is load-bearing: Windows fires repeated press events
    for a held key. A press of a key already in :attr:`_held` returns ``None``.
    Without it, the second press of the hold key would be read as a stop and the
    recording would instantly cancel itself.
    """

    def __init__(
        self,
        mode: str = "both",
        hold_key: str = "alt_r",
        toggle_key: str = "f9",
        cancel_key: str = "esc",
    ) -> None:
        self._mode = mode
        self._hold_key = hold_key
        self._toggle_key = toggle_key
        self._cancel_key = cancel_key
        self._held: set[str] = set()
        self._recording = False
        self._source: str | None = None

    # -- read-only configuration, exposed for the manager ------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def hold_key(self) -> str:
        return self._hold_key

    @property
    def toggle_key(self) -> str:
        return self._toggle_key

    @property
    def cancel_key(self) -> str:
        return self._cancel_key

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def recording_source(self) -> str | None:
        """``"hold"`` or ``"toggle"`` when recording, else ``None``."""
        return self._source

    # -- decision core -----------------------------------------------------
    def on_press(self, name: str | None) -> str | None:
        """Handle a key press; return ``"start"``, ``"stop"``, ``"cancel"`` or ``None``."""
        if not name:
            return None
        if name in self._held:
            # Auto-repeat guard: the OS re-fires presses for a held key.
            return None
        self._held.add(name)

        if name == self._cancel_key:
            if self._recording:
                self._clear_recording()
                return "cancel"
            return None

        trigger = self._trigger_kind(name)
        if trigger is None:
            return None

        if self._recording:
            # Any start trigger while already recording stops it: predictable
            # and symmetric across hold/toggle.
            self._clear_recording()
            return "stop"

        self._recording = True
        self._source = trigger
        return "start"

    def on_release(self, name: str | None) -> str | None:
        """Handle a key release; return ``"stop"`` or ``None``."""
        if not name:
            return None
        self._held.discard(name)

        if name != self._hold_key:
            return None
        if self._mode not in ("hold", "both"):
            return None
        # A hold release only stops a recording that hold itself started. This
        # is what makes "toggle on, release hold" a no-op while "hold on, tap
        # toggle" stops.
        if self._recording and self._source == "hold":
            self._clear_recording()
            return "stop"
        return None

    def reset(self) -> None:
        """Clear all held-key and recording state."""
        self._held.clear()
        self._clear_recording()

    # -- internals ---------------------------------------------------------
    def _trigger_kind(self, name: str) -> str | None:
        if self._mode in ("hold", "both") and name == self._hold_key:
            return "hold"
        if self._mode in ("toggle", "both") and name == self._toggle_key:
            return "toggle"
        return None

    def _clear_recording(self) -> None:
        self._recording = False
        self._source = None


def _vk_for_name(name: str) -> int | None:
    """Best-effort virtual-key code for a canonical key name, or ``None``.

    Only used to match the cancel key inside the Windows hook, where the event
    carries a ``vkCode`` rather than a pynput key object.
    """
    if not name:
        return None
    try:
        from pynput.keyboard import Key
    except Exception:  # pragma: no cover - pynput missing / unsupported platform
        return None
    attr = {"win": "cmd"}.get(name, name)
    member = getattr(Key, attr, None)
    # In pynput 1.8 the member's ``.value`` is a ``KeyCode``, not an int; the
    # virtual-key code lives on ``.value.vk`` (e.g. Key.esc.value.vk == 27).
    value = getattr(member, "value", None)
    vk = getattr(value, "vk", None)
    if isinstance(vk, int):
        return vk & 0xFF
    if isinstance(value, int):
        return value & 0xFF
    if len(name) == 1:
        return ord(name.upper())
    return None


class HotkeyManager(QObject):
    """Owns the pynput keyboard listener and emits ``action`` for app.py.

    The signal carries ``"start"``, ``"stop"`` or ``"cancel"``. The manager
    never mutates the bus; only ``app.py`` does that in response to the signal.
    """

    action = Signal(str)

    def __init__(self, bus, config) -> None:
        super().__init__()
        self._bus = bus
        self._config = config
        self._listener = None
        self._suppress_note_printed = False
        self._state_machine = self._build_state_machine()
        self._cancel_vk = _vk_for_name(self._state_machine.cancel_key)
        bus.config_changed.connect(self._on_config_changed)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Create and start the listener. Idempotent while already running."""
        if self._listener is not None:
            return
        self._state_machine = self._build_state_machine()
        self._cancel_vk = _vk_for_name(self._state_machine.cancel_key)
        try:
            from pynput import keyboard
        except Exception as exc:  # pragma: no cover - pynput is a declared dep
            print(f"[hotkeys] pynput unavailable, hotkeys disabled: {exc}", file=sys.stderr)
            return

        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["win32_event_filter"] = self._win32_event_filter
        else:
            self._note_no_suppression(
                f"selective cancel suppression is Windows-only (platform={sys.platform!r})"
            )

        try:
            listener = keyboard.Listener(
                on_press=self._on_pynput_press,
                on_release=self._on_pynput_release,
                **kwargs,
            )
        except TypeError as exc:
            # The hook option is not accepted by this build: keep cancel working
            # through the normal path, just without suppression.
            self._note_no_suppression(f"listener rejected win32_event_filter: {exc}")
            try:
                listener = keyboard.Listener(
                    on_press=self._on_pynput_press,
                    on_release=self._on_pynput_release,
                )
            except Exception as exc2:
                print(f"[hotkeys] listener could not be created: {exc2}", file=sys.stderr)
                return
        except Exception as exc:
            print(f"[hotkeys] listener could not be created: {exc}", file=sys.stderr)
            return

        self._listener = listener
        try:
            listener.start()
        except Exception as exc:
            print(f"[hotkeys] listener could not start: {exc}", file=sys.stderr)
            self._listener = None

    def stop(self) -> None:
        """Stop the listener and join its thread so nothing is left behind."""
        listener = self._listener
        self._listener = None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[hotkeys] listener.stop() failed: {exc}", file=sys.stderr)
        try:
            listener.join(timeout=2.0)
        except Exception as exc:  # pragma: no cover - join re-raises stored errors
            print(f"[hotkeys] listener thread ended with: {exc}", file=sys.stderr)

    # -- config ------------------------------------------------------------
    def _build_state_machine(self) -> HotkeyStateMachine:
        return HotkeyStateMachine(
            mode=str(self._config.get("mode")),
            hold_key=str(self._config.get("hold_key")),
            toggle_key=str(self._config.get("toggle_key")),
            cancel_key=str(self._config.get("cancel_key")),
        )

    def _on_config_changed(self, key: str) -> None:
        if key not in _WATCHED_CONFIG_KEYS:
            return
        self._state_machine = self._build_state_machine()
        self._cancel_vk = _vk_for_name(self._state_machine.cancel_key)

    # -- pynput callbacks --------------------------------------------------
    def _on_pynput_press(self, key, injected: bool = False) -> None:
        self._dispatch(self._state_machine.on_press(key_name_from_pynput(key)))

    def _on_pynput_release(self, key, injected: bool = False) -> None:
        self._dispatch(self._state_machine.on_release(key_name_from_pynput(key)))

    def _dispatch(self, action: str | None) -> None:
        if action:
            self.action.emit(action)

    # -- Windows selective suppression ------------------------------------
    def _win32_event_filter(self, msg, data):
        """Suppress the cancel key while recording; let everything else pass.

        Must return ``True`` (or not return ``False``) for normal propagation.
        Suppression is requested by raising through ``suppress_event()``, so that
        call is deliberately *outside* the try/except: catching it would swallow
        the suppression. Because a suppressed event is never dispatched by
        pynput, the cancel action is driven here directly.
        """
        suppress = False
        try:
            vk = getattr(data, "vkCode", None)
            if (
                self._state_machine.is_recording
                and self._cancel_vk is not None
                and vk == self._cancel_vk
                and msg in _PRESS_MESSAGES
            ):
                self._dispatch(self._state_machine.on_press(self._state_machine.cancel_key))
                suppress = True
        except Exception as exc:
            if not self._suppress_note_printed:
                self._suppress_note_printed = True
                print(
                    f"[hotkeys] cancel suppression disabled after error: {exc}",
                    file=sys.stderr,
                )
            suppress = False

        if suppress:
            listener = self._listener
            if listener is not None:
                listener.suppress_event()  # raises; propagates out of the hook
        return True

    def _note_no_suppression(self, reason: str) -> None:
        if self._suppress_note_printed:
            return
        self._suppress_note_printed = True
        print(
            f"[hotkeys] {reason}; the cancel key will also reach the focused app",
            file=sys.stderr,
        )
