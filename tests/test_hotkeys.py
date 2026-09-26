"""Unit tests for the pure hotkey logic in ``hotkeys.py``.

Everything here runs with no keyboard, no OS hook and no Qt event loop: the
:class:`~hotkeys.HotkeyStateMachine` is driven with plain key-name strings.

The negative-control test at the bottom deliberately mutates the object under
test (it clears the held-key set between presses, which is exactly what removing
the auto-repeat guard would do) to prove the repeat-guard assertion can fail.
Without that, a green "three presses -> one start" test would be unfalsifiable.
"""

from __future__ import annotations

import pytest
from pynput.keyboard import Key, KeyCode

from hotkeys import HotkeyStateMachine, key_name_from_pynput


# --------------------------------------------------------------------------
# key_name_from_pynput
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key, expected",
    [
        (KeyCode.from_char("a"), "a"),
        (KeyCode.from_char("1"), "1"),
        (KeyCode.from_char("/"), "/"),
        (KeyCode.from_char("A"), "a"),
        (Key.f9, "f9"),
        (Key.esc, "esc"),
        (Key.enter, "enter"),
        (Key.space, "space"),
        (Key.alt_l, "alt_l"),
        (Key.alt_r, "alt_r"),
        (Key.alt_gr, "alt_r"),  # AltGr collapses onto right-Alt
        (Key.ctrl_l, "ctrl_l"),
        (Key.ctrl_r, "ctrl_r"),
        (Key.shift_l, "shift_l"),
        (Key.shift_r, "shift_r"),
        (Key.cmd, "win"),
        (Key.cmd_r, "win"),
        (Key.media_play_pause, "media_play_pause"),
    ],
)
def test_key_name_mapping(key, expected):
    assert key_name_from_pynput(key) == expected


@pytest.mark.parametrize("key", [None, object(), KeyCode(vk=255), ""])
def test_key_name_unnameable_is_none(key):
    assert key_name_from_pynput(key) is None


# --------------------------------------------------------------------------
# hold semantics
# --------------------------------------------------------------------------
def test_hold_start_on_press_stop_on_release():
    m = HotkeyStateMachine(mode="hold")
    assert m.on_press("alt_r") == "start"
    assert m.is_recording is True
    assert m.recording_source == "hold"
    assert m.on_release("alt_r") == "stop"
    assert m.is_recording is False
    assert m.recording_source is None


def test_three_rapid_repeated_presses_produce_one_start_and_no_stop():
    m = HotkeyStateMachine(mode="hold")
    actions = [m.on_press("alt_r") for _ in range(3)]
    assert actions == ["start", None, None]
    assert actions.count("start") == 1
    assert "stop" not in actions
    assert m.is_recording is True
    # A genuine release still stops it once.
    assert m.on_release("alt_r") == "stop"


# --------------------------------------------------------------------------
# toggle semantics
# --------------------------------------------------------------------------
def test_toggle_tap_on_then_off():
    m = HotkeyStateMachine(mode="toggle")
    assert m.on_press("f9") == "start"
    assert m.on_release("f9") is None
    assert m.is_recording is True
    assert m.on_press("f9") == "stop"
    assert m.on_release("f9") is None
    assert m.is_recording is False


def test_toggle_then_hold_release_does_nothing():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("f9") == "start"
    assert m.recording_source == "toggle"
    # Releasing the hold key must not stop a toggle-started recording.
    assert m.on_release("alt_r") is None
    assert m.is_recording is True
    assert m.recording_source == "toggle"


def test_hold_then_tap_toggle_stops():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("alt_r") == "start"
    assert m.recording_source == "hold"
    assert m.on_press("f9") == "stop"
    assert m.is_recording is False
    # The trailing hold release is now a no-op.
    assert m.on_release("alt_r") is None


def test_any_start_trigger_while_recording_stops():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("f9") == "start"
    assert m.on_press("alt_r") == "stop"  # hold press while toggle-recording
    assert m.on_press("alt_r") is None  # auto-repeat of the still-held alt_r
    assert m.on_release("alt_r") is None  # already stopped
    # A fresh toggle press starts again.
    assert m.on_release("f9") is None
    assert m.on_press("f9") == "start"


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------
def test_cancel_while_recording_clears_state():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("alt_r") == "start"
    assert m.on_press("esc") == "cancel"
    assert m.is_recording is False
    assert m.recording_source is None
    # Releasing esc is inert once cancelled.
    assert m.on_release("esc") is None


def test_cancel_while_idle_does_nothing():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("esc") is None
    assert m.on_release("esc") is None
    assert m.is_recording is False


# --------------------------------------------------------------------------
# mode filtering and unknown keys
# --------------------------------------------------------------------------
def test_hold_mode_ignores_toggle_key():
    m = HotkeyStateMachine(mode="hold")
    assert m.on_press("f9") is None
    assert m.on_release("f9") is None
    assert m.is_recording is False


def test_toggle_mode_ignores_hold_key():
    m = HotkeyStateMachine(mode="toggle")
    assert m.on_press("alt_r") is None
    assert m.on_release("alt_r") is None
    assert m.is_recording is False


def test_unknown_key_is_inert():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("nope") is None
    assert m.on_release("nope") is None
    assert m.on_press(None) is None
    assert m.on_release(None) is None
    assert m.is_recording is False


# --------------------------------------------------------------------------
# reset
# --------------------------------------------------------------------------
def test_reset_clears_state_and_held_keys():
    m = HotkeyStateMachine(mode="both")
    assert m.on_press("alt_r") == "start"
    m.reset()
    assert m.is_recording is False
    assert m.recording_source is None
    # Held set was cleared, so a fresh press starts again rather than being
    # mistaken for auto-repeat, and the stale release is inert.
    assert m.on_release("alt_r") is None
    assert m.on_press("alt_r") == "start"


# --------------------------------------------------------------------------
# negative control
# --------------------------------------------------------------------------
def test_repeat_guard_negative_control():
    """Prove the auto-repeat assertion is discriminating, not vacuous.

    We rebuild the scenario from ``test_three_rapid_repeated_presses_*`` but
    bypass the guard by emptying the machine's held-key set between presses,
    which is exactly the state a build *without* the guard would be in. If the
    positive test's ``[start, None, None]`` merely reflected a state machine
    that can only ever return one action, this mutant would also return ``None``
    for the second press. It does not: it returns ``stop``, so the positive
    assertion genuinely depends on the guard existing.
    """
    m = HotkeyStateMachine(mode="hold")
    first = m.on_press("alt_r")
    m._held.clear()  # simulate a build with the repeat guard removed
    second = m.on_press("alt_r")
    m._held.clear()
    third = m.on_press("alt_r")

    assert first == "start"
    assert second == "stop", "without the guard the second press must be read as a stop"
    assert third == "start"
    assert second is not None
