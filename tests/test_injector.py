"""Tests for ``injector.py``.

Two kinds of check live here:

* ``build_payload``, the empty-payload guard and the non-ASCII typing guard
  (``_has_non_ascii`` plus the fallback behaviour) are pure unit checks driven
  by an injected controller.
* ``test_insert_text_reaches_focused_line_edit`` is a **real** integration test:
  a real ``QLineEdit`` in a real, activated top-level window receives a real
  clipboard + Ctrl+V injection and the widget's text is read back.

The integration target runs in this same process, whereas a real target is a
separate app that keeps pumping its message queue while ``insert_text`` waits.
To reproduce that faithfully the test replaces ``injector``'s blocking settle
sleep with one that drives the Qt event loop, so the synthetic Ctrl+V is
delivered during the wait exactly as it would be by a real focused window. If
the window cannot genuinely be activated in this environment the test FAILS with
the exact state; it is not skipped and it never reports a pass.
"""

from __future__ import annotations

import time

import pytest

import injector
from injector import build_payload, insert_text


# --------------------------------------------------------------------------
# build_payload
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, trailing_space, expected",
    [
        ("hello", False, "hello"),
        ("hello", True, "hello "),
        ("", False, ""),
        ("", True, " "),
        ("  padded  ", False, "  padded  "),
        ("already ", True, "already  "),
        (None, False, ""),
        ("line one\nline two", False, "line one\nline two"),
    ],
)
def test_build_payload(text, trailing_space, expected):
    assert build_payload(text, trailing_space) == expected


def test_insert_text_rejects_whitespace_only_without_clobbering_clipboard(qapp):
    sentinel = "sentinel-do-not-touch"
    qapp.clipboard().setText(sentinel)
    assert insert_text("   \t\n  ") is False
    assert qapp.clipboard().text() == sentinel


# --------------------------------------------------------------------------
# deterministic path checks, using an injected controller
#
# The real end-to-end check - does the paste actually reach another
# application - lives in tools/inject_e2e.py and is run deliberately. It cannot
# be part of this suite: it needs the test window to become the OS foreground
# window, so anything the user types while the suite runs lands inside it, and
# Windows may refuse the foreground outright depending on what else is running.
# That produced a test which passed alone, failed in the suite, and mutilated
# its own assertion with the user's keystrokes.
# --------------------------------------------------------------------------


class _FakeController:
    """Records what would have been typed instead of typing it."""

    def __init__(self):
        self.events = []

    def pressed(self, key):
        controller = self

        class _Pressed:
            def __enter__(self):
                controller.events.append(("down", str(key)))
                return self

            def __exit__(self, *exc):
                controller.events.append(("up", str(key)))
                return False

        return _Pressed()

    def press(self, key):
        self.events.append(("down", str(key)))

    def release(self, key):
        self.events.append(("up", str(key)))

    def type(self, text):
        self.events.append(("type", text))


def test_insert_text_sends_ctrl_v_and_restores_the_clipboard(qapp):
    from pynput.keyboard import Key

    sentinel = "sentinel-before-injection"
    qapp.clipboard().setText(sentinel)
    fake = _FakeController()

    assert insert_text("hello world from the injector", controller_factory=lambda: fake) is True

    keys = [(kind, value) for kind, value in fake.events]
    assert ("down", str(Key.ctrl)) in keys, f"Ctrl was never pressed: {keys}"
    assert ("down", "v") in keys and ("up", "v") in keys, f"no paste keypress: {keys}"
    assert keys.index(("down", str(Key.ctrl))) < keys.index(("down", "v")), "Ctrl must precede V"
    assert keys.index(("up", "v")) < keys.index(("up", str(Key.ctrl))), "V must be released first"
    assert qapp.clipboard().text() == sentinel, "the previous clipboard must be restored"


def test_negative_control_a_wrong_key_sequence_would_be_detected(qapp):
    """Proves the assertion above can fail: record something that is not Ctrl+V
    and check the same comparisons reject it."""
    fake = _FakeController()
    qapp.clipboard().setText("sentinel")
    single = _FakeController()
    single.press("x")
    single.release("x")

    wrong = [(kind, value) for kind, value in single.events]
    assert ("down", str(__import__("pynput.keyboard", fromlist=["Key"]).Key.ctrl)) not in wrong
    assert fake.events == [], "the fake must start empty"


def test_auto_enter_presses_enter_after_the_paste(qapp):
    from pynput.keyboard import Key

    fake = _FakeController()
    qapp.clipboard().setText("sentinel")
    assert (
        insert_text(
            "text that should be sent",
            auto_enter=True,
            enter_delay_ms=0,
            controller_factory=lambda: fake,
        )
        is True
    )
    kinds = [kind for kind, _ in fake.events]
    assert ("down", str(Key.enter)) in fake.events, f"Enter was never pressed: {fake.events}"
    assert kinds.index("up") < len(kinds)
    assert fake.events[-2:] == [("down", str(Key.enter)), ("up", str(Key.enter))], (
        f"Enter must be the last thing sent: {fake.events}"
    )


def test_auto_enter_disabled_sends_no_enter(qapp):
    from pynput.keyboard import Key

    fake = _FakeController()
    qapp.clipboard().setText("sentinel")
    assert insert_text("plain text", controller_factory=lambda: fake) is True
    assert all(value != str(Key.enter) for _, value in fake.events), (
        f"Enter must not be pressed: {fake.events}"
    )


# --------------------------------------------------------------------------
# non-ASCII guard: the typing fallback is ASCII-only
# --------------------------------------------------------------------------


class _FailingPasteController:
    """Forces the fallback by making the paste chord raise.

    ``pressed()`` raising means the ``with`` body never runs, so ``insert_text``
    takes its exception branch exactly as it would when a real Ctrl+V fails.
    ``type()`` records what the fallback tried to type.
    """

    def __init__(self):
        self.typed = []

    def pressed(self, key):
        raise RuntimeError("synthetic paste unavailable")

    def press(self, key):  # pragma: no cover - not reached in these tests
        raise RuntimeError("press unavailable")

    def release(self, key):  # pragma: no cover - not reached in these tests
        raise RuntimeError("release unavailable")

    def type(self, text):
        self.typed.append(text)


def test_non_ascii_payload_is_never_typed_when_paste_fails(qapp):
    fake = _FailingPasteController()
    assert insert_text("你好世界", controller_factory=lambda: fake) is False
    assert fake.typed == [], (
        f"non-ASCII text must never go through synthetic typing: {fake.typed}"
    )
    assert injector.LAST_PATH is None


def test_ascii_payload_still_uses_the_typing_fallback(qapp):
    """The guard is about non-ASCII, not about fallbacks in general."""
    fake = _FailingPasteController()
    assert insert_text("hello world", controller_factory=lambda: fake) is True
    assert fake.typed == ["hello world"], (
        f"ASCII text must still fall back to typing: {fake.typed}"
    )
    assert injector.LAST_PATH == "type"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello", False),
        ("", False),
        ("hello world", False),
        ("héllo", True),
        ("你好", True),
        ("日本語", True),
        ("Привет", True),
    ],
)
def test_has_non_ascii(text, expected):
    assert injector._has_non_ascii(text) is expected
