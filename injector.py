"""Insert transcribed text into the focused window.

The reliable, format-preserving path is clipboard + Ctrl+V: set the clipboard to
the payload, send the paste chord, wait for the target app to consume it, then
put the previous clipboard back *only if* it still holds our payload (so a copy
the user made during the wait is never clobbered). If the paste path raises for
any reason, fall back to synthetic typing; the clipboard remains the only
supported path for non-ASCII text (see :func:`_has_non_ascii`).

Clipboard access prefers ``QGuiApplication.clipboard()`` when a Qt application
exists and falls back to the Win32 clipboard through ``ctypes`` otherwise; both
readers and writers report failures to stderr rather than swallowing them.

Complexity
----------
``build_payload`` is O(n) time and O(n) space in the input length (one string
concatenation). ``insert_text`` is O(n) time for the clipboard copy and the
optional typing fallback, plus three fixed sleeps; O(n) space for the payload
and the saved clipboard text.
"""

from __future__ import annotations

import ctypes
import sys
import time

PASTE_SETTLE_SECONDS = 0.3

#: The injection path used by the most recent :func:`insert_text` call:
#: ``"clipboard"``, ``"type"``, or ``None`` before the first call.
LAST_PATH: str | None = None


def build_payload(text, trailing_space: bool = False) -> str:
    """Return the exact string to insert, optionally with a trailing space.

    ``None`` becomes ``""`` so callers can pass an unset transcript safely.
    """
    payload = "" if text is None else str(text)
    if trailing_space:
        payload = payload + " "
    return payload


def _has_non_ascii(text: str) -> bool:
    """True when ``text`` contains any character outside ASCII (``ord > 127``).

    Synthetic typing drives the active keyboard layout, so a CJK, Thai or other
    non-ASCII character the layout cannot produce is dropped or substituted and
    the inserted text is mangled. The clipboard paste path is Unicode-correct,
    so it is the only supported path for non-ASCII text; typing is reserved for
    ASCII payloads. Complexity: O(n) in the length of ``text``.
    """
    return any(ord(ch) > 127 for ch in text)


def insert_text(
    text,
    auto_enter: bool = False,
    enter_delay_ms: int = 120,
    trailing_space: bool = False,
    restore_clipboard: bool = True,
    controller_factory=None,
) -> bool:
    """Paste ``text`` through the clipboard into the focused window.

    Returns ``True`` on success and ``False`` on failure, printing the reason to
    stderr. An empty or whitespace-only payload is a no-op that returns ``False``
    without touching the clipboard. Every path used is printed and recorded in
    :data:`LAST_PATH`.
    """
    global LAST_PATH

    payload = build_payload(text, trailing_space)
    if not payload or not payload.strip():
        print("[injector] refusing empty/whitespace payload", file=sys.stderr)
        return False

    try:
        from pynput.keyboard import Controller, Key
    except Exception as exc:
        print(f"[injector] pynput unavailable: {exc}", file=sys.stderr)
        return False

    try:
        controller = (controller_factory or Controller)()
    except Exception as exc:  # pragma: no cover - platform controller failure
        print(f"[injector] keyboard controller unavailable: {exc}", file=sys.stderr)
        return False

    previous = _read_clipboard() if restore_clipboard else None
    try:
        _write_clipboard(payload)
    except Exception as exc:
        print(f"[injector] could not set clipboard: {exc}", file=sys.stderr)
        return False

    try:
        with controller.pressed(Key.ctrl):
            controller.press("v")
            controller.release("v")
        path = "clipboard"
    except Exception as exc:
        print(
            f"[injector] clipboard paste failed ({exc}); falling back to typing",
            file=sys.stderr,
        )
        if _has_non_ascii(payload):
            print(
                "[injector] refusing to type non-ASCII text: synthetic keystrokes "
                "drive the active keyboard layout and mangle CJK/Thai/other "
                "non-ASCII output; the clipboard paste path is the only supported "
                "one for it",
                file=sys.stderr,
            )
            if restore_clipboard:
                _restore_clipboard(previous, payload)
            LAST_PATH = None
            return False
        try:
            controller.type(payload)
            path = "type"
        except Exception as exc2:
            print(f"[injector] typing fallback failed: {exc2}", file=sys.stderr)
            if restore_clipboard:
                _restore_clipboard(previous, payload)
            LAST_PATH = None
            return False

    LAST_PATH = path
    print(f"[injector] inserted {len(payload)} chars via {path}", file=sys.stderr)

    time.sleep(PASTE_SETTLE_SECONDS)
    if restore_clipboard:
        _restore_clipboard(previous, payload)

    if auto_enter:
        try:
            time.sleep(max(0, enter_delay_ms) / 1000.0)
            controller.press(Key.enter)
            controller.release(Key.enter)
        except Exception as exc:
            print(f"[injector] auto-enter failed: {exc}", file=sys.stderr)
            return False

    return True


# -- clipboard restore -----------------------------------------------------
def _restore_clipboard(previous: str | None, payload: str) -> None:
    """Restore ``previous`` only if the clipboard still holds our payload."""
    try:
        current = _read_clipboard()
    except Exception as exc:
        print(f"[injector] could not read clipboard for restore: {exc}", file=sys.stderr)
        return
    if current != payload:
        return
    try:
        _write_clipboard(previous or "")
    except Exception as exc:
        print(f"[injector] could not restore clipboard: {exc}", file=sys.stderr)


# -- clipboard access: Qt first, Win32 ctypes otherwise --------------------
def _qt_clipboard():
    try:
        from PySide6.QtGui import QGuiApplication
    except Exception:
        return None
    if QGuiApplication.instance() is None:
        return None
    return QGuiApplication.clipboard()


def _read_clipboard() -> str:
    """Read the clipboard, native first.

    The native path is synchronous and needs no message pump. Qt's clipboard is
    only a fallback because Qt owns its clipboard data through the event loop:
    if the loop is blocked, Qt cannot serve a paste request that arrives while we
    are waiting, and the paste silently pastes stale content instead.
    """
    try:
        return _win32_read_clipboard()
    except Exception as exc:
        print(f"[injector] native clipboard read failed: {exc}", file=sys.stderr)
    cb = _qt_clipboard()
    if cb is not None:
        try:
            return cb.text()
        except Exception as exc:
            print(f"[injector] Qt clipboard read failed: {exc}", file=sys.stderr)
    return ""


def _write_clipboard(text: str) -> None:
    """Write the clipboard, native first. See ``_read_clipboard`` for why."""
    try:
        _win32_write_clipboard(text)
        return
    except Exception as exc:
        print(f"[injector] native clipboard write failed: {exc}", file=sys.stderr)
    cb = _qt_clipboard()
    if cb is not None:
        try:
            cb.setText(text)
            return
        except Exception as exc:
            print(f"[injector] Qt clipboard write failed: {exc}", file=sys.stderr)


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def _win32_read_clipboard() -> str:
    if sys.platform != "win32":
        raise RuntimeError("Win32 clipboard is only available on Windows")
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool

    if not user32.OpenClipboard(None):
        raise OSError(f"OpenClipboard failed (error {ctypes.GetLastError()})")
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _win32_write_clipboard(text: str) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Win32 clipboard is only available on Windows")
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.EmptyClipboard.restype = ctypes.c_bool
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool

    buffer = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(buffer)

    if not user32.OpenClipboard(None):
        raise OSError(f"OpenClipboard failed (error {ctypes.GetLastError()})")
    try:
        if not user32.EmptyClipboard():
            raise OSError("EmptyClipboard failed")
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            raise OSError("GlobalAlloc failed")
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            kernel32.GlobalFree(handle)
            raise OSError("GlobalLock failed")
        try:
            ctypes.memmove(ptr, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise OSError("SetClipboardData failed")
        # Ownership of `handle` transfers to the clipboard on success.
    finally:
        user32.CloseClipboard()
