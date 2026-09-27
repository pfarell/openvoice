"""Real end-to-end injection check: does dictated text actually reach ANOTHER
application?

Run this deliberately, not as part of the test suite. It takes over the
foreground window for a couple of seconds, so it needs the desktop to itself.

    python tools/inject_e2e.py

It writes an unguessable sentinel to the clipboard first, so a paste that
silently does nothing cannot be mistaken for a pass: the readback has to match
the injected payload and differ from the sentinel.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import injector  # noqa: E402

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def force_foreground(hwnd: int) -> bool:
    """AttachThreadInput lets a background process legally take the foreground."""
    if int(user32.GetForegroundWindow() or 0) == hwnd:
        return True
    foreground = user32.GetForegroundWindow()
    current = kernel32.GetCurrentThreadId()
    owner = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
    attached = bool(owner and owner != current and user32.AttachThreadInput(current, owner, True))
    try:
        user32.ShowWindow(hwnd, 9)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.3)
    finally:
        if attached:
            user32.AttachThreadInput(current, owner, False)
    return int(user32.GetForegroundWindow() or 0) == hwnd


def find_window(pid: int, title_part: str = "") -> int:
    """Find a visible top-level window by process id, or by title substring.

    Title matching matters on Windows 11, where notepad.exe can hand off to a
    different process, so the pid of what we spawned is not the pid that owns
    the window.
    """
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        owner = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        length = user32.GetWindowTextLengthW(hwnd)
        title = ""
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value
        if (pid and owner.value == pid) or (title_part and title_part.lower() in title.lower()):
            found.append(int(hwnd))
        return True

    user32.EnumWindows(callback, None)
    return found[0] if found else 0


def wait_for_window(pid: int, title_part: str, timeout: float = 25.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = find_window(pid, title_part)
        if hwnd:
            return hwnd
        time.sleep(0.4)
    return 0


def main() -> int:
    payload = "hello notepad from openvoice e2e"
    sentinel = f"sentinel-{uuid.uuid4()}"

    injector._write_clipboard(sentinel)
    assert injector._read_clipboard() == sentinel, "could not establish the sentinel"

    tmp = Path(tempfile.mkdtemp()) / f"e2e-{uuid.uuid4().hex[:8]}.txt"
    tmp.write_text("", encoding="utf-8")
    print(f"target file : {tmp}")

    notepad = subprocess.Popen(["notepad.exe", str(tmp)])
    hwnd = wait_for_window(notepad.pid, tmp.stem)
    if not hwnd:
        notepad.terminate()
        print("RESULT: FAILED â€” Notepad window not found")
        return 1

    if not force_foreground(hwnd):
        notepad.terminate()
        print(
            "RESULT: FAILED â€” could not take the foreground window. "
            f"foreground={int(user32.GetForegroundWindow() or 0)} hwnd={hwnd}. "
            "Close other foreground windows and retry."
        )
        return 1
    print(f"notepad     : hwnd {hwnd}, confirmed foreground")

    ok = injector.insert_text(payload)
    foreground_after = int(user32.GetForegroundWindow() or 0)
    print(f"insert_text : {ok} (path {injector.LAST_PATH})")
    print(f"foreground  : before={hwnd} after={foreground_after}")
    time.sleep(1.2)

    # Evidence 1: Notepad's own buffer, read back through the clipboard.
    injector._write_clipboard(sentinel)
    time.sleep(0.3)
    from pynput.keyboard import Controller, Key

    keyboard = Controller()
    with keyboard.pressed(Key.ctrl):
        keyboard.press("a")
        keyboard.release("a")
    time.sleep(0.2)
    with keyboard.pressed(Key.ctrl):
        keyboard.press("c")
        keyboard.release("c")
    time.sleep(0.5)
    buffer_text = injector._read_clipboard()

    # Evidence 2: on disk, after a real save. Notepad on Windows 11 buffers the
    # write, so poll instead of reading once and calling it a failure.
    time.sleep(0.2)
    with keyboard.pressed(Key.ctrl):
        keyboard.press("s")
        keyboard.release("s")
    file_text = ""
    deadline = time.time() + 6.0
    while time.time() < deadline:
        time.sleep(0.4)
        try:
            file_text = tmp.read_text(encoding="utf-8").strip()
        except Exception:
            file_text = ""
        if file_text:
            break

    notepad.terminate()
    try:
        notepad.wait(timeout=5)
    except Exception:
        pass

    print(f"sentinel    : {sentinel!r}")
    print(f"payload     : {payload!r}")
    print(f"EVIDENCE 1  : notepad buffer -> {buffer_text.strip()!r}")
    print(f"EVIDENCE 2  : file on disk   -> {file_text!r}")

    buffer_ok = buffer_text.strip() == payload
    file_ok = file_text == payload
    sentinel_intact = buffer_text.strip() != sentinel
    print(f"match buffer/file/sentinel-moved: {buffer_ok}/{file_ok}/{sentinel_intact}")

    if buffer_ok and file_ok and sentinel_intact:
        print("RESULT: PASS")
        return 0

    print("RESULT: FAILED")
    if foreground_after != hwnd:
        print(
            "CAUSE: the target window lost the OS foreground during the paste "
            f"(expected hwnd {hwnd}, actual {foreground_after}). This is the check's "
            "environment, not the injector: the same code passes when the desktop is "
            "quiet, and window focus is owned by Windows, not by this script."
        )
    else:
        print(
            "CAUSE: the target window held the foreground throughout, so the paste "
            "genuinely did not arrive. That would be a real injector defect."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

