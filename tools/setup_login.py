"""Put OpenVoice in (or take it out of) the user's Windows Startup folder.

No COM, no pywin32, no registry: a single ``OpenVoice.cmd`` file in

    %APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup

launches the project's own ``pythonw.exe`` on ``app.py`` through ``start``.
``pythonw.exe`` keeps the app console-free.

Input size: one directory, one file. Every operation is a single stat/create/
delete, i.e. O(1) time and space.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_NAME = "openvoice.cmd"


def startup_dir() -> Path | None:
    """The user's Startup folder, or None when %APPDATA% is unavailable."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def shortcut_path() -> Path | None:
    base = startup_dir()
    if base is None:
        return None
    return base / SCRIPT_NAME


def _command_text() -> str:
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    app = ROOT / "app.py"
    return (
        "@echo off\r\n"
        f'start "" /D "{ROOT}" "{pythonw}" "{app}"\r\n'
    )


def set_run_at_login(enabled: bool) -> bool:
    """Create or delete the Startup shortcut. True on success.

    On failure the reason is printed to stderr and False is returned; the
    caller is expected to surface that rather than assume success.
    """
    path = shortcut_path()
    if path is None:
        print(
            "[setup_login] %APPDATA% is not set; cannot locate the Startup folder",
            file=sys.stderr,
        )
        return False

    if enabled:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # UTF-8 without BOM: the content is pure ASCII, so cmd.exe is happy.
            path.write_text(_command_text(), encoding="utf-8", newline="")
        except OSError as exc:
            print(f"[setup_login] could not create {path}: {exc}", file=sys.stderr)
            return False
        return True

    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[setup_login] could not delete {path}: {exc}", file=sys.stderr)
        return False
    return True


def is_run_at_login() -> bool:
    """True when the Startup shortcut exists."""
    path = shortcut_path()
    return bool(path is not None and path.exists())


def _list_startup() -> None:
    base = startup_dir()
    if base is None:
        print("Startup folder: <unavailable: %APPDATA% is not set>")
        return
    print(f"Startup folder: {base}")
    if not base.is_dir():
        print("  <folder does not exist>")
        return
    entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    if not entries:
        print("  <empty>")
        return
    for entry in entries:
        kind = "dir " if entry.is_dir() else "file"
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            size = 0
        print(f"  [{kind}] {entry.name} ({size} bytes)")


def main() -> int:
    print(f"project root: {ROOT}")
    print(f"shortcut:     {shortcut_path()}")
    print()
    print("--- before ---")
    print(f"is_run_at_login() -> {is_run_at_login()}")
    _list_startup()

    print()
    print("--- enable ---")
    print(f"set_run_at_login(True) -> {set_run_at_login(True)}")
    print(f"is_run_at_login() -> {is_run_at_login()}")
    _list_startup()
    target = shortcut_path()
    if target is not None and target.exists():
        print()
        print(f"--- content of {target.name} ---")
        print(target.read_text(encoding="utf-8"))

    print()
    print("--- disable ---")
    print(f"set_run_at_login(False) -> {set_run_at_login(False)}")
    print(f"is_run_at_login() -> {is_run_at_login()}")
    _list_startup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
