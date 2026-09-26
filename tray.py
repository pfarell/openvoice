"""The system-tray settings surface for OpenVoice.

``Tray`` owns a ``QSystemTrayIcon`` and a menu whose checkable items mirror the
current ``Config`` values. Check states are refreshed every time the menu is
opened (``aboutToShow``), so a value changed elsewhere is reflected without
re-creating the menu. The one exception is "Start with Windows": the Start
Menu shortcut on disk is the truth, and ``config['run_at_login']`` is treated
as its cached mirror.

Dependency direction per docs/design.md: this module imports only ``bus``,
``config`` and stdlib/third-party. ``audio`` and ``tools.setup_login`` are
imported lazily so that importing this module never depends on them existing.

Input size: one small menu (O(10) items) plus the input-device list. Menu work
is O(devices) time; there is no per-item loop over anything larger.

Complexity: O(n) in the number of menu items / devices on a refresh; O(1)
otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QInputDialog, QMenu, QSystemTrayIcon

STATE_LABELS = {
    "idle": "Idle",
    "recording": "Recording…",
    "transcribing": "Transcribing…",
    "error": "Error",
}

MODE_LABELS = (
    ("hold", "Hold to talk"),
    ("toggle", "Tap to toggle"),
    ("both", "Both"),
)

#: Selectable recognition languages, in menu order: ``(code, label)``. "auto"
#: lets the engine detect the language itself; every other code is a hard hint
#: handed to faster-whisper as the ``language`` argument.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("id", "Indonesian"),
    ("zh", "Chinese (Mandarin)"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("tr", "Turkish"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("fil", "Filipino"),
    ("ta", "Tamil"),
    ("fa", "Persian"),
    ("he", "Hebrew"),
)

ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


def language_label(code: str) -> str:
    """Human label for a recognition language code, e.g. ``"en-US" -> "English"``.

    Matching is case-insensitive and on the base code only, so ``en-US``,
    ``EN_us`` and ``en`` all resolve to English. An empty code is ``"Unknown"``
    and an unrecognised one is returned verbatim rather than guessed at.

    Complexity: O(1) in ``LANGUAGES`` (a fixed 25-entry tuple); the code is
    split a constant number of times.
    """
    raw = "" if code is None else str(code).strip()
    if not raw:
        return "Unknown"
    base = raw.replace("_", "-").split("-", 1)[0].lower()
    for known, label in LANGUAGES:
        if known == base:
            return label
    return raw


class Tray(QObject):
    """Tray icon + settings menu, driven entirely by ``bus`` and ``config``.

    ``actions`` provides the callables the tray may invoke:

        toggle_recording   required - left-click / menu-less activation
        quit               required - Quit menu item
        reload_mic         optional - called after the microphone changes
    """

    def __init__(self, bus, config, actions: dict, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.config = config
        self.actions = dict(actions or {})
        self._syncing = False
        self._last_language: str | None = None

        self.menu = QMenu()
        self.menu.setToolTipsVisible(True)
        self._build_menu()

        self.tray_icon = QSystemTrayIcon(self._app_icon(), self)
        self.tray_icon.setToolTip("OpenVoice")
        self.tray_icon.setContextMenu(self.menu)

        self.menu.aboutToShow.connect(self._sync_from_config)
        self.bus.state_changed.connect(self._on_state_changed)
        self.config.on_change(self._on_config_changed)
        self.tray_icon.activated.connect(self._on_activated)

        self._sync_from_config()
        self.tray_icon.show()

    # ------------------------------------------------------------------ build

    def _build_menu(self) -> None:
        menu = self.menu

        self._status_action = QAction(STATE_LABELS["idle"], menu)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)
        menu.addSeparator()

        mode_menu = menu.addMenu("Recording mode")
        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_actions: dict[str, QAction] = {}
        for key, label in MODE_LABELS:
            action = QAction(label, menu)
            action.setCheckable(True)
            self._mode_group.addAction(action)
            action.triggered.connect(lambda _checked=False, k=key: self._set_mode(k))
            mode_menu.addAction(action)
            self._mode_actions[key] = action

        self._auto_enter_action = self._add_checkable(
            "Auto-enter after paste", "auto_enter"
        )
        self._trailing_space_action = self._add_checkable(
            "Append a space after text", "trailing_space"
        )

        menu.addSeparator()
        self._language_menu = menu.addMenu("Language")
        self._language_menu.setToolTip(
            "Which language the recogniser should expect. Auto-detect lets the "
            "engine decide from the audio."
        )
        self._language_group = QActionGroup(self)
        self._language_group.setExclusive(True)
        self._language_actions: dict[str, QAction] = {}
        for code, label in LANGUAGES:
            action = QAction(label, menu)
            action.setCheckable(True)
            self._language_group.addAction(action)
            action.triggered.connect(lambda _checked=False, c=code: self._set_language(c))
            self._language_menu.addAction(action)
            self._language_actions[code] = action

        menu.addSeparator()
        self._mic_menu = menu.addMenu("Microphone")
        self._mic_actions: list[tuple[QAction, object, str]] = []
        self._mic_group = QActionGroup(self)
        self._mic_group.setExclusive(True)
        self._populate_mic_menu()

        vocab_action = QAction("Custom vocabulary…", menu)
        vocab_action.setToolTip(
            "Names, jargon and acronyms you expect to dictate, so the recogniser "
            "is primed for them."
        )
        vocab_action.setStatusTip(vocab_action.toolTip())
        vocab_action.triggered.connect(self._edit_vocabulary)
        menu.addAction(vocab_action)

        menu.addSeparator()
        self._pill_action = self._add_checkable("Show the indicator", "pill_enabled")

        self._run_login_action = QAction("Start with Windows", menu)
        self._run_login_action.setCheckable(True)
        self._run_login_action.setToolTip(
            "Launch OpenVoice from your Windows Startup folder."
        )
        self._run_login_action.triggered.connect(self._set_run_at_login)
        menu.addAction(self._run_login_action)

        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(lambda: self._invoke("quit"))
        menu.addAction(quit_action)

    def _add_checkable(self, label: str, config_key: str) -> QAction:
        action = QAction(label, self.menu)
        action.setCheckable(True)
        action.triggered.connect(
            lambda checked=False, k=config_key: self.config.set(k, bool(checked))
        )
        self.menu.addAction(action)
        return action

    def _populate_mic_menu(self) -> None:
        """Rebuild the microphone list from the current input devices."""
        self._mic_menu.clear()
        self._mic_group = QActionGroup(self)
        self._mic_group.setExclusive(True)
        self._mic_actions = []

        default_action = QAction("System default", self._mic_menu)
        default_action.setCheckable(True)
        default_action.triggered.connect(lambda _checked=False: self._set_mic(None))
        self._mic_group.addAction(default_action)
        self._mic_menu.addAction(default_action)
        self._mic_actions.append((default_action, None, "System default"))

        devices = self._list_input_devices()
        for index, name in devices:
            action = QAction(name, self._mic_menu)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, i=index: self._set_mic(i)
            )
            self._mic_group.addAction(action)
            self._mic_menu.addAction(action)
            self._mic_actions.append((action, index, name))

        if not devices:
            empty = QAction("No input devices found", self._mic_menu)
            empty.setEnabled(False)
            self._mic_menu.addAction(empty)

    # ------------------------------------------------------------------- sync

    def _sync_from_config(self) -> None:
        """Full refresh: rebuild the device list, then re-read every value."""
        self._populate_mic_menu()
        self._sync_checks()
        self.tray_icon.setToolTip(self._tooltip_text())

    def _sync_checks(self) -> None:
        """Set every checkable item from the current config (or disk) state."""
        if self._syncing:
            return
        self._syncing = True
        try:
            label = self._status_text()
            self._status_action.setText(label)
            self.tray_icon.setToolTip(self._tooltip_text())

            mode = self.config.get("mode")
            for key, action in self._mode_actions.items():
                action.setChecked(key == mode)

            self._auto_enter_action.setChecked(bool(self.config.get("auto_enter")))
            self._trailing_space_action.setChecked(
                bool(self.config.get("trailing_space"))
            )
            self._pill_action.setChecked(bool(self.config.get("pill_enabled")))

            current_language = str(self.config.get("language") or "auto")
            base_language = current_language.replace("_", "-").split("-", 1)[0].lower()
            for code, action in self._language_actions.items():
                action.setChecked(code == base_language)

            actual = self._is_run_at_login()
            self._run_login_action.setChecked(actual)
            if bool(self.config.get("run_at_login")) != actual:
                self.config.set("run_at_login", actual)

            self._sync_mic_checks(self.config.get("mic_device"))
        finally:
            self._syncing = False

    def _sync_mic_checks(self, value) -> None:
        for action, index, name in self._mic_actions:
            if value is None:
                action.setChecked(index is None)
            elif isinstance(value, bool):
                action.setChecked(index is None)
            elif isinstance(value, int):
                action.setChecked(index == value)
            elif isinstance(value, str):
                action.setChecked(
                    index is not None
                    and (name == value or value.lower() in name.lower())
                )
            else:
                action.setChecked(index is None)

    def _status_text(self) -> str:
        state = getattr(self.bus, "state", "idle")
        return STATE_LABELS.get(state, str(state).capitalize())

    def _tooltip_text(self, status: str | None = None) -> str:
        """Tooltip body: status, plus the last detected language once there is one.

        The language label is appended only after a dictation has been tagged
        with a language, so a freshly started tray keeps its original two-part
        tooltip. ``status`` may be supplied by the caller (an emitted state) so
        the tooltip matches what the status line just showed.
        """
        status = self._status_text() if status is None else status
        if self._last_language:
            return f"OpenVoice — {status} · {language_label(self._last_language)}"
        return f"OpenVoice — {status}"

    def set_last_language(self, code: str) -> None:
        """Remember the last dictation's language and show its label in the tooltip.

        Called by ``app`` after a successful insert. An empty code means nothing
        was detected and the tooltip is left in its current format.
        """
        if not code:
            return
        self._last_language = str(code)
        self.tray_icon.setToolTip(self._tooltip_text())

    def _on_state_changed(self, state: str) -> None:
        label = STATE_LABELS.get(state, str(state).capitalize())
        self._status_action.setText(label)
        self.tray_icon.setToolTip(self._tooltip_text(label))

    def _on_config_changed(self, _key: str) -> None:
        self._sync_checks()

    # ---------------------------------------------------------------- actions

    def _set_mode(self, mode: str) -> None:
        self.config.set("mode", mode)

    def _set_language(self, code: str) -> None:
        self.config.set("language", code)

    def _set_mic(self, value) -> None:
        self.config.set("mic_device", value)
        reload_mic = self.actions.get("reload_mic")
        if callable(reload_mic):
            try:
                reload_mic()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[tray] reload_mic failed: {exc}", file=sys.stderr)

    def _edit_vocabulary(self) -> None:
        current = self.config.get("custom_vocabulary") or ""
        text, accepted = QInputDialog.getMultiLineText(
            None,
            "Custom vocabulary",
            "Names, jargon and acronyms the recogniser should expect.\n"
            "One per line or comma-separated:",
            current,
        )
        if accepted:
            self.config.set("custom_vocabulary", text)

    def _set_run_at_login(self, checked: bool = False) -> None:
        desired = bool(checked)
        module = self._setup_login()
        ok = False
        if module is not None:
            try:
                ok = bool(module.set_run_at_login(desired))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[tray] set_run_at_login raised: {exc}", file=sys.stderr)
                ok = False

        if not ok:
            self.notify(
                "OpenVoice",
                "Could not update the Windows Startup entry. See the console "
                "for details.",
            )
            self._sync_checks()
            return

        self.config.set("run_at_login", desired)
        self._sync_checks()

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._invoke("toggle_recording")

    def _invoke(self, name: str) -> None:
        fn = self.actions.get(name)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[tray] action {name!r} failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------- interfaces

    def notify(self, title: str, message: str) -> None:
        """Surface a message through the tray balloon."""
        try:
            self.tray_icon.showMessage(
                title, message, QSystemTrayIcon.MessageIcon.Warning, 5000
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[tray] notify failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------ deps

    def _list_input_devices(self) -> list[tuple[int, str]]:
        try:
            import audio
        except ImportError:
            return []
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[tray] audio import failed: {exc}", file=sys.stderr)
            return []
        try:
            raw = audio.list_input_devices()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[tray] list_input_devices failed: {exc}", file=sys.stderr)
            return []
        devices: list[tuple[int, str]] = []
        for entry in raw or []:
            try:
                devices.append((int(entry[0]), str(entry[1])))
            except Exception:
                continue
        return devices

    def _setup_login(self):
        try:
            from tools import setup_login
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[tray] setup_login unavailable: {exc}", file=sys.stderr)
            return None
        return setup_login

    def _is_run_at_login(self) -> bool:
        module = self._setup_login()
        if module is not None:
            try:
                return bool(module.is_run_at_login())
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[tray] is_run_at_login failed: {exc}", file=sys.stderr)
        return bool(self.config.get("run_at_login"))

    # ------------------------------------------------------------------ icon

    def _app_icon(self) -> QIcon:
        if ICON_PATH.exists():
            icon = QIcon(str(ICON_PATH))
            if not icon.isNull():
                return icon
        return self._fallback_icon()

    @staticmethod
    def _fallback_icon(size: int = 64) -> QIcon:
        """A minimal drawn microphone, used only when assets/icon.ico is absent."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        s = float(size)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        inset = s * 0.035
        square = QRectF(inset, inset, s - 2.0 * inset, s - 2.0 * inset)
        grad = QLinearGradient(square.topLeft(), square.bottomLeft())
        grad.setColorAt(0.0, QColor("#2b3444"))
        grad.setColorAt(1.0, QColor("#171c24"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(square, square.width() * 0.235, square.width() * 0.235)

        cx = s * 0.5
        p.setBrush(QColor("#ffffff"))
        cap_w = s * 0.255
        p.drawRoundedRect(
            QRectF(cx - cap_w * 0.5, s * 0.155, cap_w, s * 0.40),
            cap_w * 0.5,
            cap_w * 0.5,
        )

        arc_r = s * 0.195
        arc_cy = s * 0.505
        arc_pen = QPen(QColor("#ffffff"), s * 0.072)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(
            QRectF(cx - arc_r, arc_cy - arc_r, 2.0 * arc_r, 2.0 * arc_r),
            180 * 16,
            180 * 16,
        )

        stem_pen = QPen(QColor("#ffffff"), s * 0.052)
        stem_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(stem_pen)
        p.drawLine(QPointF(cx, arc_cy + arc_r), QPointF(cx, s * 0.795))

        base_pen = QPen(QColor("#ffffff"), s * 0.058)
        base_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(base_pen)
        p.drawLine(QPointF(cx - s * 0.125, s * 0.805), QPointF(cx + s * 0.125, s * 0.805))

        p.end()
        return QIcon(pixmap)
