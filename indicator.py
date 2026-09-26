"""Floating pill indicator for OpenVoice.

A frameless, click-through, never-focusing capsule that shows what the app is
doing. Everything is painted by hand in :meth:`Indicator.paintEvent`; there is
no widget styling and no child widget.

Level model
-----------
The Recorder pushes raw RMS in [0, 1] on ``bus.audio_level``. All smoothing
lives here, per the design contract:

* one exponential envelope per frame with **fast attack / slow release** so a
  spoken syllable snaps up and then dies away gently instead of flickering;
* a fixed-length history ring buffer of those envelopes;
* each of the 7 bars reads the ring at its own small delay, so a change in
  level travels across the pill rather than moving every bar in lockstep;
* each bar has its own gain, attack and release, so the row never looks
  mechanically rigid (a light "breathing" pulse keeps a silent mic alive).

Per-frame paint complexity
--------------------------
Each ``paintEvent`` draws a constant amount of geometry: one rounded-rect fill,
one rounded-rect stroke, and N_BARS = 7 rounded bars (or N_DOTS = 3 circles
while transcribing), plus at most one short glyph. Every bar reads exactly one
element from a fixed-size ring buffer and does O(1) arithmetic, so a frame is
**O(N_BARS + N_DOTS) = O(1)** time and reuses fixed buffers, allocating nothing
per bar beyond the small QRectF/QColor objects QPainter itself needs. The
matching level update in :meth:`_tick` is likewise O(N_BARS): one ring write and
one exponential step per bar. The 60 Hz timer is stopped whenever the widget is
hidden, so an idle indicator costs zero frames.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from bus import Bus
from config import Config

# --- geometry (logical / device-independent pixels) -------------------------
N_BARS = 7
BAR_W = 4.0
BAR_GAP = 4.0
GROUP_W = N_BARS * BAR_W + (N_BARS - 1) * BAR_GAP  # 52.0
MIN_BAR_H = 4.0
MAX_BAR_H = 24.0

PILL_W = 200.0
PILL_H = 44.0
MARGIN = 4.0  # AA + scale headroom around the pill inside the window
SCREEN_BOTTOM_OFFSET = 72.0

N_DOTS = 3
DOT_RADIUS = 2.6
DOT_SPAN = 76.0
DOT_SPEED = 0.00052  # cycles per millisecond

# --- level model ------------------------------------------------------------
HISTORY = 48  # ring length; must exceed max(DELAYS)
DELAYS = (0, 2, 4, 6, 8, 10, 12)  # frames of lag per bar -> travel direction
GAINS = (0.80, 0.92, 1.04, 1.12, 1.04, 0.90, 0.78)  # gentle arch
BAR_ATTACK = (0.72, 0.70, 0.68, 0.66, 0.68, 0.70, 0.72)
BAR_RELEASE = (0.34, 0.36, 0.38, 0.40, 0.38, 0.36, 0.34)
ENV_ATTACK = 0.60  # per 60 Hz frame; rises almost instantly
ENV_RELEASE = 0.06  # per 60 Hz frame; ~270 ms tail
BREATH_DEPTH = 0.035
BREATH_RATE = 0.0032  # radians per millisecond

# --- transitions ------------------------------------------------------------
FRAME_MS = 1000.0 / 60.0
SHOW_MS = 150.0
HIDE_MS = 120.0
SCALE_FROM = 0.96
ERROR_IN_MS = 60.0
ERROR_HOLD_MS = 300.0
ERROR_OUT_MS = 180.0

# --- palette ----------------------------------------------------------------
FILL = QColor(11, 11, 15, 205)  # #0B0B0F @ ~205 alpha
# Spec called for rgba(255,255,255,28); measured on a near-black background that
# only stepped the rim to ~(36,36,39) against a (20,20,24) field, leaving the
# capsule all but invisible. 40 keeps it a hairline while separating it on dark.
BORDER = QColor(255, 255, 255, 40)
BAR_LIGHT = QColor(242, 243, 245)  # #F2F3F5
BAR_DIM = QColor(168, 173, 181, 200)
BAR_ERROR = QColor(228, 112, 112)
FILL_ERROR = QColor(26, 10, 12, 205)
BORDER_ERROR = QColor(232, 96, 96, 90)
GLYPH = QColor(242, 243, 245, 110)

STATES = ("idle", "recording", "transcribing", "error")


def _ease_out_cubic(p: float) -> float:
    return 1.0 - (1.0 - p) ** 3


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear RGBA blend of two colours; t clamped to [0, 1]."""
    if t <= 0.0:
        return QColor(a)
    if t >= 1.0:
        return QColor(b)
    return QColor(
        int(round(a.red() + (b.red() - a.red()) * t)),
        int(round(a.green() + (b.green() - a.green()) * t)),
        int(round(a.blue() + (b.blue() - a.blue()) * t)),
        int(round(a.alpha() + (b.alpha() - a.alpha()) * t)),
    )


class Indicator(QWidget):
    """The floating pill. Driven by ``bus`` and read from ``config``."""

    def __init__(self, bus: Bus, config: Config) -> None:
        super().__init__(None)
        self._bus = bus
        self._config = config
        self._state = "idle"

        self.setWindowTitle("OpenVoice")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAutoFillBackground(False)
        self.setFixedSize(int(PILL_W + 2 * MARGIN), int(PILL_H + 2 * MARGIN))

        # transition state
        self._alpha = 0.0
        self._scale = 1.0
        self._anim = "idle"  # "idle" | "in" | "out" | "steady"
        self._anim_elapsed = 0.0
        self._clock = QElapsedTimer()
        self._last_ms = 0.0

        # level state
        self._target = 0.0
        self._env = 0.0
        self._history = [0.0] * HISTORY
        self._head = 0
        self._bar_val = [0.0] * N_BARS
        self._breath = 0.0

        # mode state
        self._dots_blend = 0.0
        self._dot_phase = 0.0
        self._error_t = 0.0
        self._err = 0.0

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(int(FRAME_MS))
        self._timer.timeout.connect(self._tick)

        bus.state_changed.connect(self.set_state)
        bus.audio_level.connect(self._on_level)
        bus.config_changed.connect(self._on_config)

    # ------------------------------------------------------------------ bus
    def _on_level(self, level: float) -> None:
        if level != level:  # NaN guard; never poison the smoother
            return
        if level < 0.0:
            level = 0.0
        elif level > 1.0:
            level = 1.0
        self._target = level

    def _on_config(self, key: str) -> None:
        if key == "pill_enabled" and not self._config.get("pill_enabled"):
            self._hard_hide()
            return
        if key == "pill_enabled" and self._state != "idle" and not self.isVisible():
            self._begin_show()
        self.update()

    # ---------------------------------------------------------------- states
    def set_state(self, state: str) -> None:
        """One of ``idle`` | ``recording`` | ``transcribing`` | ``error``."""
        if state not in STATES:
            return
        if state == self._state and state != "error":
            return
        self._state = state

        if not self._config.get("pill_enabled"):
            self._hard_hide()
            return

        if state == "idle":
            self._target = 0.0
            self._err = 0.0
            self._error_t = 0.0
            if self.isVisible():
                self._begin_hide()
            else:
                self._hard_hide()
            return

        if state == "error":
            self._target = 0.0
            self._dots_blend = 0.0
            self._err = 0.0
            self._error_t = 0.0
            self._scale = 1.0
            self._alpha = 1.0
            self._anim = "steady"
            if not self.isVisible():
                self._show_now()
            self._start_timer()
            self.update()
            return

        # recording / transcribing
        if state != "recording":
            self._target = 0.0
        if not self.isVisible() or self._anim == "idle" or self._alpha <= 0.0:
            self._begin_show()
        else:
            self._start_timer()
        self.update()

    # ----------------------------------------------------------- transitions
    def _show_now(self) -> None:
        self._position()
        if not self.isVisible():
            self.show()

    def _begin_show(self) -> None:
        self._position()
        if not self.isVisible():
            self.show()
        self._anim = "in"
        self._anim_elapsed = 0.0
        self._start_timer()

    def _begin_hide(self) -> None:
        self._anim = "out"
        self._anim_elapsed = 0.0
        self._start_timer()

    def _hard_hide(self) -> None:
        self._timer.stop()
        self._alpha = 0.0
        self._scale = 1.0
        self._anim = "idle"
        self._dots_blend = 0.0
        if self.isVisible():
            self.hide()

    def _start_timer(self) -> None:
        if not self._timer.isActive():
            self._clock.start()
            self._last_ms = 0.0
            self._timer.start()

    def _position(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - int(SCREEN_BOTTOM_OFFSET) - self.height()
        self.move(x, y)

    # ------------------------------------------------------------------ tick
    def _tick(self, dt_ms: float | None = None) -> None:
        if dt_ms is None:
            now = float(self._clock.elapsed())
            dt = now - self._last_ms
            self._last_ms = now
            if dt <= 0.0:
                dt = FRAME_MS
            elif dt > 100.0:
                dt = 100.0
        else:
            dt = float(dt_ms)

        self._advance_anim(dt)
        self._update_levels(dt)

        if self._state == "transcribing":
            self._dots_blend = min(1.0, self._dots_blend + dt / 160.0)
            self._dot_phase = (self._dot_phase + dt * DOT_SPEED) % 1.0
        else:
            self._dots_blend = max(0.0, self._dots_blend - dt / 120.0)

        if self._state != "error":
            self._err = max(0.0, self._err - dt / 120.0)

        self.update()

        if self._anim == "idle" and self._state == "idle":
            self._timer.stop()

    def _advance_anim(self, dt: float) -> None:
        if self._state == "error":
            self._error_t += dt
            t = self._error_t
            if t < ERROR_IN_MS:
                a = _smoothstep(t / ERROR_IN_MS)
            elif t < ERROR_IN_MS + ERROR_HOLD_MS:
                a = 1.0
            else:
                a = max(0.0, 1.0 - (t - ERROR_IN_MS - ERROR_HOLD_MS) / ERROR_OUT_MS)
            self._alpha = a
            self._err = a
            self._scale = 1.0
            if t >= ERROR_IN_MS + ERROR_HOLD_MS + ERROR_OUT_MS:
                self._hard_hide()
            return

        if self._anim == "in":
            self._anim_elapsed += dt
            p = min(1.0, self._anim_elapsed / SHOW_MS)
            e = _ease_out_cubic(p)
            self._alpha = e
            self._scale = SCALE_FROM + (1.0 - SCALE_FROM) * e
            if p >= 1.0:
                self._anim = "steady"
                self._alpha = 1.0
                self._scale = 1.0
        elif self._anim == "out":
            self._anim_elapsed += dt
            p = min(1.0, self._anim_elapsed / HIDE_MS)
            self._alpha = 1.0 - p
            self._scale = 1.0 - 0.015 * p
            if p >= 1.0:
                self._alpha = 0.0
                self._anim = "idle"
                if self.isVisible():
                    self.hide()
        # "steady": alpha/scale stay put

    def _update_levels(self, dt: float) -> None:
        scale = dt / FRAME_MS
        if scale <= 0.0:
            return
        # one envelope for the whole pill: fast attack, slow release
        coef = ENV_ATTACK if self._target > self._env else ENV_RELEASE
        k = 1.0 - (1.0 - coef) ** scale
        self._env += (self._target - self._env) * k
        if self._env < 0.0:
            self._env = 0.0

        self._history[self._head] = self._env
        self._head = (self._head + 1) % HISTORY

        self._breath = (self._breath + dt * BREATH_RATE) % (2.0 * math.pi)
        recording = self._state == "recording"

        for i in range(N_BARS):
            idx = (self._head - 1 - DELAYS[i]) % HISTORY
            raw = self._history[idx] * GAINS[i]
            if recording and raw < 0.10:
                raw += BREATH_DEPTH * (0.5 + 0.5 * math.sin(self._breath + i * 0.85))
            if raw > 1.0:
                raw = 1.0
            cur = self._bar_val[i]
            c = BAR_ATTACK[i] if raw > cur else BAR_RELEASE[i]
            kk = 1.0 - (1.0 - c) ** scale
            self._bar_val[i] = cur + (raw - cur) * kk

    # ----------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        a = self._alpha
        if a <= 0.004:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        p.translate(cx, cy)
        p.scale(self._scale, self._scale)
        p.translate(-cx, -cy)

        pill = QRectF(MARGIN, MARGIN, PILL_W, PILL_H)
        radius = PILL_H / 2.0
        err = self._err

        # capsule body
        p.setOpacity(a)
        p.setPen(Qt.NoPen)
        p.setBrush(_mix(FILL, FILL_ERROR, err))
        p.drawRoundedRect(pill, radius, radius)

        # hairline border
        pen = QPen(_mix(BORDER, BORDER_ERROR, err))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), radius - 0.5, radius - 0.5)

        bars_alpha = 1.0 - self._dots_blend
        if bars_alpha > 0.01:
            p.setPen(Qt.NoPen)
            p.setBrush(_mix(BAR_LIGHT, BAR_ERROR, err))
            for i in range(N_BARS):
                v = self._bar_val[i]
                h = MIN_BAR_H + (MAX_BAR_H - MIN_BAR_H) * (v ** 0.75)
                bx = cx - GROUP_W / 2.0 + i * (BAR_W + BAR_GAP)
                r = QRectF(bx, cy - h / 2.0, BAR_W, h)
                rr = (BAR_W if BAR_W < h else h) / 2.0
                p.setOpacity(a * bars_alpha)
                p.drawRoundedRect(r, rr, rr)

        dots_alpha = self._dots_blend
        if dots_alpha > 0.01:
            p.setPen(Qt.NoPen)
            p.setBrush(BAR_DIM)
            for k_dot in range(N_DOTS):
                t = (self._dot_phase + k_dot / float(N_DOTS)) % 1.0
                x = cx - DOT_SPAN / 2.0 + DOT_SPAN * t
                edge = (t if t < 1.0 - t else 1.0 - t) / 0.12
                if edge > 1.0:
                    edge = 1.0
                p.setOpacity(a * dots_alpha * 0.85 * (0.25 + 0.75 * edge))
                p.drawEllipse(QPointF(x, cy), DOT_RADIUS, DOT_RADIUS)

        # return glyph, only when auto-enter is armed
        if self._config.get("auto_enter"):
            f = QFont()
            f.setFamilies(["Segoe UI Symbol", "Segoe UI", "Arial"])
            f.setPixelSize(13)
            f.setWeight(QFont.Weight.DemiBold)
            p.setFont(f)
            p.setPen(GLYPH)
            p.setOpacity(a * 0.9)
            gr = QRectF(pill.right() - 24.0, cy - 10.0, 18.0, 20.0)
            p.drawText(gr, Qt.AlignCenter, "\u21b5")

        p.end()
