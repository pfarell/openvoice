"""Render every pill state to real PNGs so a human can judge the design.

Run from the project root:

    .venv\\Scripts\\python.exe tools\\shoot_indicator.py

For each state it writes ``tools/shots/<name>.png`` (transparent background)
and ``tools/shots/<name>_on_background.png`` (the same pill composited over a
two-tone strip - dark left half, light right half - so contrast is judgeable on
both surfaces at once).

The printed non-transparent-pixel count is a STRUCTURAL sanity check only: it
proves the widget rendered *something*. It is not evidence that the pill looks
good - that requires a human looking at the images.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus import Bus  # noqa: E402
from config import Config  # noqa: E402
from indicator import Indicator  # noqa: E402

SHOTS = ROOT / "tools" / "shots"
FRAME = 1000.0 / 60.0

STRIP_W, STRIP_H = 440, 120
DARK = QColor(20, 20, 24)
LIGHT = QColor(232, 233, 236)


# --------------------------------------------------------------- widget build
def _make(tmp: Path, auto_enter: bool = False):
    """Fresh Bus + Config + Indicator per shot so states never bleed."""
    bus = Bus()
    cfg = Config(path=tmp / "config.json")
    cfg.update({"pill_enabled": True, "auto_enter": auto_enter})
    ind = Indicator(bus, cfg)
    ind.setAttribute(Qt.WA_DontShowOnScreen, True)
    ind.show()
    QApplication.processEvents()
    return bus, cfg, ind


def _tick(ind: Indicator, n: int, dt: float = FRAME) -> None:
    for _ in range(n):
        ind._tick(dt)


def _pump(bus: Bus, ind: Indicator, ticks: int) -> None:
    """Feed a varying synthetic level so bars land at differing heights."""
    for t in range(ticks):
        lvl = 0.18 + 0.62 * abs(math.sin(t * 0.31)) * (0.65 + 0.35 * math.sin(t * 0.83))
        bus.audio_level.emit(max(0.0, min(1.0, lvl)))
        ind._tick(FRAME)


# ------------------------------------------------------------------- builders
def build_idle(bus: Bus, ind: Indicator) -> None:
    ind.set_state("idle")
    _tick(ind, 24)


def build_recording(bus: Bus, ind: Indicator) -> None:
    ind.set_state("recording")
    _tick(ind, 16)  # finish show transition
    _pump(bus, ind, 56)


def build_transcribing(bus: Bus, ind: Indicator) -> None:
    ind.set_state("transcribing")
    _tick(ind, 40)  # show done, dots blended in, phase advanced


def build_error(bus: Bus, ind: Indicator) -> None:
    ind.set_state("error")
    _tick(ind, 6)  # ~100 ms: past ERROR_IN, at peak red, before the fade


def build_recording_autoenter(bus: Bus, ind: Indicator) -> None:
    ind.set_state("recording")
    _tick(ind, 16)
    _pump(bus, ind, 56)


# ------------------------------------------------------------------- render
def _render(ind: Indicator) -> QImage:
    img = QImage(ind.size(), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    ind.render(p, QPoint(0, 0))
    p.end()
    return img


def _composite(img: QImage) -> QImage:
    out = QImage(STRIP_W, STRIP_H, QImage.Format_ARGB32_Premultiplied)
    p = QPainter(out)
    p.fillRect(0, 0, STRIP_W // 2, STRIP_H, DARK)
    p.fillRect(STRIP_W // 2, 0, STRIP_W - STRIP_W // 2, STRIP_H, LIGHT)
    p.drawImage((STRIP_W - img.width()) // 2, (STRIP_H - img.height()) // 2, img)
    p.end()
    return out


def _count_nontransparent(img: QImage) -> int:
    conv = img.convertToFormat(QImage.Format_ARGB32)
    w, h, bpl = conv.width(), conv.height(), conv.bytesPerLine()
    buf = np.frombuffer(conv.constBits(), dtype=np.uint8)
    arr = buf[: h * bpl].reshape(h, bpl)
    alpha = arr[:, : w * 4].reshape(h, w, 4)[:, :, 3]
    return int((alpha > 0).sum())


SPECS = [
    ("pill_idle", build_idle, False, "expected: nothing / fully transparent"),
    ("pill_recording", build_recording, False, ""),
    ("pill_transcribing", build_transcribing, False, ""),
    ("pill_error", build_error, False, ""),
    ("pill_recording_autoenter", build_recording_autoenter, True, ""),
]


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    SHOTS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rocky_shots_"))

    print(f"output directory: {SHOTS}")
    print("")
    for name, builder, auto_enter, note in SPECS:
        bus, cfg, ind = _make(tmp, auto_enter=auto_enter)
        builder(bus, ind)
        img = _render(ind)

        raw_path = SHOTS / f"{name}.png"
        img.save(str(raw_path), "PNG")
        comp = _composite(img)
        comp_path = SHOTS / f"{name}_on_background.png"
        comp.save(str(comp_path), "PNG")

        count = _count_nontransparent(img)
        suffix = f"  ({note})" if note else ""
        print(
            f"{raw_path}  {img.width()}x{img.height()}  "
            f"non-transparent pixels: {count}  [STRUCTURAL sanity only]{suffix}"
        )
        print(
            f"{comp_path}  {STRIP_W}x{STRIP_H}  "
            f"non-transparent pixels: {_count_nontransparent(comp)}  "
            f"[STRUCTURAL sanity only]"
        )
        ind.close()
    print("")
    print("NOTE: pixel counts only prove a render happened, not that it looks right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
