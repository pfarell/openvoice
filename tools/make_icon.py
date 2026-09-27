"""Generate the OpenVoice tray icon.

Draws a minimal white microphone (capsule body, U-shaped cradle, stem, base) on
a dark rounded square with QPainter and writes:

    assets/icon_64.png
    assets/icon_128.png
    assets/icon_256.png
    assets/icon.ico          multi-size: 16, 24, 32, 48, 64, 128, 256

Qt's ICO writer stores exactly one image per file, so a multi-size icon cannot
come from a single ``QImage.save(path, "ICO")``. Each size is rendered to a
temporary single-image ICO with Qt's writer and the image blobs are merged into
one multi-size container. If Qt cannot write ICO at all, the PNGs are still
written and the ICO is reported as missing rather than faked.

Input size: a fixed list of seven icon sizes. Time is O(sum(size^2)) pixels of
rasterisation, space is O(largest image); nothing scales with user data.

Run:
    .venv\\Scripts\\python.exe tools\\make_icon.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

# Platform is chosen when QGuiApplication is constructed; offscreen keeps this
# script runnable from a bare console with no window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QGuiApplication,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

PNG_SIZES = (64, 128, 256)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

BG_TOP = QColor("#2b3444")
BG_BOTTOM = QColor("#171c24")
FG = QColor("#ffffff")
HIGHLIGHT = QColor(255, 255, 255, 24)


def draw_icon(size: int) -> QImage:
    """Render the full icon (background + glyph) at ``size`` pixels square."""
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    s = float(size)

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    # --- rounded-square background, with a hairline highlight ---
    inset = s * 0.035
    square = QRectF(inset, inset, s - 2.0 * inset, s - 2.0 * inset)
    radius = square.width() * 0.235
    grad = QLinearGradient(square.topLeft(), square.bottomLeft())
    grad.setColorAt(0.0, BG_TOP)
    grad.setColorAt(1.0, BG_BOTTOM)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(grad))
    p.drawRoundedRect(square, radius, radius)

    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(HIGHLIGHT, max(1.0, s * 0.006)))
    p.drawRoundedRect(square, radius, radius)

    # --- microphone glyph (vertically centred in the square) ---
    cx = s * 0.5

    # capsule body
    cap_w = s * 0.255
    cap_h = s * 0.400
    cap_top = s * 0.155
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(FG)
    p.drawRoundedRect(
        QRectF(cx - cap_w * 0.5, cap_top, cap_w, cap_h), cap_w * 0.5, cap_w * 0.5
    )

    # U-shaped cradle: lower half of a circle, opening upward
    arc_r = s * 0.195
    arc_cy = s * 0.505
    arc_pen = QPen(FG, s * 0.072)
    arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(arc_pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(
        QRectF(cx - arc_r, arc_cy - arc_r, 2.0 * arc_r, 2.0 * arc_r),
        180 * 16,
        180 * 16,
    )

    # stem from the bottom of the cradle
    stem_pen = QPen(FG, s * 0.052)
    stem_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(stem_pen)
    p.drawLine(
        QPointF(cx, arc_cy + arc_r - arc_pen.widthF() * 0.15),
        QPointF(cx, s * 0.795),
    )

    # base
    base_pen = QPen(FG, s * 0.058)
    base_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(base_pen)
    p.drawLine(QPointF(cx - s * 0.125, s * 0.805), QPointF(cx + s * 0.125, s * 0.805))

    p.end()
    return img


def _single_ico_blob(img: QImage, tmpdir: Path, size: int):
    """Write one image with Qt's ICO writer and return its embedded blob.

    Returns ``(blob, width_byte, height_byte, bit_count)`` or ``None`` when Qt
    cannot write this format at all.
    """
    tmp = tmpdir / f"icon_{size}.ico"
    if not img.save(str(tmp), "ICO"):
        return None
    data = tmp.read_bytes()
    if len(data) < 22 or struct.unpack_from("<H", data, 2)[0] != 1:
        return None
    width, height, _colors, _res, _planes, bit_count, length, offset = struct.unpack_from(
        "<BBBBHHII", data, 6
    )
    blob = data[offset:offset + length]
    if len(blob) != length:
        return None
    return blob, width, height, bit_count


def write_ico(path: Path, images) -> bool:
    """Merge single-image ICO blobs into one multi-size ICO container."""
    entries = []
    blobs = []
    with tempfile.TemporaryDirectory(prefix="rocky_icon_") as td:
        tmpdir = Path(td)
        for size, img in images:
            parsed = _single_ico_blob(img, tmpdir, size)
            if parsed is None:
                return False
            blob, width, height, bit_count = parsed
            entries.append((width, height, bit_count, len(blob)))
            blobs.append(blob)

    count = len(entries)
    header = struct.pack("<HHH", 0, 1, count)  # reserved, type=icon, count
    offset = 6 + 16 * count
    directory = bytearray()
    for width, height, bit_count, length in entries:
        directory += struct.pack(
            "<BBBBHHII", width, height, 0, 0, 1, bit_count, length, offset
        )
        offset += length
    path.write_bytes(header + bytes(directory) + b"".join(blobs))
    return True


def _ensure_app() -> QGuiApplication:
    app = QGuiApplication.instance()
    if app is not None:
        return app
    try:
        return QGuiApplication(sys.argv)
    except Exception:  # pragma: no cover - platform fallback
        os.environ.pop("QT_QPA_PLATFORM", None)
        return QGuiApplication(sys.argv)


def main() -> int:
    _APP = _ensure_app()  # noqa: F841 - must outlive the painting calls
    ASSETS.mkdir(parents=True, exist_ok=True)

    for size in PNG_SIZES:
        out = ASSETS / f"icon_{size}.png"
        if not draw_icon(size).save(str(out), "PNG"):
            print(f"FAILED to write {out}", file=sys.stderr)
            return 1
        print(f"wrote {out} ({out.stat().st_size} bytes)")

    images = [(size, draw_icon(size)) for size in ICO_SIZES]
    ico = ASSETS / "icon.ico"
    if not write_ico(ico, images):
        print(
            "ICO writer unavailable: wrote the PNG sizes only; "
            "assets/icon.ico was NOT created",
            file=sys.stderr,
        )
        return 2

    printed = ", ".join(str(s) for s in ICO_SIZES)
    print(f"wrote {ico} ({ico.stat().st_size} bytes), sizes: {printed}")

    # Structural read-back: does an icon reader actually see every size?
    read_back = QIcon(str(ico)).availableSizes()
    seen = sorted({(s.width(), s.height()) for s in read_back})
    print(f"read back with QIcon: {seen}")
    if len(seen) != len(ICO_SIZES):
        print(
            f"WARNING: expected {len(ICO_SIZES)} sizes, reader saw {len(seen)}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
