"""Console-free launcher for OpenVoice.

Windows runs a ``.pyw`` with ``pythonw.exe``, so double-clicking this file starts
``app.py`` from its own directory without a console window. The directory is put
on ``sys.path`` first, then ``runpy`` executes ``app.py`` as ``__main__``.

Input size: one file. O(1) time and space.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app.py"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# app.py is written to expect its own directory as the working directory.
try:
    os.chdir(HERE)
except OSError:
    pass

if not APP.exists():
    # No console here (pythonw), but the message reaches any redirected stderr.
    print(f"OpenVoice: {APP} does not exist yet", file=sys.stderr)
    raise SystemExit(1)

runpy.run_path(str(APP), run_name="__main__")
