"""Pure data/helper tests for ``tray.py``'s language support.

No ``Tray`` is constructed: it needs a live system tray and would test Qt rather
than our logic. These checks cover the frozen ``LANGUAGES`` table and the pure
``language_label`` helper. ``tray`` imports Qt at module level, which is fine
here - the suite already runs with the ``qapp`` fixture and ``QT_QPA_PLATFORM``
set by ``conftest.py``.
"""

from __future__ import annotations

import pytest

from tray import LANGUAGES, language_label


@pytest.mark.parametrize(
    "code, expected",
    [
        ("auto", "Auto-detect"),
        ("AUTO", "Auto-detect"),
        ("en", "English"),
        ("EN-us", "English"),
        ("en-US", "English"),
        ("en_US", "English"),
        ("id", "Indonesian"),
        ("zh", "Chinese (Mandarin)"),
        ("fil", "Filipino"),
        ("xx", "xx"),
        ("", "Unknown"),
        (None, "Unknown"),
    ],
)
def test_language_label(code, expected):
    assert language_label(code) == expected


def test_every_language_code_is_unique():
    codes = [code for code, _ in LANGUAGES]
    assert len(codes) == len(set(codes)), f"duplicate language codes: {codes}"


def test_every_label_is_non_empty():
    empty = [code for code, label in LANGUAGES if not str(label).strip()]
    assert not empty, f"language(s) with an empty label: {empty}"


def test_english_is_present():
    assert any(code == "en" for code, _ in LANGUAGES)


def test_auto_detect_is_present_and_first():
    """The menu defaults to auto, so it must be the first entry."""
    assert LANGUAGES[0][0] == "auto"
    assert LANGUAGES[0][1] == "Auto-detect"
