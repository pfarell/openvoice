"""Mutation harness proving ``tests/test_vad.py`` discriminates.

Two broken replacements for ``audio.trim_silence`` are injected: one that
returns its input unchanged, and one that always returns an empty array. For
each mutant the shared case-2 predicate from ``test_vad`` is evaluated against
the mutant's output. If the predicate still passes against a broken function,
the test it belongs to is VOID because it cannot detect the breakage.

Exit 0 means DISCRIMINATING (both mutants fail the predicate); exit 1 means
VOID (at least one mutant passes).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import audio
import test_vad


def _mutant_identity(trim_input: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
    """Broken trim that always returns the input unchanged."""
    return np.asarray(trim_input, dtype=np.float32)


def _mutant_empty(trim_input: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
    """Broken trim that always returns an empty array."""
    return np.empty(0, dtype=np.float32)


def main() -> int:
    """Run the case-2 predicate against both mutants and report the verdict."""
    signal, tone_n, tone_energy = test_vad.case2_signal()
    original = audio.trim_silence
    baseline = original(signal)
    if not test_vad.case2_holds(baseline, signal, tone_n, tone_energy):
        print("VOID (positive control: real trim_silence fails the case-2 predicate)")
        return 1
    survivors = 0
    for name, mutant in (("identity", _mutant_identity), ("empty", _mutant_empty)):
        audio.trim_silence = mutant
        result = audio.trim_silence(signal)
        passed = test_vad.case2_holds(result, signal, tone_n, tone_energy)
        outcome = "passed" if passed else "failed"
        verdict = "VOID" if passed else "DISCRIMINATING"
        print(f"{name} mutant: case-2 assertion {outcome} -> {verdict}")
        survivors += int(passed)
    audio.trim_silence = original
    if survivors:
        print("VOID")
        return 1
    print("DISCRIMINATING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
