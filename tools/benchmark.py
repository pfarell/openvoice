"""A diagnostic dictation benchmark.

Eight probes, each isolating a different failure class, so the result says WHAT
to fix rather than giving one opaque number. Record once; re-score as often as
you like without speaking again.

    python tools/benchmark.py                 record all probes, then score
    python tools/benchmark.py --list          just print the sentences
    python tools/benchmark.py --score-only    re-score the most recent run

Every probe writes its audio AND its reference text next to each other, so a
disagreement can always be traced back to the exact second of audio that caused
it. Scoring runs entirely offline.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16000
RUNS_DIR = ROOT / "tools" / "benchmark_runs"

PROBES = [
    (
        "01-prose",
        "ordinary English, no names or numbers",
        "The meeting is tomorrow afternoon and I will bring the documents you asked for.",
    ),
    (
        "02-first-word",
        "known weak spot: a fricative at the very start",
        "Send the invoice to the accounting department tomorrow morning.",
    ),
    (
        "03-proper-nouns",
        "the vocabulary list",
        "Open NovaGate and check whether the Zenor integration finished before the Vint release.",
    ),
    (
        "04-numbers",
        "numbers, percentage, range, thousands separator",
        "The budget is about five thousand dollars, the margin is twelve point five percent, "
        "and the range is twenty to thirty units.",
    ),
    (
        "05-jargon",
        "technical terms",
        "The CadQuery script exports the bracket while the Luau code handles the Roblox shop.",
    ),
    (
        "06-homophones",
        "words that sound alike",
        "They're going to their office to see if the two of them can work there too.",
    ),
    (
        "07-punctuation",
        "spoken punctuation commands",
        "First item comma second item comma and third item period new line Done.",
    ),
    (
        "08-long-form",
        "six sentences in one breath",
        "I finished the review of the NovaGate dashboard this morning. "
        "The Zenor integration is working, but the sync is slow. "
        "It takes about four seconds to load each screen. "
        "I think we should cache the results for thirty seconds. "
        "That would cut the load time by roughly eighty percent. "
        "Let me know what you think before Friday.",
    ),
    (
        "09-vocabulary",
        "a configured vocabulary, said once, so a miss names itself",
        "NovaGate, Zenor, Kestrel, Vint, Quill, Luau, opencode.",
    ),
]

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000,
}
NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)"
)
# The separator only appears BETWEEN words, never after the last one: a trailing
# space inside the match would be swallowed by the replacement and weld the
# following word onto the number ("five thousand dollars" -> "5000dollars").
# The closing \b also stops "ten" from matching inside "thirteen".
NUMBER_RUN = re.compile(rf"\b{NUMBER_WORD}(?:[\s-]+{NUMBER_WORD})*\b")

CONTRACTIONS = {
    "they're": "they are", "it's": "it is", "i'm": "i am", "don't": "do not",
    "can't": "cannot", "won't": "will not", "that's": "that is",
    "there's": "there is", "we're": "we are", "you're": "you are",
}


def _words_to_number(match: re.Match) -> str:
    """Convert a run of number words to digits. `findall` rather than `split`, because
    the run pattern can capture a trailing space and an empty token would abort the
    conversion and leave the words as they were."""
    total = 0
    current = 0
    parts = re.findall(r"[a-z]+", match.group(0))
    if not parts:
        return match.group(0)
    for part in parts:
        if part not in NUMBER_WORDS:
            return match.group(0)
        value = NUMBER_WORDS[part]
        if value >= 100:
            current = max(current, 1) * value
        else:
            current += value
        if value >= 1000:
            total += current
            current = 0
    return str(total + current)


def canonical(text: str) -> str:
    """Fold formatting so the score measures wording, not typing style.

    Numbers become digits on both sides, so "five thousand dollars" and "$5,000"
    compare equal, as do "12.5 percent" and "12.5%". O(n) over the tokens.
    """
    work = text.lower()
    for short, long_form in CONTRACTIONS.items():
        work = work.replace(short, long_form)
    work = NUMBER_RUN.sub(_words_to_number, work)
    work = work.replace("%", " percent ")
    work = work.replace("$", "").replace("£", "").replace("€", "").replace(",", "")
    work = re.sub(r"\bpoint\b", " ", work)
    work = re.sub(r"(\d)\.(\d)", r"\1 \2", work)
    # Currency words have to go too, or "$5,000" (symbol, no word) could never equal
    # "five thousand dollars" (word, no symbol). The post-ITN score below still
    # compares the written forms exactly, so a dropped "$" is not hidden.
    work = re.sub(r"\b(dollars?|euros?|pounds?|cents?|rupiah|yen)\b", " ", work)
    work = re.sub(r"[^a-z0-9\s]", " ", work)
    return " ".join(work.split())


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, list[tuple]]:
    """Levenshtein over words, O(r*h) time and O(h) space.

    Returns the rate and the edit operations, so every error can be shown rather
    than only counted.
    """
    ref = canonical(reference).split()
    hyp = canonical(hypothesis).split()
    if not ref:
        return (0.0 if not hyp else 1.0), []
    rows = len(ref) + 1
    cols = len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + cost,
            )

    edits: list[tuple] = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            edits.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            edits.append(("missed", ref[i - 1], ""))
            i -= 1
        else:
            edits.append(("added", "", hyp[j - 1]))
            j -= 1
    edits.reverse()
    return distance[rows - 1][cols - 1] / len(ref), edits


def tone(freq: int, ms: int) -> None:
    try:
        import winsound

        winsound.Beep(freq, ms)
    except Exception:
        pass


def record_all(seconds: float, countdown: float) -> Path:
    import sounddevice as sd

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    device = sd.default.device[0]
    try:
        name = sd.query_devices(device)["name"]
    except Exception:
        name = "system default"
    print(f"microphone : [{device}] {name}")
    print(f"run        : {run_dir}")
    print()
    print("Read each sentence out loud, naturally, then wait for the low tone.")
    print("A high double-tone means SPEAK NOW. You get a fresh countdown each time.")
    print()

    for name, purpose, reference in PROBES:
        print()

        print("=" * 78)
        print(f"  {name}  ({purpose})")
        print("=" * 78)
        print()
        print(f"    {reference}")
        print()

        for remaining in range(int(max(countdown, 0)), 0, -1):
            print(f"    starting in {remaining}...", flush=True)
            tone(520, 70)
            time.sleep(1.0)
        tone(1000, 130)
        time.sleep(0.15)
        tone(1000, 130)
        print("    >>> SPEAK NOW <<<", flush=True)

        words = len(reference.split())
        budget = max(seconds, words * 0.55 + 1.5)
        try:
            captured = sd.rec(
                int(budget * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            print(f"    recording failed: {exc}")
            continue
        tone(380, 250)

        audio = captured.reshape(-1)
        np.save(run_dir / f"{name}.npy", audio)
        (run_dir / f"{name}.ref.txt").write_text(reference, encoding="utf-8")
        print(f"    captured {audio.size / SAMPLE_RATE:.2f} s")

    print()
    print(f"recorded {len(PROBES)} probes into {run_dir}")
    return run_dir


def score(run_dir: Path) -> int:
    from faster_whisper.audio import decode_audio

    import audio as audio_module
    import formatting
    import transcribe
    from bus import Bus
    from config import Config

    config = Config()
    # Volatile on purpose: the English benchmark must not rewrite the user's own
    # language setting in %APPDATA% as a side effect.
    config.set_volatile("language", "en")
    engine = transcribe.resolve_engine(config)
    print(f"engine      : {engine[0]} on {engine[1]}/{engine[2]}")
    terms = [t.strip() for t in transcribe.vocabulary_for(config).split(",") if t.strip()]
    print(f"vocabulary  : {len(terms)} terms")
    print()

    transcriber = transcribe.Transcriber(Bus(), config)
    transcriber.load()

    rows = []
    detail: list[str] = []

    for name, purpose, reference in PROBES:
        audio_path = run_dir / f"{name}.npy"
        if not audio_path.exists():
            continue
        captured = np.load(audio_path)
        processed = audio_module.normalize_level(
            audio_module.trim_silence(
                audio_module.highpass_rumble(captured), adaptive=True
            )
        )
        started = time.perf_counter()
        result = transcriber.transcribe_audio(processed)
        elapsed = time.perf_counter() - started

        if not result.accepted:
            rows.append((name, 0.0, 1.0, f"REJECTED: {result.reject_reason}"))
            detail.append(
                f"  {name}\n    reference: {reference}\n"
                f"    REJECTED : {result.reject_reason}\n"
            )
            continue

        written = formatting.normalise(result.text, config)
        rate_wording, edits = word_error_rate(reference, written)
        rate_exact, _ = word_error_rate(
            formatting.normalise(reference, config), written
        )
        accuracy = (1.0 - rate_wording) * 100.0
        rows.append((name, accuracy, elapsed, written))
        detail.append(
            f"  {name}  ({purpose})\n"
            f"    reference: {reference}\n"
            f"    heard    : {written}\n"
            f"    wording accuracy {accuracy:5.1f}%   after ITN {((1 - rate_exact) * 100):5.1f}%\n"
        )
        for kind, want, got in edits:
            if kind == "sub":
                detail.append(f"      substituted  {want!r} -> {got!r}")
            elif kind == "missed":
                detail.append(f"      missed       {want!r}")
            else:
                detail.append(f"      added        {got!r}")
        detail.append("")

    if not rows:
        print("nothing to score in that directory")
        return 1

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    for name, accuracy, elapsed, note in rows:
        bar = "#" * int(max(accuracy, 0) / 4)
        print(f"  {name:<16} {accuracy:5.1f}%  {elapsed:5.2f}s  {bar}")
    print()
    accepted = [r for r in rows if not str(r[3]).startswith("REJECTED")]
    if accepted:
        mean = sum(r[1] for r in accepted) / len(accepted)
        print(f"  probes scored      : {len(accepted)}/{len(rows)}")
        print(f"  MEAN WORD ACCURACY : {mean:.1f}%")
    print()

    print("=" * 78)
    print("DETAIL — every disagreement, so you can see what to fix")
    print("=" * 78)
    print()
    print("\n".join(detail))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--run", default=None, help="path to a previous run directory")
    args = parser.parse_args()

    if args.list:
        for name, purpose, reference in PROBES:
            print(f"\n{name}  ({purpose})\n  {reference}")
        return 0

    if args.score_only or args.run:
        if args.run:
            run_dir = Path(args.run)
        else:
            existing = sorted(glob.glob(str(RUNS_DIR / "*")))
            if not existing:
                print("no saved runs; run without --score-only to record one")
                return 1
            run_dir = Path(existing[-1])
        print(f"scoring    : {run_dir}")
        return score(run_dir)

    run_dir = record_all(args.seconds, args.countdown)
    print()
    return score(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
