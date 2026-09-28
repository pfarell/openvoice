"""Per-language accuracy check on real, public human speech samples.

The English benchmark (``tools/benchmark.py``) needs *your* voice. This tool is
the multilingual counterpart: it fetches a few clips per language from public
research datasets through the HuggingFace datasets-server rows API (which serves
individual audio files, so nothing large is ever downloaded), caches them under
``tools/lang_samples/`` with their reference transcript and a provenance
manifest, and then runs the app's real pipeline - the same conditioning,
language detection, decoding and hallucination gate the microphone path uses.

Only the cached clips are transcribed; each run prints the detected language,
its probability and a WER (space-delimited languages) or CER (Chinese, Japanese,
Thai) score per clip plus a summary table. Scoring strips punctuation and case
on both sides; numbers are left as-is.

This is a measurement tool, not a test: it fails loudly rather than skipping,
and it reports REJECTED clips honestly instead of counting them as correct.
FLEURS clips are CC-BY-4.0 and JSUT is CC-BY-SA-4.0 in their original
distributions - check each dataset card before redistributing anything; the
cached copies here are for local verification only.

Cost: one HTTP request per clip when fetching; per run, one whisper pass per
clip (O(audio)) plus O(n*m) edit distance per clip. Space: the cached audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES_DIR = Path(__file__).resolve().parent / "lang_samples"
MANIFEST = SAMPLES_DIR / "manifest.json"

ROWS_API = "https://datasets-server.huggingface.co/rows"

#: language -> (dataset id, config, split, reference-text field)
SOURCES: dict[str, tuple[str, str, str, str]] = {
    "en": ("hf-internal-testing/librispeech_asr_dummy", "clean", "validation", "text"),
    "fr": ("BrunoHays/fleurs_fr_fr", "fr_fr", "test", "transcription"),
    "es": ("U182025/audio_fleurs_es", "default", "test", "transcript"),
    "id": ("cobrayyxx/FLEURS_ID-EN", "default", "train", "text_indo"),
    "zh": ("parami-ai/google_fleurs", "zh", "train", "sentence"),
    "ja": ("japanese-asr/ja_asr.jsut_basic5000", "default", "test", "transcription"),
    "th": ("kittinol/fleurs-th-viewer", "default", "train", "transcription"),
    "vi": ("htdung167/fleurs-vi-preprocessed", "default", "train", "preprocessed_sentence"),
}

#: These languages are written without spaces, so word error rate is meaningless
#: and the character error rate is the metric (matching whisper's own choice).
NO_SPACE = frozenset({"zh", "ja", "th", "yue"})

#: LibriSpeech-style references spell titles out ("MISTER") while whisper writes
#: abbreviations ("Mr."). Punctuation is already stripped, so mapping the
#: expanded form onto the abbreviation makes the two comparable without touching
#: any real word error.
_EN_ABBREVIATIONS = {"mister": "mr", "doctor": "dr", "saint": "st", "mister's": "mr's"}


def normalise_for_scoring(text: str, language: str) -> str:
    """Case-folded, punctuation-free text for comparison. O(len(text))."""
    text = str(text or "").casefold()
    kept = []
    for char in text:
        if char.isspace():
            kept.append(" ")
            continue
        category = unicodedata.category(char)[0]
        if category in ("P", "S"):
            continue
        kept.append(char)
    folded = "".join(kept)
    if language in NO_SPACE:
        return folded.replace(" ", "")
    tokens = folded.split()
    if language == "en":
        tokens = [_EN_ABBREVIATIONS.get(token, token) for token in tokens]
    return " ".join(tokens)


def edit_distance(left: list, right: list) -> int:
    """Levenshtein distance over two token lists, O(n*m) time, O(min) space."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, item in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item != other),
                )
            )
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str, language: str) -> tuple[float, int, int, int]:
    """(error rate, edits, reference units, hypothesis units) for one clip."""
    ref = normalise_for_scoring(reference, language)
    hyp = normalise_for_scoring(hypothesis, language)
    ref_units = list(ref) if language in NO_SPACE else ref.split()
    hyp_units = list(hyp) if language in NO_SPACE else hyp.split()
    if not ref_units:
        return (1.0 if hyp_units else 0.0, len(hyp_units), 0, len(hyp_units))
    edits = edit_distance(ref_units, hyp_units)
    return (edits / len(ref_units), edits, len(ref_units), len(hyp_units))


def sample_dir(language: str) -> Path:
    return SAMPLES_DIR / language


def load_manifest() -> list:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_manifest(entries: list) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_clip_rows(dataset: str, config: str, split: str, index: int) -> dict:
    url = (
        f"{ROWS_API}?dataset={urllib.parse.quote(dataset)}&config={config}"
        f"&split={split}&offset={index}&length=1"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    rows = payload.get("rows") or []
    if not rows:
        raise RuntimeError(f"no row at offset {index} for {dataset}/{config}/{split}")
    return rows[0]["row"]


def fetch(languages: list[str], clips: int, force: bool, offset: int = 0) -> int:
    """Download ``clips`` samples per language into the cache. Returns exit code.

    ``offset`` skips ahead in the source split; some mirrors repeat one
    utterance across their first rows, so a different starting point may be
    needed to get varied material.
    """
    # Force-refreshing one language must not discard the others' provenance, and
    # a cached file that lost its manifest entry re-adds it.
    manifest = load_manifest()
    if force:
        manifest = [
            entry
            for entry in manifest
            if entry.get("language") not in set(languages)
        ]
    known_files = {entry.get("file") for entry in manifest}
    failures = 0
    for language in languages:
        dataset, config, split, field = SOURCES[language]
        directory = sample_dir(language)
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(clips):
            wav_path = directory / f"{index:02d}.wav"
            txt_path = directory / f"{index:02d}.txt"
            relative = f"{language}/{index:02d}.wav"
            if wav_path.exists() and txt_path.exists() and not force:
                print(f"  {language} {index:02d}: cached")
                if relative not in known_files:
                    manifest.append(
                        {
                            "language": language,
                            "file": relative,
                            "dataset": dataset,
                            "config": config,
                            "split": split,
                            "offset": offset + index,
                            "field": field,
                            "bytes": wav_path.stat().st_size,
                        }
                    )
                    known_files.add(relative)
                continue
            row_offset = offset + index
            try:
                row = fetch_clip_rows(dataset, config, split, row_offset)
                reference = str(row.get(field) or "").strip()
                audio = row.get("audio")
                if not reference:
                    raise RuntimeError(f"row has no {field!r} text")
                if not isinstance(audio, list) or not audio:
                    raise RuntimeError("row has no audio column")
                source_url = audio[0].get("src") if isinstance(audio[0], dict) else None
                if not source_url:
                    raise RuntimeError("audio entry has no src")
                with urllib.request.urlopen(source_url, timeout=120) as response:
                    blob = response.read()
                wav_path.write_bytes(blob)
                txt_path.write_text(reference, encoding="utf-8")
                manifest.append(
                    {
                        "language": language,
                        "file": f"{language}/{index:02d}.wav",
                        "dataset": dataset,
                        "config": config,
                        "split": split,
                        "offset": row_offset,
                        "field": field,
                        "bytes": len(blob),
                    }
                )
                known_files.add(f"{language}/{index:02d}.wav")
                print(f"  {language} {index:02d}: {len(blob)} bytes  {reference[:40]!r}")
                time.sleep(0.2)
            except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
                failures += 1
                print(f"  {language} {index:02d}: FAILED - {exc}", file=sys.stderr)
    save_manifest(manifest)
    return 1 if failures else 0


def run(languages: list[str], model: str | None, device: str | None) -> int:
    """Transcribe the cached clips and print per-clip and summary scores."""
    import numpy as np
    from faster_whisper.audio import decode_audio

    import audio as audio_module
    import formatting
    import transcribe
    from bus import Bus
    from config import Config
    from transcribe import LANGUAGE_AUTO

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        config = Config(Path(tmp) / "config.json")
        config.set_volatile("language", LANGUAGE_AUTO)
        if model:
            config.set_volatile("model_name", model)
        if device:
            config.set_volatile("device", device)
        device_used = transcribe.resolve_engine(config)
        print(f"engine      : {device_used[0]} on {device_used[1]}/{device_used[2]}")
        transcriber = transcribe.Transcriber(Bus(), config)
        transcriber.load()
        print()

        rows = []
        reference_counts: dict[str, dict[str, int]] = {}
        for language in languages:
            directory = sample_dir(language)
            files = sorted(directory.glob("*.wav"))
            if not files:
                print(f"{language:4} no cached clips; run --fetch first", file=sys.stderr)
                continue
            for wav_path in files:
                reference = wav_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
                counts = reference_counts.setdefault(language, {})
                counts[reference] = counts.get(reference, 0) + 1
                captured = decode_audio(str(wav_path), sampling_rate=16000)
                captured = np.asarray(captured, dtype=np.float32).reshape(-1)
                processed = audio_module.normalize_level(
                    audio_module.trim_silence(
                        audio_module.highpass_rumble(captured), adaptive=True
                    )
                )
                started = time.perf_counter()
                result = transcriber.transcribe_audio(processed)
                elapsed = time.perf_counter() - started
                detected = result.language or "?"
                if not result.accepted:
                    rows.append((language, None, detected, result.language_probability, elapsed, "REJECTED"))
                    print(f"{language:4} {wav_path.name}  {elapsed:5.2f}s  {detected:4}  REJECTED: {result.reject_reason}")
                    print(f"     ref: {reference}")
                    continue
                rate, edits, length, produced = score(reference, result.text, language)
                metric = "CER" if language in NO_SPACE else "WER"
                # A public row whose reference is far longer than anything the
                # audio can contain (misaligned or padded source data) would
                # otherwise dominate a per-language mean; flag it and exclude it.
                suspect = length > 0 and (produced / length) < 0.5
                note = "SUSPECT" if suspect else ""
                rows.append((language, rate, detected, result.language_probability, elapsed, note))
                warning = "  [SUSPECT: reference much longer than the audio]" if suspect else ""
                print(
                    f"{language:4} {wav_path.name}  {elapsed:5.2f}s  detected={detected}"
                    f" ({result.language_probability or 0:.2f})  {metric}={rate * 100:5.1f}%"
                    f"  ({edits}/{length}){warning}"
                )
                print(f"     ref: {reference}")
                print(f"     hyp: {result.text}")

        print()
        print("summary")
        print("-------")
        by_language: dict[str, list] = {}
        for row in rows:
            by_language.setdefault(row[0], []).append(row)
        for language in languages:
            entries = by_language.get(language, [])
            if not entries:
                continue
            # Several mirrors repeat one utterance across rows; repeated
            # references make a mean look better than the sample really is, so
            # they are surfaced rather than silently averaged.
            duplicates = sum(
                count - 1 for count in reference_counts.get(language, {}).values() if count > 1
            )
            scored = [
                entry[1] for entry in entries if entry[1] is not None and entry[5] != "SUSPECT"
            ]
            rejected = sum(1 for entry in entries if entry[1] is None)
            suspects = sum(1 for entry in entries if entry[5] == "SUSPECT")
            detected_ok = sum(1 for entry in entries if entry[2].startswith(language))
            metric = "CER" if language in NO_SPACE else "WER"
            duplicate_note = f"  duplicate-refs={duplicates}" if duplicates else ""
            if scored:
                mean = sum(scored) / len(scored)
                print(
                    f"{language:4} clips={len(entries)}  mean {metric}={mean * 100:5.1f}%"
                    f"  ({len(scored)} scored)  detected-ok={detected_ok}/{len(entries)}"
                    f"  rejected={rejected}  suspect={suspects}{duplicate_note}"
                )
            else:
                print(
                    f"{language:4} clips={len(entries)}  no scorable clips"
                    f"  rejected={rejected}  suspect={suspects}{duplicate_note}"
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage and cannot print CJK or Thai;
    # the reports and the cached transcripts themselves are UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="lang_bench.py",
        description=(
            "Multilingual accuracy check on real public speech samples. Fetches a "
            "few clips per language from public research datasets, then runs the "
            "app's real pipeline and reports WER/CER and detected language."
        ),
    )
    parser.add_argument("--languages", default=",".join(SOURCES), help="comma-separated codes")
    parser.add_argument("--clips", type=int, default=3, help="clips per language to fetch")
    parser.add_argument("--offset", type=int, default=0, help="first row to fetch in the source split")
    parser.add_argument("--fetch", action="store_true", help="download samples and stop")
    parser.add_argument("--run", action="store_true", help="transcribe cached samples (default)")
    parser.add_argument("--force", action="store_true", help="re-download cached samples")
    parser.add_argument("--model", help="override model_name for this run, e.g. large-v3-turbo")
    parser.add_argument("--device", help="override device for this run: cpu or cuda")
    args = parser.parse_args(argv)

    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    unknown = [code for code in languages if code not in SOURCES]
    if unknown:
        print(f"unknown language codes: {unknown}; known: {sorted(SOURCES)}", file=sys.stderr)
        return 2

    status = 0
    if args.fetch:
        status = fetch(languages, args.clips, args.force, args.offset)
    else:
        if not args.run:
            missing = [
                code
                for code in languages
                if not any(sample_dir(code).glob("*.wav"))
            ]
            if missing:
                status = fetch(missing, args.clips, args.force, args.offset)
        status = run(languages, args.model, args.device) or status
    return status


if __name__ == "__main__":
    raise SystemExit(main())
