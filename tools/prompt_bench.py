"""A/B for accuracy levers on jargon-heavy dictation, measured not guessed.

Windows SAPI speaks two suites through the real Transcriber:
  tech  -- 12 sentences dense with CPU/GPU/ML jargon and slang, true refs
  ord   -- the 7 plain sentences from e2e_stt, to catch prompt drift

Variants differ only in config keys (initial_prompt, beam_size, vad_filter,
custom_vocabulary); the model loads once and the same audio feeds every
variant, so a delta is attributable to the variant, not to the decode path.

Usage:
    python tools/prompt_bench.py [--variants base,prompt,beam10]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bus import Bus  # noqa: E402
from config import Config  # noqa: E402
from e2e_stt import SENTENCES, sapi_wav, wer  # noqa: E402
from transcribe import Transcriber  # noqa: E402

TECH_PROMPT = (
    "Technical dictation about software and hardware. Terms such as CPU, GPU, "
    "machine learning, neural network, model, inference, latency, API, server, "
    "database, code, refactor, deploy and ship may appear. Keep normal "
    "punctuation and casing."
)

TECH_SENTENCES = [
    "The GPU is maxed out at ninety-nine percent while the CPU sits idle.",
    "We need to quantize the neural network before deploying it to the edge device.",
    "Fine-tuning the large language model requires a lot of VRAM.",
    "Push the refactor to the main branch after the CI pipeline passes.",
    "The API returns a JSON payload with a nested array of timestamps.",
    "Enable CUDA acceleration so the transformer runs on the tensor cores.",
    "Our latency budget is fifty milliseconds per inference request.",
    "The build is borked because the dependency tree has a circular reference.",
    "Spin up a Postgres instance and migrate the schema tonight.",
    "The transformer uses self-attention and positional embeddings.",
    "That migration is a real footgun, so write a rollback script first.",
    "Ship it, then we iterate on the feedback.",
]

VARIANTS = [
    ("base", {}),
    ("prompt", {"initial_prompt": TECH_PROMPT}),
    ("prompt+beam10", {"initial_prompt": TECH_PROMPT, "beam_size": 10}),
    ("beam10", {"beam_size": 10}),
    ("prompt+vad", {"initial_prompt": TECH_PROMPT, "vad_filter": True}),
    ("prompt-no-vocab", {"initial_prompt": TECH_PROMPT, "custom_vocabulary": ""}),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(name for name, _ in VARIANTS),
        help="comma-separated subset of variant names",
    )
    args = parser.parse_args()
    wanted = {name.strip() for name in args.variants.split(",") if name.strip()}
    variants = [(n, o) for n, o in VARIANTS if n in wanted]
    if not variants:
        print(f"no known variant in {args.variants!r}")
        return 2

    from faster_whisper.audio import decode_audio

    config = Config()
    live_vocab = config.get("custom_vocabulary") or ""
    config.set_volatile("language", "en")
    base = {
        "initial_prompt": "",
        "beam_size": 5,
        "vad_filter": False,
        "custom_vocabulary": live_vocab,
    }

    bus = Bus()
    transcriber = Transcriber(bus, config)
    print(f"engine load (live config, vocab {len(live_vocab.split(','))} terms)...")
    load_start = time.perf_counter()
    transcriber.load()
    print(f"engine: {transcriber.engine_name}  loaded in {time.perf_counter() - load_start:.2f} s")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clips: list[tuple[str, str, object]] = []
        for suite, sentences in (("tech", TECH_SENTENCES), ("ord", SENTENCES)):
            for i, sentence in enumerate(sentences, 1):
                wav = tmp_path / f"{suite}{i:02d}.wav"
                sapi_wav(sentence, wav)
                clips.append((suite, sentence, decode_audio(str(wav), sampling_rate=16000)))
        print(f"clips: {len(clips)} spoken via SAPI and decoded once")
        print()

        for name, overrides in variants:
            settings = {**base, **overrides}
            for key, value in settings.items():
                config.set_volatile(key, value)
            transcriber._beam = int(settings["beam_size"])

            rows = {"tech": [], "ord": []}
            audio_seconds = decode_seconds = 0.0
            print(f"== {name}  (beam {settings['beam_size']}, vad {settings['vad_filter']}, "
                  f"prompt {'yes' if settings['initial_prompt'] else 'no'}, "
                  f"vocab {'yes' if settings['custom_vocabulary'] else 'no'})")
            for suite, sentence, audio in clips:
                seconds = float(audio.size) / 16000.0
                start = time.perf_counter()
                text = transcriber.transcribe(audio)
                elapsed = time.perf_counter() - start
                audio_seconds += seconds
                decode_seconds += elapsed
                score = wer(sentence, text)
                rows[suite].append((sentence, text, score))
                if score:
                    print(f"  {suite} WER {score * 100:5.1f}%  ref: {sentence}")
                    print(f"  {'':>9}    got: {text or '<empty>'}")
            for suite in ("tech", "ord"):
                scores = [score for _, _, score in rows[suite]]
                exact = sum(1 for s in scores if s == 0.0)
                mean = sum(scores) / len(scores)
                print(f"  {suite}: mean WER {mean * 100:5.2f}%  exact {exact}/{len(scores)}")
            print(f"  decode {decode_seconds:.1f}s for {audio_seconds:.1f}s audio "
                  f"(RTF {decode_seconds / audio_seconds:.2f})")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
