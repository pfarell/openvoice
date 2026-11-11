"""Speech-to-text via faster-whisper, fully local.

Model load is O(model size) and happens once. Per-call cost is dominated by the
whisper encoder/decoder over the clip, which is roughly O(audio_seconds) with a
large constant; the joining and gating loops are O(segments). Language detection
is one extra O(1) encoder pass, and phrase folding is O(len(text)). Nothing here
scales super-linearly with audio length.
"""

from __future__ import annotations

import bisect
import logging
import math
import re
import sys
import threading
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("transcribe")

SAMPLE_RATE = 16000
EMPTY_AUDIO_FLOOR = 1e-4

_DLL_HANDLES: list = []

#: No vocabulary ships with the app. The words that need rescuing are the
#: speaker's own, so a generic list would only be noise; terms come from the
#: tray editor or from the CorrectionLearner. Keep any list SHORT: measured on
#: a real 3-sentence dictation, a 29-term list scored 48.1% WER while a 10-term
#: list scored 37.0% on the same audio. A long prompt skews the language model
#: and damages ORDINARY words, so add terms deliberately, not pre-emptively.
DEFAULT_VOCABULARY = ""
# Average WER on the HuggingFace Open ASR Leaderboard English short-form set:
# turbo 7.83, large-v3 7.44, distil-large-v3 7.21. Turbo trades accuracy for
# speed, and this machine has speed to spare (RTF 0.07), so prefer v3. The
# gap is much larger on noisy and accented speech (VoxPopuli: turbo 11.87 vs
# v3 9.54), which is the case that matters here.
V3_MODEL = "Systran/faster-whisper-large-v3"
DISTIL_MODEL = "Systran/faster-distil-whisper-large-v3"
TURBO_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
SMALL_MODEL = "Systran/faster-whisper-small.en"
# The multilingual sibling of SMALL_MODEL. small.en is faster and more accurate
# on English, but it cannot produce any other language at all: it reports
# supported_languages == ["en"]. When the configured language is not exactly
# English, the CPU fallback must therefore be this model.
SMALL_MULTI_MODEL = "Systran/faster-whisper-small"

# The sentinel meaning "let the model decide", shared with config.DEFAULTS.
LANGUAGE_AUTO = "auto"

DEFAULT_BEAM = 5
_FALLBACK_BEAM = 1

# A window longer than 30 s silently degrades: whisper was trained on 30 s
# chunks and slides a 30 s window, so a bad window poisons its successor.
MAX_WINDOW_SECONDS = 30.0

# Longest piece handed to whisper in one decode, and how far either side of an
# even split the cut may move to find a pause. Decoding a full 30 s window
# without timestamps drops whole sentences (measured by tools/long_bench.py),
# and a fixed 30 s step cuts words in half, so long clips are split into even
# pieces at their quietest moment instead. Cutting at the latest pause rather
# than near an even split was measured worse (5.83% vs 3.85% WER): it makes
# pieces close to 28 s, and those are the ones whisper truncates.
CHUNK_SECONDS = 28.0
CUT_SEARCH_SECONDS = 10.0

# A piece whose words per voiced second fall below this share of its siblings'
# median was truncated: prompted with the hotwords sentence, whisper sometimes
# ends a piece early and skips its last sentences (laptop clip 36 in
# tools/long_bench.py: 16 words with hotwords, 47 without). Healthy pieces
# measured 0.67 or more of the median; truncated ones 0.37-0.52. Such a piece is
# decoded again on its own without hotwords, and the retry is kept only when it
# recovers a clear number of words.
DROP_RATE_RATIO = 0.6
DROP_MIN_VOICED_SECONDS = 3.0
DROP_MIN_GAIN = 0.15


def _smoothed_energy(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mean power of 10 ms frames, smoothed over 100 ms. Time and space O(n)."""
    hop = sample_rate // 100
    frames = audio.size // hop
    if frames == 0:
        return np.zeros(0)
    energy = np.square(audio[: frames * hop].astype(np.float64)).reshape(frames, hop).mean(axis=1)
    return np.convolve(energy, np.ones(10) / 10.0, mode="same")


def voiced_frames(smooth: np.ndarray) -> np.ndarray:
    """Boolean mask of 10 ms frames clearly above the clip's noise floor.

    A clip without a quiet floor (four times below its median) has no pauses
    to tell apart, so every frame counts as voiced. Time and space O(n).
    """
    if smooth.size == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(smooth, 1))
    middle = float(np.median(smooth))
    if middle <= 4.0 * floor:
        return np.ones(smooth.size, dtype=bool)
    return smooth > floor + 0.15 * (middle - floor)


def silence_cuts(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float = CHUNK_SECONDS,
    search_seconds: float = CUT_SEARCH_SECONDS,
) -> list[float] | None:
    """faster-whisper ``clip_timestamps`` that split ``audio`` at pauses.

    Returns None when the clip fits one piece. Otherwise a flat
    ``[start, end, start, end, ...]`` list in seconds of the fewest pieces of
    at most ``max_seconds``, each cut aimed at an even split of what is left
    and moved to the quietest 100 ms within ``search_seconds`` of that aim.
    Time and space O(n).
    """
    smooth = _smoothed_energy(audio, sample_rate)
    frames = smooth.size
    total = audio.size / sample_rate
    if total <= max_seconds or frames == 0:
        return None
    limit = int(max_seconds * 100)
    reach = max(1, int(search_seconds * 100))
    points = [0.0]
    cursor = 0
    while frames - cursor > limit:
        remaining = frames - cursor
        count = math.ceil(remaining / limit)
        target = cursor + remaining // count
        low = max(cursor + 1, target - reach // 2, frames - (count - 1) * limit)
        high = min(cursor + limit, target + reach // 2)
        cut = low + int(np.argmin(smooth[low : high + 1]))
        points.extend([cut / 100.0, cut / 100.0])
        cursor = cut
    points.append(total)
    return points


def truncated_pieces(audio: np.ndarray, cuts: list[float], words: list[int]) -> list[int]:
    """Indices of pieces whose decode is much sparser than their siblings'.

    ``words`` holds the decoded word count of each piece of ``cuts``. Each
    piece is compared with the median rate of the other pieces, because with
    two pieces a median that includes the piece itself hides the drop. Time
    O(n + p^2 log p) for n samples and p pieces; p is the dictation length over
    ~25 s, so the all-pairs part stays tiny. Space O(n).
    """
    voiced = voiced_frames(_smoothed_energy(audio))
    spans = list(zip(cuts[0::2], cuts[1::2]))
    seconds = [float(voiced[int(a * 100) : int(b * 100)].sum()) / 100.0 for a, b in spans]
    rates = [
        count / voiced_seconds if voiced_seconds >= DROP_MIN_VOICED_SECONDS else None
        for count, voiced_seconds in zip(words, seconds)
    ]
    flagged = []
    for index, rate in enumerate(rates):
        others = [other for position, other in enumerate(rates) if position != index and other is not None]
        if rate is not None and others and rate < DROP_RATE_RATIO * float(np.median(others)):
            flagged.append(index)
    return flagged


# Whisper hallucinates on non-speech. Barański 2025 (arXiv:2501.11378) measured
# 40.3% of 301,317 non-speech clips producing text, and the training-data
# artefacts below dominate the output.
HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "thanks for listening",
    "please subscribe",
    "subscribe to my channel",
    "subtitles by",
    "amara.org",
    "transcription by",
    "you",
)

# The training-data artefacts whisper emits on non-speech are language-specific:
# an English list catches none of the Chinese "thanks for watching" clones. These
# are the phrases observed on real non-speech clips, kept raw and folded at match
# time so punctuation and spacing cannot hide them.
HALLUCINATIONS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "zh": (
        "谢谢观看",
        "謝謝觀看",
        "感谢观看",
        "下集预告",
        "别忘了订阅",
        "记得订阅",
        "大家好 欢迎来到我的频道",
        "字幕由amara org社区提供",
        "本集完",
        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    ),
    "ja": (
        "ご視聴ありがとうございました",
        "見てくれてありがとう",
    ),
    "th": (
        "ขอบคุณครับ",
        "ขอบคุณที่รับชม",
        "โปรดติดตามตอนต่อไป",
    ),
    "vi": (
        "cảm ơn các bạn đã xem video này",
        "hãy đăng ký kênh để ủng hộ mình nhé",
        "hẹn gặp lại các bạn trong những video tiếp theo",
        "các bạn nhớ đăng kí kênh để ủng hộ kênh",
    ),
    "id": (
        "terima kasih telah menonton",
        "terima kasih sudah menonton jangan lupa subscribe like",
    ),
    "fr": ("merci d'avoir regardé cette vidéo",),
    "es": (
        "gracias por ver el video",
        "críbete al canal gracias",
    ),
}

# Fold a phrase to a compact match key: case-folded, whitespace-free, and stripped
# of everything that is not a letter or digit. \w is already Unicode-aware for
# str patterns; the explicit CJK, kana and Thai ranges keep the intent legible.
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f]")


def fold_phrase(text) -> str:
    """The exact-match key for a hallucination phrase or a whole transcript.

    Case-folds, removes all whitespace, and drops every non-word character, so
    punctuation and spacing cannot hide a filler phrase. CJK, kana and Thai
    letters survive. O(len(text)) time and space.
    """
    return _NON_WORD.sub("", str(text).casefold())


# Folded once at import: the union of the English list and every language list.
# Membership is a whole-transcript exact match, never a substring, so an ordinary
# sentence that merely contains a phrase is not caught.
_HALLUCINATION_KEYS = frozenset(fold_phrase(phrase) for phrase in HALLUCINATIONS) | frozenset(
    fold_phrase(phrase)
    for phrases in HALLUCINATIONS_BY_LANGUAGE.values()
    for phrase in phrases
)

NO_SPEECH_PROB = 0.5
NO_SPEECH_LOGPROB = -0.5
BAD_LOGPROB = -1.0
BAD_COMPRESSION = 2.2


def enable_cuda_dlls() -> list:
    """Make the pip-installed CUDA runtime visible to CTranslate2.

    CTranslate2 loads cublas64_12.dll and cudnn64_9.dll through the normal DLL
    search path, which does not include site-packages. Returns the handles, which
    must stay referenced for the lifetime of the process or Windows unloads the
    directories again.
    """
    import os
    from pathlib import Path

    global _DLL_HANDLES
    if _DLL_HANDLES:
        return _DLL_HANDLES
    try:
        import nvidia
    except ImportError:
        return _DLL_HANDLES

    roots = [Path(entry) for entry in list(getattr(nvidia, "__path__", []) or [])]
    additions: list[str] = []
    for base in roots:
        for package in ("cublas", "cudnn", "cuda_nvrtc"):
            for sub in ("bin", "lib"):
                candidate = base / package / sub
                if candidate.is_dir():
                    additions.append(str(candidate))
    if not additions:
        return _DLL_HANDLES

    os.environ["PATH"] = os.pathsep.join(additions + [os.environ.get("PATH", "")])
    for directory in additions:
        try:
            _DLL_HANDLES.append(os.add_dll_directory(directory))
        except Exception as exc:
            log.warning("could not add %s to the DLL path: %s", directory, exc)
    return _DLL_HANDLES


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _probe_encode(model) -> None:
    """Force one real encode so an engine that cannot compute fails here, not later.

    CTranslate2 loads cublas64_12.dll and cudnn64_9.dll lazily, on the first
    encode rather than on construction, so a WhisperModel that builds is not
    proof that inference runs: without those DLLs the failure (or an endless
    hang, once the runtime had already failed inside the process) used to
    surface on the first dictation. Consuming the segment generator runs the
    encode; anything it raises sends load() to the next engine in its fallback
    list. O(1): one fixed half-second clip, one pass.
    """
    probe = np.full(SAMPLE_RATE // 2, 2e-3, dtype=np.float32)
    segments, _info = model.transcribe(probe, beam_size=1, without_timestamps=True)
    for _segment in segments:
        pass


def normalise_language_code(value) -> str | None:
    """A lowercase base language code, or None meaning "let the model decide".

    Empty, None, and the sentinels "auto"/"none" all return None. A region or
    script suffix is stripped ("en-US" -> "en", "zh_Hans" -> "zh"). O(len(value)).
    """
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text or text in (LANGUAGE_AUTO, "none"):
        return None
    base = re.split(r"[-_]", text, maxsplit=1)[0].strip()
    return base or None


def resolve_engine(config) -> tuple[str, str, str]:
    """Which model and device to try first, honouring explicit settings.

    "auto" prefers the GPU and the most accurate locally-runnable model. Measured
    on this machine: turbo at beam 5 runs at RTF 0.07 on CUDA against 2.28 on
    CPU, so the GPU is what makes a large model practical at all. On CPU a
    language of exactly "en" keeps the smaller English-only model, while "auto"
    or any other language needs the multilingual small model.
    """
    device = str(config.get("device") or "auto").lower()
    if device not in ("cpu", "cuda"):
        device = "auto"
    if device == "auto":
        device = "cuda" if _cuda_device_count() else "cpu"

    model = str(config.get("model_name") or "auto")
    if model in ("auto", "", "None"):
        if device == "cuda":
            model = V3_MODEL
        else:
            language = normalise_language_code(config.get("language"))
            model = SMALL_MODEL if language == "en" else SMALL_MULTI_MODEL

    compute = str(config.get("compute_type") or "auto").lower()
    if compute not in ("int8", "float16", "int8_float16", "float32"):
        compute = "float16" if device == "cuda" else "int8"
    return model, device, compute


def vocabulary_for(config) -> str:
    """The user's own terms, or an empty string when none are set.

    Measured with Windows SAPI speech: with no biasing a project name survived 5
    times out of 6 (heard as an ordinary word), and passing the same list as
    ``hotwords`` took that to 10 out of 10, all exact. With no list configured
    there is nothing to bias, which is the honest default.
    """
    return (config.get("custom_vocabulary") or "").strip() or DEFAULT_VOCABULARY


def hotwords_for(config, language: str = "en") -> str:
    """Shape the vocabulary into a prompt for the chosen language.

    Wrapped so the tail-biased ordering in ``vocabulary`` is used when available
    without importing it at module load (``vocabulary`` imports this module).
    English keeps the measured prose frame; every other language gets a bare term
    list, because an English prompt steers decoding back into English.
    O(n log n) inside ``vocabulary``.
    """
    try:
        from vocabulary import hotwords_for as _build

        return _build(config, language=language)
    except Exception as exc:
        log.warning("falling back to the plain vocabulary list: %s", exc)
        return vocabulary_for(config)


@dataclass
class Result:
    """One transcription, with the confidence evidence behind it."""

    text: str = ""
    reject_reason: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    audio_seconds: float = 0.0
    segments: int = 0
    language: str = ""
    language_probability: float | None = None
    language_detected: bool = False
    notes: list = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.reject_reason is None and bool(self.text)


class Transcriber:
    def __init__(self, bus, config) -> None:
        self.bus = bus
        self.config = config
        self._model = None
        self._model_key: tuple | None = None
        self._load_lock = threading.Lock()
        self._beam = int(config.get("beam_size") or DEFAULT_BEAM)

    def _key(self) -> tuple:
        return resolve_engine(self.config)

    @property
    def engine_name(self) -> str:
        """The loaded model's repository name, or "" before a load."""
        return self._model_key[0] if self._model_key else ""

    def detect_language(self, audio) -> tuple[str, float]:
        """The model's own single-segment language guess, and never an exception.

        Returns ("", 0.0) when the model is not loaded, exposes no detector, or
        the detection itself fails, so the caller can fall back to the language
        whisper reports on ``info``. One encoder pass; O(1) beyond that.
        """
        detector = getattr(self._model, "detect_language", None)
        if detector is None:
            log.warning("the loaded model has no detect_language; language stays unknown")
            return ("", 0.0)
        try:
            code, probability, _all_probs = detector(
                audio=audio,
                language_detection_segments=1,
                language_detection_threshold=0.5,
            )
            return (str(code or ""), float(probability))
        except Exception as exc:
            log.warning("language detection failed: %s", exc)
            return ("", 0.0)

    def _resolve_language(self, audio: np.ndarray) -> tuple[str | None, float | None, bool]:
        """Decide what ``language`` to pass to whisper, and what to record.

        A configured code wins with no detection call, unless the loaded model
        does not support it, in which case it degrades to auto rather than
        raising. Auto detection is O(1) and fail-open: on failure whisper is left
        to choose and the ``info`` object fills the result in. O(len(code)).
        """
        configured = normalise_language_code(self.config.get("language"))
        if configured is not None:
            supported = getattr(self._model, "supported_languages", None)
            if supported and configured not in supported:
                log.warning(
                    "configured language %r is not supported by %s; using auto detection",
                    configured, self.engine_name or "the model",
                )
            else:
                log.info("using the configured language %r", configured)
                return (configured, None, False)

        code, probability = self.detect_language(audio)
        if not code:
            log.warning("language detection yielded nothing; letting whisper decide")
            return (None, None, False)
        if probability < 0.5:
            log.warning(
                "auto-detected language %r with low probability %.3f; keeping it",
                code, probability,
            )
        else:
            log.info("auto-detected language %r (p=%.3f)", code, probability)
        return (code, probability, True)

    def load(self) -> None:
        """Load the preferred engine, or a safe fallback if it cannot run.

        A cached model must never hit the network: this machine has a metered
        connection and the app is expected to work with no internet. Only a
        model that is genuinely absent is downloaded, once.

        Every candidate must survive one real encode before it is kept: on CUDA
        the runtime DLLs load lazily on the first encode, so construction alone
        proves nothing. The lock keeps the startup warm-up and a decode that
        arrives during it from constructing the model twice.
        """
        from faster_whisper import WhisperModel

        key = self._key()
        with self._load_lock:
            if self._model is not None and self._model_key == key:
                return

            beam = int(self.config.get("beam_size") or DEFAULT_BEAM)
            # The safe CPU fallback must be able to produce every configured
            # language: small.en cannot emit anything but English, so a
            # non-English setting falls back to the multilingual small model
            # first, then to small.en.
            language = normalise_language_code(self.config.get("language"))
            cpu_model = SMALL_MODEL if language == "en" else SMALL_MULTI_MODEL
            attempts: list[tuple] = []
            for attempt in (
                (*key, beam),
                (cpu_model, "cpu", "int8", _FALLBACK_BEAM),
                (SMALL_MODEL, "cpu", "int8", _FALLBACK_BEAM),
            ):
                if attempt not in attempts:
                    attempts.append(attempt)
            last_error: Exception | None = None
            for model_name, device, compute_type, attempt_beam in attempts:
                if device == "cuda":
                    enable_cuda_dlls()
                try:
                    candidate = WhisperModel(
                        model_name, device=device, compute_type=compute_type,
                        local_files_only=True,
                    )
                    _probe_encode(candidate)
                except Exception as exc:
                    last_error = exc
                    log.warning(
                        "engine %s on %s/%s failed (%s); trying the fallback",
                        model_name, device, compute_type, exc,
                    )
                    continue
                self._model = candidate
                self._model_key = (model_name, device, compute_type)
                self._beam = attempt_beam
                log.info(
                    "engine: %s on %s/%s, beam %s",
                    model_name, device, compute_type, attempt_beam,
                )
                return

            raise RuntimeError(
                f"no usable speech model: {last_error}. If a model is genuinely absent, "
                "download it deliberately with: python tools/fetch_model.py --model small"
            )

    def warm_up(self) -> None:
        """Load the model and decode one quiet second so the first dictation is not slow.

        The buffer must not be digital silence: transcribe_audio rejects that
        before it ever touches the model, so the previous zeros warmed nothing.
        A quiet constant passes the gates and runs the real decode path, which
        is also what turns a broken engine into a startup notification instead
        of a surprise during the first dictation. Failures are reported on the
        bus rather than raised: this runs on a background thread at startup and
        must not take the app down.
        """
        try:
            self.load()
            self.transcribe_audio(np.full(SAMPLE_RATE, 2e-3, dtype=np.float32))
        except Exception as exc:
            log.exception("warm-up failed")
            self.bus.transcription_error.emit(f"Speech model failed to load: {exc}")

    def transcribe(self, audio: np.ndarray) -> str:
        """Convenience wrapper returning just the text. Blocking."""
        return self.transcribe_audio(audio).text

    def transcribe_audio(self, audio: np.ndarray) -> Result:
        """Transcribe one clip, and never keep a model that just failed.

        If the engine raises mid-inference (a lost CUDA context, a driver
        reset), the instance is dropped so the next call goes back through
        load() and its fallback list instead of reusing a broken model. The
        failure itself is re-raised for the caller to report.
        """
        try:
            return self._transcribe(audio)
        except Exception:
            self._model = None
            self._model_key = None
            raise

    def _retry_truncated(self, audio: np.ndarray, cuts: list[float], segments: list, decode: dict) -> list:
        """Re-decode pieces that ``truncated_pieces`` flags, each on its own.

        A segment belongs to the piece its start falls in. The retry drops the
        hotwords prompt, which is what made whisper end the piece early, and it
        replaces the piece only when it adds ``DROP_MIN_GAIN`` more words (at
        least three). O(segments log pieces) plus one decode per flagged piece.
        """
        starts = cuts[0::2]
        groups: list[list] = [[] for _ in starts]
        for segment in segments:
            index = bisect.bisect_right(starts, float(getattr(segment, "start", 0.0)) + 1e-3) - 1
            groups[max(0, index)].append(segment)
        counts = [sum(len(segment.text.split()) for segment in group) for group in groups]
        for index in truncated_pieces(audio, cuts, counts):
            span = [cuts[2 * index], cuts[2 * index + 1]]
            retry, _ = self._model.transcribe(audio, clip_timestamps=span, condition_on_previous_text=False, **{**decode, "hotwords": None})
            retry = list(retry)
            gained = sum(len(segment.text.split()) for segment in retry)
            if gained >= counts[index] + max(3, DROP_MIN_GAIN * counts[index]):
                log.info("re-decoded truncated piece %.1f-%.1f s: %d -> %d words", *span, counts[index], gained)
                groups[index] = retry
        return [segment for group in groups for segment in group]

    def _transcribe(self, audio: np.ndarray) -> Result:
        """Transcribe and decide whether the result is trustworthy.

        Blocking; call from a worker thread. A result is REJECTED rather than
        returned when the model's own confidence says the audio was not speech,
        because inserting a hallucination is far worse than asking the user to
        repeat themselves. The thresholds and their sources are documented on the
        module constants.
        """
        if audio is None or audio.size == 0:
            return Result(reject_reason="nothing to transcribe")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        seconds = float(audio.size) / SAMPLE_RATE
        if float(np.max(np.abs(audio))) < EMPTY_AUDIO_FLOOR:
            return Result(reject_reason="the recording is silent", audio_seconds=seconds)
        if seconds < 0.30:
            return Result(reject_reason="too short to contain speech", audio_seconds=seconds)

        self.load()
        chosen_language, detected_probability, language_detected = self._resolve_language(audio)
        initial_prompt = (self.config.get("initial_prompt") or "").strip()
        vad_filter = bool(self.config.get("vad_filter", False))
        cuts = None if vad_filter else silence_cuts(audio)
        decode = dict(
            language=chosen_language,
            multilingual=bool(self.config.get("detect_per_segment")),
            beam_size=self._beam,
            prompt_reset_on_temperature=0.5,
            vad_filter=vad_filter,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                speech_pad_ms=200,
                max_speech_duration_s=MAX_WINDOW_SECONDS - 2.0,
            ),
            initial_prompt=initial_prompt or None,
            hotwords=hotwords_for(self.config, chosen_language or "") or None,
            no_speech_threshold=NO_SPEECH_PROB,
            log_prob_threshold=BAD_LOGPROB,
            compression_ratio_threshold=BAD_COMPRESSION,
            word_timestamps=False,
            without_timestamps=True,
        )
        segments, info = self._model.transcribe(
            audio,
            clip_timestamps=cuts or "0",
            condition_on_previous_text=bool(self.config.get("condition_on_previous", True)),
            **decode,
        )
        if cuts:
            segments = self._retry_truncated(audio, cuts, list(segments), decode)

        pieces = []
        logprobs = []
        no_speech = []
        compressions = []
        temperatures = []
        for segment in segments:
            pieces.append(segment.text)
            if segment.avg_logprob is not None:
                logprobs.append(float(segment.avg_logprob))
            if segment.no_speech_prob is not None:
                no_speech.append(float(segment.no_speech_prob))
            if segment.compression_ratio is not None:
                compressions.append(float(segment.compression_ratio))
            if getattr(segment, "temperature", None) is not None:
                temperatures.append(float(segment.temperature))

        text = " ".join(piece.strip() for piece in pieces if piece.strip()).strip()

        # With no detected code, whisper picked the language itself and its info
        # object carries the guess; otherwise the code chosen above is the truth.
        if chosen_language:
            result_language = chosen_language
            result_probability = detected_probability
            result_detected = language_detected
        else:
            result_language = str(getattr(info, "language", "") or "")
            info_probability = getattr(info, "language_probability", None)
            result_probability = (
                float(info_probability) if info_probability is not None else None
            )
            result_detected = False

        result = Result(
            text=text,
            avg_logprob=min(logprobs) if logprobs else None,
            no_speech_prob=max(no_speech) if no_speech else None,
            compression_ratio=max(compressions) if compressions else None,
            temperature=max(temperatures) if temperatures else None,
            audio_seconds=seconds,
            segments=len(pieces),
            language=result_language,
            language_probability=result_probability,
            language_detected=result_detected,
        )

        if not text:
            result.reject_reason = "no words were recognised"
            return result

        if fold_phrase(text) in _HALLUCINATION_KEYS:
            result.reject_reason = f"recognised the filler phrase {text.strip()!r}"
            return result

        if result.compression_ratio is not None and result.compression_ratio >= BAD_COMPRESSION:
            result.reject_reason = "the output was repetitive, so it is not a real transcription"
            return result

        if result.avg_logprob is not None and result.avg_logprob < BAD_LOGPROB:
            result.reject_reason = "the model was not confident enough in any of it"
            return result

        if (
            result.no_speech_prob is not None
            and result.avg_logprob is not None
            and result.no_speech_prob >= NO_SPEECH_PROB
            and result.avg_logprob < NO_SPEECH_LOGPROB
        ):
            result.reject_reason = "the recording does not contain speech"
            return result

        if (
            result.temperature is not None
            and result.temperature > 0.0
            and result.avg_logprob is not None
            and result.avg_logprob < BAD_LOGPROB / 2
        ):
            result.reject_reason = "the model had to guess, so the words are unreliable"
            return result

        return result


def estimate_confidence(result: Result) -> float:
    """A 0..1 score from the evidence, used only for logging and display.

    Combines the model's mean log probability with the no-speech probability.
    Linear in the number of segments; O(1) here because they are already reduced.
    """
    if result.avg_logprob is None:
        return 0.0
    probability = math.exp(min(result.avg_logprob, 0.0))
    if result.no_speech_prob is not None:
        probability *= 1.0 - min(max(result.no_speech_prob, 0.0), 1.0)
    return max(0.0, min(1.0, probability))
