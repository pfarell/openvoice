"""Personal vocabulary for local dictation: prompt shaping, candidate mining,
and learning from the user's own corrections.

Three levers matter, in this order:

1. ``build_hotwords`` shapes the text faster-whisper feeds whisper as the
   ``<|startofprev|>`` prompt. That prompt is capped at roughly 223 tokens and
   anything past the cap is silently dropped, so only the tail carries weight.
   The list is therefore ordered worst-to-best and the terms that matter most
   sit last.
2. ``suggest_terms`` mines a folder tree for proper nouns worth adding by hand.
3. ``CorrectionLearner`` diffs the text that was inserted against the text the
   user actually kept and proposes dictionary and replacement entries, the way
   shipped dictation products learn new words.

There is no logit-level term boost in the installed CTranslate2, so prompt
shaping and post-hoc correction are the only levers available.

``DEFAULT_TERMS`` is derived from the shipped ``transcribe.DEFAULT_VOCABULARY``
so the two built-in lists cannot drift apart.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from collections import Counter, deque
from pathlib import Path

from transcribe import DEFAULT_VOCABULARY

DEFAULT_TERMS: list[str] = [term.strip() for term in DEFAULT_VOCABULARY.split(",") if term.strip()]

DEFAULT_LEARN_THRESHOLD = 2
NEAR_MISS_RATIO = 0.5
MIN_TERM_LENGTH = 4
MAX_DEPTH = 3
MAX_DIRS = 300

# The framing was chosen by measurement on the user's own recordings. A prompt
# that ENDS in a bare comma-separated list of capitalised names scored 78.9%
# while the same terms wrapped in ordinary prose scored 80.1%, and the prose form
# was the only one that improved the hard technical sentence (57% -> 64%). The
# list-ending form also biases the model toward producing invented brand-like
# names mid-sentence, which is what turned "CadQuery" into "ConnectMovie".
_HOTWORD_PREFIX = "The speaker may mention "
_HOTWORD_SUFFIX = " while dictating ordinary sentences about work, budgets and projects."

_RARE_LETTERS = frozenset("qxzjvkw")

_BUILTIN_KEYS = frozenset(term.casefold() for term in DEFAULT_TERMS)
_CANONICAL_CASING = {term.casefold(): term for term in DEFAULT_TERMS}

_NOISE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
        "site-packages",
        "dist",
        "build",
        "target",
        "coverage",
        ".next",
        ".cache",
    }
)

_STOPWORDS = frozenset(
    """
    a about all also an and any app apps are as at be been build but by cache can
    changelog code command common config copy core data day did do doc docs done
    down each edit end env file files final find first for from get git github
    had has have he her here him his how if in index init into is it its just key
    keys last lib library license like list local log logs made main make makefile
    manifest meta model models more most much new next not now office old on once
    one only open or other our out over page pages part path paths plus put
    pyproject readme requirements run same save script scripts see set setup she
    should simple since small so some src start state step still such sure system
    take temp test tests text than that the their them then there these they thing
    this those tmp to too tool tools try two type types up url use used user users
    using utils version view views was way web were what when where which while who
    why will with without work works would write you your
    """.split()
)

_NAME_SPLIT = re.compile(r"[_\-\s.]+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019-]*")


def terms_from_config(config) -> list[str]:
    """The user's terms, or the built-in list when the setting is empty.

    The setting accepts commas and newlines. Terms are de-duplicated
    case-insensitively, the user's capitalisation and order are preserved, and
    a setting that is only separators falls back to ``DEFAULT_TERMS``.

    Complexity: O(n) time and O(n) space over n characters of the setting.
    """
    raw = str(config.get("custom_vocabulary") or "").strip()
    if not raw:
        return list(DEFAULT_TERMS)
    seen: set[str] = set()
    terms: list[str] = []
    for part in re.split(r"[,\n\r]+", raw):
        term = part.strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms or list(DEFAULT_TERMS)


def _has_internal_caps(term: str) -> bool:
    return any(char.isupper() for char in term[1:])


def _is_builtin(term: str) -> bool:
    return term.casefold() in _BUILTIN_KEYS


def _importance(term: str) -> int:
    """A higher score means a more important term, which must sit later in the
    prompt because the tail is what the recogniser actually weighs.

    The ordering was measured wrong the first time. The original heuristic ranked
    LONG, distinctive terms highest, which put the short word-like ones at the
    front where they have the least influence — and those are exactly the ones the
    language model substitutes with ordinary words. On a real dictation,
    a five-letter project name (position 4 of 11) came back as an ordinary word
    and a four-letter one (position 5) as another, while the long distinctive
    names at the tail survived.

    So the weighting is inverted: SHORT, ordinary-looking terms rank highest and
    take the tail. Distinctiveness (internal capitals, digits) only mildly lowers
    a term, because such a term is already unlikely to be mutated into a common
    word. The user's own terms still outrank the built-in ones.

    Complexity: O(len(term)).
    """
    score = 0 if _is_builtin(term) else 1000
    # Two groups. SHORT word-like terms go last and get the most weight, because
    # those are the ones the language model swaps for an ordinary word. Long or
    # distinctive terms form the group just before them: they still sit in the
    # part of the prompt that carries weight, but they do not displace the terms
    # that actually need rescuing. Within each group, longer ranks first so the
    # most self-distinctive name is furthest from the tail.
    short_and_confusable = len(term) <= 6
    score += (100 if short_and_confusable else 0) + max(0, 16 - min(len(term), 16))
    if _has_internal_caps(term):
        score -= 2
    return score


def _ranked_ascending(terms) -> list[str]:
    """Terms de-duplicated case-insensitively and sorted least-to-most
    important, so the final element is the one that must reach the prompt's
    tail.

    Complexity: O(n log n) time and O(n) space over the n unique terms.
    """
    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for index, raw in enumerate(terms):
        term = str(raw).strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append((index, term))
    unique.sort(key=lambda pair: (_importance(pair[1]), pair[0]))
    return [term for _, term in unique]


def _language_base(code) -> str:
    """A lowercase base language code ("en-US" -> "en"), or "" when absent.

    Complexity: O(len(code)).
    """
    if not isinstance(code, str):
        return ""
    return code.strip().lower().replace("_", "-").split("-", 1)[0]


def _uses_english_prose(language) -> bool:
    """Whether the English prose framing is safe for ``language``.

    An English prompt biases a non-English transcription back toward English, so
    the prose frame is used only for English or an absent/unknown value. Every
    concrete other code gets the bare term list instead.

    Complexity: O(len(code)).
    """
    return _language_base(language) in ("", "en", "auto")


def build_hotwords(terms, max_tokens: int = 150, language: str = "en") -> str:
    """Build the single natural sentence passed to whisper as ``hotwords``.

    The terms are ordered least-to-most important and the sentence is trimmed
    from the front (lowest priority first) until it fits ``max_tokens``. The
    token count is approximated as ``ceil(len(text) / 4)``, which is exact for
    the character budget because ``ceil(length / 4) <= max_tokens`` holds
    exactly when ``length <= 4 * max_tokens``.

    For English (the default, and any absent/unknown code) the shipped prose
    framing and " and " join are kept. Every other language gets a bare
    comma-separated list with no English connective: an English sentence biases
    a non-English transcription toward English, and " and " is an English word.
    The tail ordering and front-trimming are identical for both.

    Empty input, a non-positive budget, or a single term too large to fit all
    return ``""`` rather than a sentence that breaks the cap.

    Complexity: O(n log n) time and O(n) space over the n terms, plus O(total
    characters) to join the kept terms.
    """
    if not terms or max_tokens <= 0:
        return ""
    english = _uses_english_prose(language)
    if english:
        # The prose framing costs about 26 tokens. When the caller's budget is
        # too small for that plus any terms, fall back to a terse list so the
        # budget is still honoured and something useful is still passed.
        prose = _HOTWORD_PREFIX + _HOTWORD_SUFFIX
        if len(prose) + 12 > 4 * max_tokens:
            prefix, suffix = "Terms: ", ""
        else:
            prefix, suffix = _HOTWORD_PREFIX, _HOTWORD_SUFFIX
        separator = " and "
    else:
        prefix, suffix = "", ""
        separator = ", "
    budget = 4 * max_tokens - len(prefix) - len(suffix)
    if budget <= 0:
        return ""

    kept: list[str] = []
    used = 0
    # The separator cost must be the worst case the final join can use (for
    # English " and ", not ", "), or the count overruns the token budget.
    for term in reversed(_ranked_ascending(terms)):
        extra = len(term) + (len(separator) if kept else 0)
        if used + extra > budget:
            break
        used += extra
        kept.append(term)
    if not kept:
        return ""
    kept.reverse()
    if len(kept) == 1:
        body = kept[0]
    elif english:
        body = ", ".join(kept[:-1]) + " and " + kept[-1]
    else:
        body = ", ".join(kept)
    return prefix + body + suffix


def hotwords_for(config, max_tokens: int = 150, language: str = "en") -> str:
    """Convenience entry point: the prompt sentence for the current config.

    ``language`` is forwarded to :func:`build_hotwords`.

    Complexity: O(n log n) time and O(n) space over the configured terms.
    """
    return build_hotwords(terms_from_config(config), max_tokens, language)


def _is_noise_dir(name: str) -> bool:
    return name.startswith(".") or name.casefold() in _NOISE_DIRS


def _canonical_part(part: str) -> str | None:
    """One name component as a term, or None when it is not worth suggesting.

    Length, digits, and the stoplist are filtered here; the built-in list acts
    as a canonical-casing table when the shipped list defines one, so a
    lowercased folder can come back with its documented capitalisation.

    Complexity: O(len(part)).
    """
    if not part:
        return None
    if part.isdigit():
        return None
    key = part.casefold()
    if key in _STOPWORDS:
        return None
    if len(part) < MIN_TERM_LENGTH:
        return None
    if part.isupper() and part.isalpha():
        return part
    canonical = _CANONICAL_CASING.get(key)
    if canonical is not None:
        return canonical
    if _has_internal_caps(part):
        return part
    return part[:1].upper() + part[1:]


def _candidate_from_name(name: str) -> str | None:
    """A term mined from a directory name or a source file stem.

    snake_case and kebab-case are split, CamelCase is preserved as one term,
    stopword components are dropped, and the survivors are joined into one
    phrase (``Wallet_Tracker`` -> ``Wallet Tracker``).

    Complexity: O(len(name)).
    """
    parts = [part for part in _NAME_SPLIT.split(name) if part]
    kept = [term for term in (_canonical_part(part) for part in parts) if term]
    if not kept:
        return None
    phrase = " ".join(kept)
    return phrase if len(phrase) >= MIN_TERM_LENGTH else None


def _mine_score(term: str, count: int) -> int:
    """Ranking score for a mined term: repetition, length, and distinctiveness.

    Complexity: O(len(term)).
    """
    score = count * 10
    score += min(len(term), 24)
    if _has_internal_caps(term):
        score += 8
    if any(char.isdigit() for char in term):
        score += 6
    if any(char.lower() in _RARE_LETTERS for char in term):
        score += 3
    return score


def suggest_terms(directory, limit: int = 200) -> list[str]:
    """Mine a folder tree for distinctive candidate names to offer the user.

    Directory names and source file stems are tokenised; noise directories
    (``.git``, ``node_modules``, ``__pycache__``, ``.venv``, ``dist``,
    ``build``, and any dot-directory), hidden files, components shorter than
    four characters, pure numbers, and common English words are skipped. The
    walk is breadth-first, capped at ``MAX_DEPTH`` levels and ``MAX_DIRS``
    directories so it stays fast on a large checkout. Results are de-duplicated
    case-insensitively and sorted by score, then by name for determinism.

    Complexity: O(entries visited) to scan, plus O(u log u) time and O(u) space
    to sort the u unique candidates.
    """
    root = Path(directory)
    if not root.is_dir():
        return []

    counts: Counter = Counter()
    casing: dict[str, str] = {}
    visited = 0
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    while queue and visited < MAX_DIRS:
        current, depth = queue.popleft()
        visited += 1
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            if is_dir:
                if depth < MAX_DEPTH and not _is_noise_dir(name):
                    queue.append((Path(entry.path), depth + 1))
                candidate = _candidate_from_name(name)
            else:
                candidate = _candidate_from_name(Path(name).stem)
            if candidate is None:
                continue
            key = candidate.casefold()
            counts[key] += 1
            casing.setdefault(key, candidate)

    ranked = sorted(
        ((casing[key], count) for key, count in counts.items()),
        key=lambda item: (-_mine_score(item[0], item[1]), item[0].casefold(), item[0]),
    )
    return [term for term, _ in ranked[:limit]]


def _shared_prefix(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _is_near_miss(wrong: str, good: str) -> bool:
    """Whether ``good`` looks like a mis-hearing of ``wrong``.

    The comparison is case-sensitive on purpose: it is what makes the gate
    discriminate. ``"ready"`` -> ``"Vantry"`` shares no leading character and
    scores 0.36 under a case-sensitive ratio, so it is rejected, while
    ``"Vynt"`` -> ``"Vint"`` scores 0.75 and ``"NovaGatx"`` -> ``"NovaGate"``
    shares four leading characters, so both are accepted.

    Complexity: O(len(wrong) + len(good)).
    """
    if _shared_prefix(wrong, good) >= 3:
        return True
    return difflib.SequenceMatcher(None, wrong, good, autojunk=False).ratio() >= NEAR_MISS_RATIO


def _looks_like_term(token: str) -> bool:
    """A plausible dictionary entry: one word, alphabetic or alphanumeric, not
    a common English word, not a bare number.

    Complexity: O(len(token)).
    """
    if not token or token.isdigit():
        return False
    if token.casefold() in _STOPWORDS:
        return False
    return _WORD_RE.fullmatch(token) is not None


def _pair_replacements(wrongs: list[str], rights: list[str]) -> list[tuple[str, str]]:
    """Pair the words on each side of a ``replace`` opcode.

    Equal-length sides pair positionally. Unequal sides greedily pair each new
    word with the most similar old one. Both sides are small (the words changed
    in one dictation), so the quadratic worst case is bounded and irrelevant.

    Complexity: O(w * r) time and O(w) space over w wrong and r right words.
    """
    if len(wrongs) == len(rights):
        return list(zip(wrongs, rights))
    pairs: list[tuple[str, str]] = []
    unused = list(wrongs)
    for good in rights:
        if not unused:
            break
        best = max(
            unused,
            key=lambda word: difflib.SequenceMatcher(None, word, good, autojunk=False).ratio(),
        )
        unused.remove(best)
        pairs.append((best, good))
    return pairs


def _match_case(sample: str, good: str) -> str:
    """Apply ``good`` using the casing the transcript used for ``sample``.

    Complexity: O(len(sample) + len(good)).
    """
    if sample.isupper() and any(char.isalpha() for char in sample):
        return good.upper()
    if sample.islower():
        return good.lower()
    if sample[:1].isupper() and sample[1:].islower():
        return good[:1].upper() + good[1:]
    return good


class CorrectionLearner:
    """Learn dictionary terms and replacements from the user's own edits.

    Storage is ``learned_terms.json`` next to the config file, with keys
    ``terms`` (term -> times seen) and ``replacements`` (wrong -> right). A
    missing or corrupt file starts empty and reports the reason on stderr;
    nothing here raises.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._terms: dict[str, int] = {}
        self._replacements: dict[str, str] = {}
        self._load()

    def _path(self) -> Path:
        return Path(self.config.path).parent / "learned_terms.json"

    def _threshold(self) -> int:
        try:
            value = int(self.config.get("learn_threshold"))
        except (TypeError, ValueError):
            value = DEFAULT_LEARN_THRESHOLD
        return max(1, value)

    def _load(self) -> None:
        """Read the store, tolerating any corruption.

        Complexity: O(entries) time and space.
        """
        self._terms = {}
        self._replacements = {}
        path = self._path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[vocabulary] unreadable {path.name} ({exc}); starting empty", file=sys.stderr)
            return
        if not isinstance(raw, dict):
            print(
                f"[vocabulary] {path.name} holds {type(raw).__name__}, not an object; starting empty",
                file=sys.stderr,
            )
            return
        terms = raw.get("terms")
        if isinstance(terms, dict):
            for key, value in terms.items():
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    self._terms[str(key)] = count
        replacements = raw.get("replacements")
        if isinstance(replacements, dict):
            for wrong, good in replacements.items():
                if isinstance(wrong, str) and isinstance(good, str) and wrong and good:
                    self._replacements[wrong] = good

    def _save(self) -> None:
        """Atomically replace the store; a failed write is reported, not raised.

        Complexity: O(entries) time and space.
        """
        try:
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"terms": dict(self._terms), "replacements": dict(self._replacements)}
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[vocabulary] could not save learned terms ({exc})", file=sys.stderr)

    def _count(self, term: str) -> int:
        return self._terms.get(term, 0)

    def observe(self, inserted: str, final: str) -> list[str]:
        """Compare what was inserted with what the user kept and learn from it.

        Only ``replace`` opcodes are considered. A replacement is recorded when
        it looks like a term (one word, alphabetic or alphanumeric, not a
        common word) and the replaced text is a near-miss of it. Terms become
        active once seen ``learn_threshold`` times (default 2), so a one-off
        typo never reaches the dictionary. Returns the terms that crossed the
        threshold on this call.

        Complexity: O(w) time and space over the w words in the two
        transcripts, dominated by the sequence diff.
        """
        inserted = str(inserted or "")
        final = str(final or "")
        if not inserted or not final or inserted == final:
            return []

        left = _WORD_RE.findall(inserted)
        right = _WORD_RE.findall(final)
        matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
        changed = False
        graduated: list[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            for wrong, good in _pair_replacements(left[i1:i2], right[j1:j2]):
                if wrong.casefold() == good.casefold():
                    continue
                if not _looks_like_term(good):
                    continue
                if not _is_near_miss(wrong, good):
                    continue
                self._terms[good] = self._count(good) + 1
                if self._replacements.get(wrong) != good:
                    self._replacements[wrong] = good
                changed = True
                if self._count(good) == self._threshold():
                    graduated.append(good)
        if changed:
            self._save()
        return graduated

    def learned_terms(self) -> list[str]:
        """Every term seen at least ``learn_threshold`` times, most-seen first.

        Complexity: O(n log n) time and O(n) space over the stored terms.
        """
        threshold = self._threshold()
        active = [(term, count) for term, count in self._terms.items() if count >= threshold]
        active.sort(key=lambda item: (-item[1], item[0].casefold(), item[0]))
        return [term for term, _ in active]

    def replacements(self) -> dict:
        """The stored wrong -> right mappings, whether or not yet active.

        Complexity: O(n) time and space.
        """
        return dict(self._replacements)

    def apply(self, text: str) -> str:
        """Apply active replacements to ``text`` at word boundaries.

        Only mappings whose right-hand term is active are applied, so an
        unconfirmed one-off edit cannot rewrite text. Longer keys are applied
        first so a short key cannot clobber part of a longer one, and casing
        follows the matched text.

        Complexity: O(k * len(text)) time and O(len(text)) space over the k
        active replacements.
        """
        if not text or not self._replacements:
            return text
        threshold = self._threshold()
        active = {
            wrong: good
            for wrong, good in self._replacements.items()
            if self._count(good) >= threshold
        }
        if not active:
            return text
        result = str(text)
        for wrong in sorted(active, key=len, reverse=True):
            good = active[wrong]
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(wrong) + r"(?![A-Za-z0-9])")
            result = pattern.sub(lambda match, target=good: _match_case(match.group(0), target), result)
        return result

    def forget(self, term) -> None:
        """Drop a term and any replacement pointing at or out of it.

        Complexity: O(entries) time and space.
        """
        key = str(term).casefold()
        removed = False
        for existing in list(self._terms):
            if existing.casefold() == key:
                del self._terms[existing]
                removed = True
        for wrong, good in list(self._replacements.items()):
            if wrong.casefold() == key or good.casefold() == key:
                del self._replacements[wrong]
                removed = True
        if removed:
            self._save()
