"""Deterministic inverse text normalisation for dictated text.

faster-whisper emits spoken forms inconsistently ("twelve point five", "five
percent", "five to ten"). This converts clearly spoken forms into the written
form a typist would produce. It is rule-based on purpose: NeMo's ITN needs
Pynini (conda-only on Windows) and neural ITN hallucinates, and ITN has "an
extremely low tolerance to unrecoverable errors" (Interspeech 2021).

Two rules govern everything here:
  1. Never invent content. If a pattern is not clearly present, return the text
     unchanged.
  2. Be conservative with small counts in prose. "I have two cats" stays words;
     only counts with a unit, currency, percent or a value >= 10000 become
     digits, because that is where a typist would use digits too.

Language routing is mandatory, not an optimisation. An English prompt biases a
non-English transcription toward English, so the English word tables must not
touch another language: "million" is an English scale word and rewrites French
"deux million d'euros" as "deux 1,000,000 d'euros". Chinese, Japanese and Thai
are written without spaces, so the space-token pipeline cannot see their word
boundaries at all and is skipped in favour of character-level cleanup. See
:func:`normalise` for the routing itself.

Time: O(n) over the token count for every rule; the number parser walks each
token once because it consumes runs greedily. Space: O(n) for the token list.
"""

from __future__ import annotations

import re

SAMPLE_RATE_UNUSED = None

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {
    "hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9,
    "trillion": 10 ** 12,
}
ORDINAL_CARDINAL = {
    "first": ("one", "st"), "second": ("two", "nd"), "third": ("three", "rd"),
    "fourth": ("four", "th"), "fifth": ("five", "th"), "sixth": ("six", "th"),
    "seventh": ("seven", "th"), "eighth": ("eight", "th"), "ninth": ("nine", "th"),
    "tenth": ("ten", "th"), "eleventh": ("eleven", "th"), "twelfth": ("twelve", "th"),
    "thirteenth": ("thirteen", "th"), "fourteenth": ("fourteen", "th"),
    "fifteenth": ("fifteen", "th"), "sixteenth": ("sixteen", "th"),
    "seventeenth": ("seventeen", "th"), "eighteenth": ("eighteen", "th"),
    "nineteenth": ("nineteen", "th"), "twentieth": ("twenty", "th"),
    "thirtieth": ("thirty", "th"), "fortieth": ("forty", "th"),
    "fiftieth": ("fifty", "th"), "sixtieth": ("sixty", "th"),
    "seventieth": ("seventy", "th"), "eightieth": ("eighty", "th"),
    "ninetieth": ("ninety", "th"), "hundredth": ("hundred", "th"),
    "thousandth": ("thousand", "th"), "millionth": ("million", "th"),
}
FRACTIONS = {
    ("a", "half"): "1/2", ("one", "half"): "1/2",
    ("a", "quarter"): "1/4", ("one", "quarter"): "1/4",
    ("three", "quarters"): "3/4",
    ("a", "third"): "1/3", ("one", "third"): "1/3", ("two", "thirds"): "2/3",
}
MATH_WORDS = {"plus", "minus", "times", "multiplied", "divided", "over"}

PUNCTUATION = {
    "comma": ",", "period": ".", "full stop": ".", "question mark": "?",
    "exclamation mark": "!", "exclamation point": "!", "colon": ":",
    "semicolon": ";", "hyphen": "-", "ellipsis": "...",
    "open parenthesis": "(", "close parenthesis": ")",
    "open paren": "(", "close paren": ")",
    "open bracket": "[", "close bracket": "]",
    "open quote": '"', "close quote": '"',
    "new line": "\n", "newline": "\n", "new paragraph": "\n\n",
}
PUNCTUATION_MULTI = {k: v for k, v in PUNCTUATION.items() if " " in k}
PUNCTUATION_SINGLE = {k: v for k, v in PUNCTUATION.items() if " " not in k}

UNITS_MAP = {
    "millimetre": "mm", "millimeter": "mm", "millimetres": "mm", "millimeters": "mm",
    "centimetre": "cm", "centimeter": "cm", "centimetres": "cm", "centimeters": "cm",
    "metre": "m", "meter": "m", "metres": "m", "meters": "m",
    "kilometre": "km", "kilometer": "km", "kilometres": "km", "kilometers": "km",
    "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "gram": "g", "grams": "g", "milligram": "mg", "milligrams": "mg",
    "litre": "L", "liter": "L", "litres": "L", "liters": "L",
    "millilitre": "mL", "milliliter": "mL", "millilitres": "mL", "milliliters": "mL",
    "gigabyte": "GB", "gigabytes": "GB", "megabyte": "MB", "megabytes": "MB",
    "kilobyte": "KB", "kilobytes": "KB", "terabyte": "TB", "terabytes": "TB",
    "hertz": "Hz", "kilohertz": "kHz", "megahertz": "MHz", "gigahertz": "GHz",
    "millisecond": "ms", "milliseconds": "ms", "second": "s", "seconds": "s",
    "minute": "min", "minutes": "min", "hour": "h", "hours": "h",
    "volt": "V", "volts": "V", "watt": "W", "watts": "W",
    "amp": "A", "amps": "A", "ampere": "A", "amperes": "A",
    "celsius": "\u00b0C", "fahrenheit": "\u00b0F",
}

CURRENCY = {
    "dollar": "$", "dollars": "$", "euro": "\u20ac", "euros": "\u20ac",
    "pound": "\u00a3", "pounds": "\u00a3", "rupiah": "Rp", "yen": "\u00a5",
    "cent": "\u00a2", "cents": "\u00a2",
}
CURRENCY_SUFFIX = {"\u00a2"}

MATH_SYMBOLS = {
    "plus": "+", "minus": "-", "times": "\u00d7", "multiplied by": "\u00d7",
    "divided by": "/", "over": "/",
}
SQUARED = "squared"
CUBED = "cubed"

TOKEN = re.compile(r"\S+")
TRAILING = re.compile(r"[.,!?;:]+$")
LEADING = re.compile(r"^[.,!?;:(\[]+")

BIG_NUMBER = 10000

# --- Language routing -------------------------------------------------------
# An English prompt biases non-English transcription toward English, and the
# space-token pipeline cannot represent languages written without spaces, so
# each language runs only the passes that are safe for it.

#: Languages written without word spaces: the whole space-token pipeline is
#: skipped for these in favour of character-level cleanup.
NO_SPACE_LANGUAGES = frozenset({"zh", "ja", "th", "yue"})

#: Languages that get the digit-based decimal/percent/currency rules this phase.
DIGIT_LANGUAGES = frozenset({"fr", "es", "id", "vi"})

#: Decimal separator words, keyed by language, tried longest-first.
DECIMAL_WORDS: dict[str, tuple[str, ...]] = {
    "fr": ("virgule",),
    "es": ("coma",),
    "id": ("koma",),
    "vi": ("phẩy", "phảy"),
}

#: Spoken percent phrases, keyed by language.
PERCENT_PHRASES: dict[str, tuple[str, ...]] = {
    "fr": ("pour cent",),
    "es": ("por ciento",),
    "id": ("persen",),
    "vi": ("phần trăm",),
}

#: Currency words (written AFTER the number in these languages) -> symbol.
CURRENCY_SYMBOLS: dict[str, dict[str, str]] = {
    "fr": {"euros": "€", "euro": "€", "dollars": "$", "dollar": "$"},
    "es": {"euros": "€", "dólares": "$", "dolares": "$", "pesos": "$"},
    "id": {"rupiah": "Rp", "dolar": "$"},
    "vi": {"đồng": "₫", "đô la": "$"},
}

#: Localised spoken punctuation. Multi-word entries are iterated first so a
#: short entry cannot consume part of a longer phrase ("dấu chấm hỏi" before
#: "dấu chấm"). NOTE: bare French "point" is deliberately absent - it is an
#: ordinary noun ("le point sur la situation"), so mapping it would corrupt
#: normal prose.
PUNCTUATION_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "fr": {
        "point d'interrogation": "?",
        "point d'exclamation": "!",
        "deux points": ":",
        "point final": ".",
        "point-virgule": ";",
        "virgule": ",",
        "nouvelle ligne": "\n",
    },
    "es": {
        "signo de interrogación": "?",
        "signo de exclamación": "!",
        "punto y coma": ";",
        "dos puntos": ":",
        "punto y aparte": "\n\n",
        "punto y seguido": ". ",
        "nueva línea": "\n",
        "coma": ",",
        "punto": ".",
    },
    "id": {
        "tanda tanya": "?",
        "tanda seru": "!",
        "baris baru": "\n",
        "koma": ",",
        "titik": ".",
    },
    "vi": {
        "dấu chấm hỏi": "?",
        "dấu chấm than": "!",
        "xuống dòng": "\n",
        "dấu phẩy": ",",
        "dấu chấm": ".",
    },
}

# --- CJK / no-space constants ----------------------------------------------
# Han, Ext-A, compatibility ideographs and the kana block.
_CJK_RANGES = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff"
_CJK_PUNCTUATION = "，。！？、；：（）「」『』【】…—"
_ASCII_PUNCTUATION_EQUIVALENTS = ",.!?;:()"
_PUNCTUATION_CLASS = re.escape(_CJK_PUNCTUATION + _ASCII_PUNCTUATION_EQUIVALENTS)
_HALFWIDTH_TO_FULLWIDTH = {",": "，", "!": "！", "?": "？", ".": "。"}


def _strip_edges(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punctuation, core, trailing punctuation)."""
    lead = ""
    tail = ""
    match = LEADING.match(token)
    if match:
        lead = match.group(0)
        token = token[len(lead):]
    match = TRAILING.search(token)
    if match:
        tail = match.group(0)
        token = token[: match.start()]
    return lead, token, tail


def _format_integer(value: int) -> str:
    """>= 10000 gets thousands separators, because a typist would add them."""
    if abs(value) >= BIG_NUMBER:
        return f"{value:,}"
    return str(value)


def _numeric_of(rendered: str) -> float | None:
    try:
        return abs(float(rendered.replace(",", "").replace("st", "").replace("nd", "")
                       .replace("rd", "").replace("th", "")))
    except ValueError:
        return None


def parse_run(tokens: list[str], index: int):
    """Parse a run of number words starting at ``index``.

    Returns ``(rendered, next_index, is_ordinal, is_fraction)`` or None.
    Linear in the run length; each token is consumed at most once.
    """
    if index >= len(tokens):
        return None
    cursor = index
    sign = ""
    lowered = tokens[cursor].lower()
    if lowered in ("minus", "negative"):
        sign = "-"
        cursor += 1
        if cursor >= len(tokens):
            return None

    if cursor + 1 < len(tokens):
        pair = (tokens[cursor].lower(), tokens[cursor + 1].lower())
        if pair in FRACTIONS:
            return FRACTIONS[pair], cursor + 2, False, True

    total = 0
    current = 0
    saw = False
    suffix = ""
    while cursor < len(tokens):
        word = tokens[cursor].lower()
        if word in UNITS:
            current += UNITS[word]
            saw = True
        elif word in TENS:
            current += TENS[word]
            saw = True
        elif word in SCALES:
            current = max(current, 1) * SCALES[word]
            saw = True
        elif word in ORDINAL_CARDINAL and saw:
            cardinal, suffix = ORDINAL_CARDINAL[word]
            current += UNITS.get(cardinal, 0) if cardinal in UNITS else SCALES.get(cardinal, 0)
            cursor += 1
            break
        elif word == "and" and saw and cursor + 1 < len(tokens):
            nxt = tokens[cursor + 1].lower()
            if nxt in UNITS or nxt in TENS or nxt in SCALES:
                cursor += 1
                continue
            break
        else:
            break
        cursor += 1

    if not saw:
        return None

    total += current

    if cursor < len(tokens) and tokens[cursor].lower() == "point":
        digits = []
        scan = cursor + 1
        while scan < len(tokens):
            word = tokens[scan].lower()
            if word in UNITS and UNITS[word] < 10:
                digits.append(str(UNITS[word]))
                scan += 1
            elif word.isdigit():
                digits.extend(list(word))
                scan += 1
            else:
                break
        if digits:
            return f"{sign}{_format_integer(total)}." + "".join(digits), scan, False, False

    return f"{sign}{_format_integer(total)}{suffix}", cursor, bool(suffix), False


def _consume_spoken_punctuation(text: str, table: dict | None = None) -> str:
    """Replace standalone spoken punctuation words. O(n) over the tokens.

    Whisper usually punctuates the sentence itself AND leaves the spoken word in,
    so "item, comma second" arrives as "item, comma second". The spoken word has to
    REPLACE whatever punctuation precedes it, not append to it, or the output gets
    ",," and ",." artefacts. The pattern therefore eats any punctuation before the
    phrase.

    ``table`` maps a spoken phrase to its symbol and defaults to the English
    :data:`PUNCTUATION`, so the original behaviour is unchanged. Multi-word
    phrases are matched first so a short entry cannot consume part of a longer
    one; the English and localised tables therefore share one implementation.
    """
    if table is None:
        multi = PUNCTUATION_MULTI
        single = PUNCTUATION_SINGLE
    else:
        multi = {phrase: symbol for phrase, symbol in table.items() if " " in phrase}
        single = {phrase: symbol for phrase, symbol in table.items() if " " not in phrase}
    prefix = r"[,\;:.!?\-]*\s*"
    for phrase, symbol in multi.items():
        text = re.sub(
            rf"{prefix}(?<!\w){re.escape(phrase)}(?!\w)[,\;:.!?\-]*", f" {symbol} ", text, flags=re.I
        )
    tokens = text.split(" ")
    out = []
    for token in tokens:
        lead, core, tail = _strip_edges(token)
        lower = core.lower()
        if lower in single:
            out.append(single[lower])
        else:
            out.append(token)
    return " ".join(out)


def _tidy_spacing(text: str, keep_percent_space: bool = False) -> str:
    """Collapse stray whitespace and punctuation artefacts.

    ``keep_percent_space`` leaves a space before ``%`` alone, which the French
    convention ("12 %") needs. The English pipeline keeps the default so a
    "5 %" that arrives already typed is still tightened to "5%".
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"([,.;:!?])\1+", r"\1", text)
    text = re.sub(r"[,;:]+([.!?])", r"\1", text)
    if keep_percent_space:
        text = re.sub(r"\s+([,.!?;:)\]}])", r"\1", text)
    else:
        text = re.sub(r"\s+([,.!?;:)\]}%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _normalise_tokens(tokens: list[str]) -> list[str]:
    """Rewrite number runs whose written form a typist would use.

    A run is converted only when there is a reason to: it carries a unit, a
    percent sign, a currency, a decimal, an ordinal, a fraction, a math operator,
    a range, or it is large enough (>= 10000) that digits are the natural written
    form. Small bare counts stay words, which is what keeps "I have two cats",
    "one of the best" and "first of all" intact.

    Two passes over the token list, each O(n): one to mark runs, one to emit.
    """
    cores = [_strip_edges(token)[1] for token in tokens]
    lowered = [core.lower() for core in cores]
    total = len(tokens)

    run_at: dict[int, tuple] = {}
    consumed: set[int] = set()
    index = 0
    while index < total:
        parsed = parse_run(lowered, index)
        if parsed is None:
            index += 1
            continue
        rendered, next_index, is_ordinal, is_fraction = parsed
        if next_index <= index:
            index += 1
            continue
        run_at[index] = (rendered, next_index, is_ordinal, is_fraction)
        for position in range(index, next_index):
            consumed.add(position)
        index = next_index

    out: list[str] = []
    index = 0
    last_converted_end = -1
    while index < total:
        if index not in run_at:
            if index in consumed:
                index += 1
                continue
            out.append(tokens[index])
            index += 1
            continue

        rendered, next_index, is_ordinal, is_fraction = run_at[index]
        lead = _strip_edges(tokens[index])[0]
        tail = _strip_edges(tokens[next_index - 1])[2]

        following = ""
        if next_index < total:
            following = lowered[next_index]
        after_to = ""
        after_to_is_number = False
        if following == "to" and next_index + 1 < total:
            after_to = lowered[next_index + 1]
            after_to_is_number = parse_run(lowered, next_index + 1) is not None

        numeric = _numeric_of(rendered)
        previous_word = lowered[index - 1] if index > 0 else ""
        used_scale = any(lowered[position] in SCALES for position in range(index, next_index))
        follows_to_number = previous_word == "to" and last_converted_end == index - 1
        fraction_unit = None
        if is_fraction:
            scan = next_index
            if scan < total and lowered[scan] in ("of",):
                scan += 1
                if scan < total and lowered[scan] in ("a", "an"):
                    scan += 1
            if scan < total and (
                lowered[scan] in UNITS_MAP or lowered[scan] in ("percent", "per")
            ):
                fraction_unit = scan
        fraction_is_measurement = fraction_unit is not None
        significant = (
            is_ordinal
            or fraction_is_measurement
            or (numeric is not None and numeric >= BIG_NUMBER)
            or "." in rendered
            or rendered.startswith("-")
            or used_scale
            or following in UNITS_MAP
            or following in ("percent", "per", "degrees")
            or following in MATH_WORDS
            or following in (SQUARED, CUBED)
            or following == "to" and after_to_is_number
            or previous_word == "from"
            or previous_word in MATH_WORDS
            or follows_to_number
        )
        if not significant and next_index < total:
            following_token = _strip_edges(tokens[next_index])[2]
            if following_token in ("'s",):
                significant = True

        if fraction_unit is not None:
            unit_word = lowered[fraction_unit]
            if unit_word in ("percent", "per"):
                out.append(f"{lead}{rendered}%{tail}")
                index = fraction_unit + 1
            else:
                out.append(f"{lead}{rendered} {UNITS_MAP[unit_word]}{tail}")
                index = fraction_unit + 1
            last_converted_end = index
            continue
        if following in ("percent", "per") and next_index + 1 < total and lowered[next_index + 1] == "cent":
            out.append(f"{lead}{rendered}%{tail}")
            index = next_index + 2
            last_converted_end = index
            continue
        if following == "percent":
            out.append(f"{lead}{rendered}%{tail}")
            index = next_index + 1
            last_converted_end = index
            continue
        if following == "degrees":
            out.append(f"{lead}{rendered}\u00b0{tail}")
            index = next_index + 1
            last_converted_end = index
            continue
        if following in UNITS_MAP:
            out.append(f"{lead}{rendered} {UNITS_MAP[following]}{tail}")
            index = next_index + 1
            last_converted_end = index
            continue

        if not significant:
            for position in range(index, next_index):
                out.append(tokens[position])
            index = next_index
            continue

        out.append(f"{lead}{rendered}{tail}")
        last_converted_end = next_index
        index = next_index

    return out


def _apply_currency(words: list[str]) -> list[str]:
    """Attach a leading currency word to the number that follows it. O(n)."""
    out: list[str] = []
    index = 0
    while index < len(words):
        lead, core, tail = _strip_edges(words[index])
        if core.lower() in CURRENCY and index + 1 < len(words):
            nxt_lead, nxt_core, nxt_tail = _strip_edges(words[index + 1])
            if nxt_core.replace(",", "").replace(".", "").isdigit():
                symbol = CURRENCY[core.lower()]
                if symbol in CURRENCY_SUFFIX:
                    out.append(f"{lead}{nxt_core}{symbol}{nxt_tail}")
                else:
                    out.append(f"{lead}{symbol}{nxt_core}{nxt_tail}")
                index += 2
                continue
        out.append(words[index])
        index += 1
    return out


def _apply_math(text: str) -> str:
    for phrase in ("multiplied by", "divided by", "plus", "minus", "times", "over"):
        symbol = MATH_SYMBOLS[phrase]
        text = re.sub(
            rf"(?<=\d)\s+{re.escape(phrase)}\s+(?=\d)",
            f"{symbol}",
            text,
            flags=re.I,
        )
    text = re.sub(r"(?<=\d)\s+squared\b", "\u00b2", text, flags=re.I)
    text = re.sub(r"(?<=\d)\s+cubed\b", "\u00b3", text, flags=re.I)
    return text


def _apply_ranges(text: str) -> str:
    def replace(match: re.Match) -> str:
        return f"{match.group(1)}-{match.group(2)}"

    text = re.sub(
        r"\bfrom\s+(\d[\d,.]*)\s+to\s+(\d[\d,.]*)\b", replace, text, flags=re.I
    )
    text = re.sub(r"\b(\d[\d,.]*)\s+to\s+(\d[\d,.]*)\b", replace, text)
    return text


def _base_language(code) -> str:
    """A lowercase base language code ("en-US" -> "en"), or "" if not concrete.

    Complexity: O(len(code)).
    """
    if not isinstance(code, str):
        return ""
    return code.strip().lower().replace("_", "-").split("-", 1)[0]


def _resolve_language(config, language) -> str:
    """Pick the effective language: explicit argument, then config, then "en".

    A falsy or "auto" value is not a concrete code and falls through to the next
    source, matching the rule that keep legacy callers working.

    Complexity: O(len(code)).
    """
    base = _base_language(language)
    if base and base != "auto":
        return base
    if config is not None:
        try:
            configured = config.get("language")
        except Exception:
            configured = None
        base = _base_language(configured)
        if base and base != "auto":
            return base
    return "en"


def _collapse_spaces(text: str) -> str:
    """Collapse runs of spaces/tabs and strip the ends. O(n) time and space."""
    return re.sub(r"[ \t]+", " ", text).strip()


def _to_fullwidth_punctuation(text: str) -> str:
    """Turn a half-width , ! ? . straight after a CJK/kana char into full width.

    Complexity: O(n) time and space.
    """
    pattern = re.compile(rf"([{_CJK_RANGES}])([,.!?])")
    return pattern.sub(
        lambda match: match.group(1) + _HALFWIDTH_TO_FULLWIDTH[match.group(2)], text
    )


def _strip_space_around_cjk_punctuation(text: str) -> str:
    """Remove whitespace touching CJK punctuation or its ASCII form. O(n)."""
    return re.sub(rf"[ \t]*([{_PUNCTUATION_CLASS}])[ \t]*", r"\1", text)


def _remove_space_between_cjk(text: str) -> str:
    """Remove whitespace sitting between two CJK/kana characters. O(n)."""
    return re.sub(rf"(?<=[{_CJK_RANGES}])[ \t]+(?=[{_CJK_RANGES}])", "", text)


def _normalise_no_space(text: str, language: str) -> str:
    """Character-level cleanup for languages written without word spaces.

    The space-token pipeline cannot see word boundaries here, and whisper
    already punctuates Chinese and Japanese, so nothing word-based runs:
    spoken-punctuation words, number parsing and currency rules are all
    deliberately skipped as unsafe. Thai is space-delimited between phrases but
    not between words, so it only gets the whitespace tidy-up.

    Complexity: O(n) time and space over the text length.
    """
    if language == "th":
        return _collapse_spaces(text)
    work = _to_fullwidth_punctuation(text)
    work = _strip_space_around_cjk_punctuation(work)
    work = _remove_space_between_cjk(work)
    return _collapse_spaces(work)


def _apply_decimal_rules(text: str, language: str) -> str:
    """Turn ``digit <decimal word> digit`` into a comma decimal.

    A digit either side is required, so "coma" or "koma" only becomes a decimal
    separator between numbers; anywhere else it is left for the punctuation
    table. This runs BEFORE the punctuation pass for exactly that reason.

    Complexity: O(w * n) time and O(n) space over w decimal words.
    """
    for word in DECIMAL_WORDS.get(language, ()):
        text = re.sub(rf"(\d+)\s+{re.escape(word)}\s+(\d+)", r"\1,\2", text, flags=re.I)
    return text


def _apply_percent_rules(text: str, language: str) -> str:
    """Turn ``digit <percent phrase>`` into a percent sign.

    French writes a space before the sign ("12 %"); the other languages do not
    ("12%"). Case-insensitive and anchored on a preceding digit.

    Complexity: O(n) per phrase.
    """
    gap = " " if language == "fr" else ""
    for phrase in PERCENT_PHRASES.get(language, ()):
        text = re.sub(
            rf"(\d[\d.,]*)\s+{re.escape(phrase)}(?!\w)",
            lambda match: f"{match.group(1)}{gap}%",
            text,
            flags=re.I,
        )
    return text


def _apply_currency_rules(text: str, language: str) -> str:
    """Attach a currency symbol after ``digit <currency word>``.

    Only a number immediately before the word triggers this, so the word stays
    untouched in ordinary prose. Longer words are tried first so "euros" cannot
    be half-matched by "euro".

    Complexity: O(n) per currency word.
    """
    table = CURRENCY_SYMBOLS.get(language, {})
    for word in sorted(table, key=len, reverse=True):
        symbol = table[word]
        text = re.sub(
            rf"(\d[\d.,]*)\s+{re.escape(word)}(?!\w)",
            lambda match: f"{match.group(1)} {symbol}",
            text,
            flags=re.I,
        )
    return text


def _normalise_space_delimited(text: str, language: str, spoken_punctuation: bool) -> str:
    """A localised pipeline for space-delimited languages (fr, es, id, vi).

    Only spacing, the language's own spoken punctuation, and the digit-based
    decimal / percent / currency rules run. The English word-based passes
    (``_normalise_tokens``, ``_apply_currency``, ``_apply_math``,
    ``_apply_ranges``) are deliberately NOT used: their tables are English and
    would rewrite another language's words (e.g. "million" in French).

    Complexity: O(n) time and O(n) space apart from the small per-language
    tables.
    """
    work = _tidy_spacing(text, keep_percent_space=True)
    work = _apply_decimal_rules(work, language)
    if spoken_punctuation:
        table = PUNCTUATION_BY_LANGUAGE.get(language)
        if table:
            work = _consume_spoken_punctuation(work, table)
    work = _tidy_spacing(work, keep_percent_space=True)
    work = _apply_percent_rules(work, language)
    work = _apply_currency_rules(work, language)
    return work


def _normalise_english(text: str, spoken_punctuation: bool) -> str:
    """The original English pipeline, unchanged. O(n) time and O(n) space."""
    work = text
    if spoken_punctuation:
        work = _consume_spoken_punctuation(work)

    work = _tidy_spacing(work)
    tokens = _normalise_tokens(work.split(" "))
    tokens = _apply_currency(tokens)
    work = " ".join(tokens)
    work = _apply_math(work)
    work = _apply_ranges(work)
    work = _tidy_spacing(work)
    return work


def normalise(text: str, config=None, language=None) -> str:
    """Convert clearly spoken forms into written forms.

    ``config`` may supply ``spoken_punctuation`` (default True) and
    ``language``. ``language`` overrides the config: the effective code is the
    explicit argument, else the config's concrete code (not "auto"/empty), else
    "en"; codes are reduced to their lowercase base ("en-US" -> "en"). Routing
    keeps every language on the passes that are safe for it:

    * no-space languages (zh, ja, th, yue): character-level cleanup only;
    * "en": the original space-token pipeline, byte-for-byte;
    * digit languages (fr, es, id, vi): a localised pipeline;
    * anything else: whitespace tidy only, the safest default.

    Idempotent: running it twice changes nothing the second time. Text that
    matches no rule is returned unchanged. O(n) time and O(n) space.
    """
    if not text or not text.strip():
        return text or ""

    spoken_punctuation = True
    if config is not None:
        try:
            spoken_punctuation = bool(config.get("spoken_punctuation", True))
        except Exception:
            spoken_punctuation = True

    stripped = text.strip()
    if stripped.lower().startswith("literal "):
        return stripped[len("literal "):]

    effective = _resolve_language(config, language)
    if effective in NO_SPACE_LANGUAGES:
        return _normalise_no_space(text, effective)
    if effective == "en":
        return _normalise_english(text, spoken_punctuation)
    if effective in DIGIT_LANGUAGES:
        return _normalise_space_delimited(text, effective, spoken_punctuation)
    return _tidy_spacing(text)
