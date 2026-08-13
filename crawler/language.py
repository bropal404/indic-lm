import re
import string
from dataclasses import dataclass, asdict
from typing import Dict

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}।॥“”‘’…]")


@dataclass
class LanguageStats:
    devanagari_ratio: float
    latin_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    nepali_unicode_ratio: float
    char_count: int
    language_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def calculate_language_stats(text: str) -> LanguageStats:
    chars = [ch for ch in text if not ch.isspace()]
    total = len(chars)
    if total == 0:
        return LanguageStats(0, 0, 0, 0, 0, 0, 0)
    joined = "".join(chars)
    dev = len(DEVANAGARI_RE.findall(joined))
    latin = len(LATIN_RE.findall(joined))
    digits = len(DIGIT_RE.findall(joined))
    punct = len(PUNCT_RE.findall(joined))
    nepali_unicode = sum(1 for ch in joined if "\u0900" <= ch <= "\u097F" or ch in "।॥")
    dev_ratio = dev / total
    latin_ratio = latin / total
    digit_ratio = digits / total
    punct_ratio = punct / total
    nepali_ratio = nepali_unicode / total
    score = max(0.0, min(1.0, nepali_ratio - (latin_ratio * 0.45) - max(0.0, punct_ratio - 0.20)))
    return LanguageStats(dev_ratio, latin_ratio, digit_ratio, punct_ratio, nepali_ratio, len(text), score)


def is_predominantly_nepali(
    text: str,
    min_devanagari_ratio: float = 0.62,
    min_nepali_unicode_ratio: float = 0.58,
    max_latin_ratio: float = 0.22,
) -> bool:
    stats = calculate_language_stats(text)
    return (
        stats.devanagari_ratio >= min_devanagari_ratio
        and stats.nepali_unicode_ratio >= min_nepali_unicode_ratio
        and stats.latin_ratio <= max_latin_ratio
    )
