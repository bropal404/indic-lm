import html
import re
import unicodedata

ZERO_WIDTH_RE = re.compile(r"[\u200b\ufeff]")
ZWJ_NOISE_RE = re.compile(r"[\u200c\u200d]{2,}")


def repair_mojibake(text: str) -> str:
    if not text:
        return ""
    # Common failure mode: UTF-8 Nepali bytes decoded as Latin-1/Windows-1252.
    if "à¤" in text or "à¥" in text:
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                return repaired
        except UnicodeError:
            pass
    return text


def normalize_unicode(text: str) -> str:
    text = html.unescape(text or "")
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = ZWJ_NOISE_RE.sub("", text)
    return text


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"([!?.,।॥]){4,}", r"\1\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = []
    seen = set()
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        key = re.sub(r"\s+", " ", para)
        if key in seen and len(key) < 220:
            continue
        seen.add(key)
        paragraphs.append(key)
    return "\n\n".join(paragraphs).strip()
