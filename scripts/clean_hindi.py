#!/usr/bin/env python3
"""
Fast Hindi/Bhojpuri cleaning pipeline for crawler raw dumps.

Stages:
  1. Boilerplate/HTML extraction with trafilatura when HTML is present.
  2. Unicode normalization to NFC.
  3. Script filtering by Devanagari ratio.
  4. Length filtering.
  5. Language ID with ftlangdetect plus Bhojpuri lexical hints.
  6. Exact deduplication by SHA-256 of normalized text.
  7. Near deduplication with MinHash/LSH over word shingles.
"""

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

try:
    import trafilatura
except ImportError:  # pragma: no cover - optional runtime fallback
    trafilatura = None

try:
    from ftlangdetect import detect as ft_detect
except ImportError:  # pragma: no cover - optional runtime fallback
    ft_detect = None


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_HINT_RE = re.compile(r"<!doctype|<html|<body|<article|<p[\s>]|<div[\s>]", re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)

BOILERPLATE_RE = re.compile(
    r"("
    r"click here|read more|also read|share\s+(this|tweet|on)|"
    r"subscribe|newsletter|follow us|advertisement|"
    r"privacy policy|terms of (use|service)|cookie policy|"
    r"copyright|all rights reserved|"
    r"टिप्पणी|प्रतिक्रिया|शेयर|सब्सक्राइब|विज्ञापन|"
    r"अगली खबर|और पढ़ें|यह भी पढ़ें"
    r")",
    re.IGNORECASE,
)

BHOJPURI_HINTS = {
    "बा",
    "बाड़े",
    "बाटे",
    "बानी",
    "बाड़ू",
    "बाड़ऽ",
    "हई",
    "हउवे",
    "होखेला",
    "होला",
    "भइल",
    "भईल",
    "गइल",
    "गईल",
    "कइल",
    "कहल",
    "रउआ",
    "रउरा",
    "हमनी",
    "तोहरा",
    "कइसे",
    "काहे",
    "एह",
    "ओह",
}

DEFAULT_MIN_DEVA_RATIO = 0.35
DEFAULT_MIN_TOKENS = 20
DEFAULT_MAX_TOKENS = 50_000
DEFAULT_ACCEPTED_LANGS = {"hi", "bho"}
DEFAULT_LANGID_MIN_SCORE = 0.30
DEFAULT_BHOJPURI_HINT_THRESHOLD = 0.015
DEFAULT_MINHASH_THRESHOLD = 0.86
DEFAULT_MINHASH_NUM_PERM = 128
DEFAULT_SHINGLE_N = 5


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": True, "_source_file": path, "_line_no": line_no}
                continue
            obj.setdefault("_source_file", path)
            obj.setdefault("_line_no", line_no)
            yield obj


def looks_like_html(text: str) -> bool:
    return bool(text and HTML_HINT_RE.search(text[:4000]))


def extract_main_text(raw: str, use_trafilatura: bool = True) -> str:
    if not raw:
        return ""
    if use_trafilatura and trafilatura is not None and looks_like_html(raw):
        extracted = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted:
            return extracted
    return HTML_TAG_RE.sub(" ", raw)


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"[\u200c\u200d]{2,}", "", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    paragraphs = []
    seen = set()
    for para in (p.strip() for p in re.split(r"\n{2,}", text)):
        if not para or BOILERPLATE_RE.search(para):
            continue
        key = re.sub(r"\s+", " ", para)
        if len(key) < 240 and key in seen:
            continue
        seen.add(key)
        paragraphs.append(key)
    return "\n\n".join(paragraphs).strip()


def devanagari_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if "\u0900" <= ch <= "\u097F") / len(chars)


def token_count(text: str) -> int:
    return len(text.split())


def tokenize_for_dedup(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_minhash(text: str, n: int, num_perm: int) -> MinHash:
    tokens = tokenize_for_dedup(text)
    shingles = tokens if len(tokens) < n else [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


_LANGID_CACHE: dict[str, tuple[str, float]] = {}


def fasttext_langid(text: str) -> tuple[str, float]:
    sample = text[:1000].replace("\n", " ")
    key = sample[:300]
    if key in _LANGID_CACHE:
        return _LANGID_CACHE[key]
    if ft_detect is None:
        out = ("unk", 0.0)
    else:
        try:
            result = ft_detect(sample, low_memory=True)
            out = (str(result.get("lang", "unk")), float(result.get("score", 0.0)))
        except Exception:
            out = ("unk", 0.0)
    _LANGID_CACHE[key] = out
    return out


def bhojpuri_hint_score(text: str) -> float:
    tokens = tokenize_for_dedup(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in BHOJPURI_HINTS)
    return hits / len(tokens)


def detect_hindi_bhojpuri(text: str, bhojpuri_hint_threshold: float) -> tuple[str, float, str]:
    lang, score = fasttext_langid(text)
    hint_score = bhojpuri_hint_score(text)
    if lang == "bho":
        return "bho", score, "fasttext"
    if lang == "hi" and hint_score >= bhojpuri_hint_threshold:
        return "bho", max(score, hint_score), "bhojpuri_hint"
    return lang, score, "fasttext"


def reject(writer, obj: dict, reason: str, text: str | None = None) -> None:
    row = dict(obj)
    row["_reject_reason"] = reason
    if text is not None:
        row["text"] = text
    writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pipeline(
    input_files: list[str],
    output_dir: str,
    min_deva_ratio: float = DEFAULT_MIN_DEVA_RATIO,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    accepted_langs: set[str] = DEFAULT_ACCEPTED_LANGS,
    langid_min_score: float = DEFAULT_LANGID_MIN_SCORE,
    bhojpuri_hint_threshold: float = DEFAULT_BHOJPURI_HINT_THRESHOLD,
    minhash_threshold: float = DEFAULT_MINHASH_THRESHOLD,
    num_perm: int = DEFAULT_MINHASH_NUM_PERM,
    shingle_n: int = DEFAULT_SHINGLE_N,
    text_field: str = "text",
    raw_text_field: str = "raw_text",
    skip_langid: bool = False,
    skip_neardedup: bool = False,
    skip_trafilatura: bool = False,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    language_counts = Counter()
    source_counts = Counter()
    exact_seen: set[str] = set()
    lsh = MinHashLSH(threshold=minhash_threshold, num_perm=num_perm) if not skip_neardedup else None
    minhash_id = 0

    with ExitStack() as stack:
        writers = {
            "clean": stack.enter_context(open(out / "clean.jsonl", "w", encoding="utf-8")),
            "script": stack.enter_context(open(out / "rejected_script.jsonl", "w", encoding="utf-8")),
            "length": stack.enter_context(open(out / "rejected_length.jsonl", "w", encoding="utf-8")),
            "langid": stack.enter_context(open(out / "rejected_langid.jsonl", "w", encoding="utf-8")),
            "dedup": stack.enter_context(open(out / "rejected_dedup.jsonl", "w", encoding="utf-8")),
            "parse": stack.enter_context(open(out / "rejected_parse.jsonl", "w", encoding="utf-8")),
        }
        lang_writers = {
            lang: stack.enter_context(open(out / f"clean_{lang}.jsonl", "w", encoding="utf-8"))
            for lang in sorted(accepted_langs)
        }

        for input_file in input_files:
            print(f"\nProcessing {input_file} ...")
            for obj in tqdm(iter_jsonl(input_file), desc=Path(input_file).name):
                if obj.get("_parse_error"):
                    counts["parse_error"] += 1
                    reject(writers["parse"], obj, "json_parse_error")
                    continue

                counts["input"] += 1
                raw = obj.get(raw_text_field) or obj.get("html") or obj.get("raw_html") or obj.get(text_field) or ""
                text = obj.get(text_field) or raw
                if looks_like_html(raw):
                    text = extract_main_text(raw, use_trafilatura=not skip_trafilatura)

                text = normalize_text(text)
                text = remove_boilerplate(text)
                text = normalize_text(text)
                if not text:
                    counts["empty_after_cleaning"] += 1
                    reject(writers["length"], obj, "empty_after_cleaning")
                    continue

                dr = devanagari_ratio(text)
                if dr < min_deva_ratio:
                    counts["rejected_script"] += 1
                    reject(writers["script"], obj, f"devanagari_ratio={dr:.4f} < {min_deva_ratio}", text)
                    continue

                words = token_count(text)
                if words < min_tokens or words > max_tokens:
                    counts["rejected_length"] += 1
                    reject(writers["length"], obj, f"token_count={words}", text)
                    continue

                detected_lang = "und"
                langid_score = 0.0
                langid_method = "skipped"
                bho_hint = bhojpuri_hint_score(text)
                if not skip_langid:
                    detected_lang, langid_score, langid_method = detect_hindi_bhojpuri(text, bhojpuri_hint_threshold)
                    if detected_lang not in accepted_langs or langid_score < langid_min_score:
                        counts["rejected_langid"] += 1
                        reject(
                            writers["langid"],
                            obj,
                            f"lang={detected_lang} score={langid_score:.4f}",
                            text,
                        )
                        continue

                digest = content_hash(text)
                if digest in exact_seen:
                    counts["rejected_exact_dedup"] += 1
                    reject(writers["dedup"], obj, "exact_duplicate", text)
                    continue
                exact_seen.add(digest)

                if lsh is not None:
                    mh = make_minhash(text, shingle_n, num_perm)
                    neighbors = lsh.query(mh)
                    if neighbors:
                        counts["rejected_near_dedup"] += 1
                        reject(writers["dedup"], obj, f"near_duplicate_of={neighbors[0]}", text)
                        continue
                    lsh.insert(f"doc_{minhash_id}", mh)
                    minhash_id += 1

                row = dict(obj)
                row[text_field] = text
                row["char_count"] = len(text)
                row["word_count"] = words
                row["devanagari_ratio"] = round(dr, 4)
                row["content_hash"] = digest
                row["bhojpuri_hint_score"] = round(bho_hint, 4)
                if not skip_langid:
                    row["detected_lang"] = detected_lang
                    row["langid_score"] = round(langid_score, 4)
                    row["langid_method"] = langid_method

                serialized = json.dumps(row, ensure_ascii=False) + "\n"
                writers["clean"].write(serialized)
                if detected_lang in lang_writers:
                    lang_writers[detected_lang].write(serialized)
                counts["clean"] += 1
                language_counts[detected_lang] += 1
                source_counts[row.get("source", "unknown")] += 1

    total_in = counts["input"]
    total_out = counts["clean"]
    rejection_rate = round((1 - total_out / total_in) * 100, 2) if total_in else 0
    report = {
        "input_files": input_files,
        "output_dir": str(out),
        "config": {
            "unicode_normalization": "NFC",
            "min_devanagari_ratio": min_deva_ratio,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "accepted_langs": sorted(accepted_langs),
            "langid_min_score": langid_min_score,
            "bhojpuri_hint_threshold": bhojpuri_hint_threshold,
            "minhash_threshold": minhash_threshold if not skip_neardedup else "disabled",
            "minhash_num_perm": num_perm,
            "shingle_n": shingle_n,
            "skip_langid": skip_langid,
            "skip_neardedup": skip_neardedup,
            "skip_trafilatura": skip_trafilatura,
        },
        "dependencies": {
            "trafilatura": trafilatura is not None,
            "ftlangdetect": ft_detect is not None,
            "datasketch": True,
        },
        "counts": dict(counts),
        "language_counts": dict(language_counts),
        "top_sources": source_counts.most_common(25),
        "total_input": total_in,
        "total_clean_output": total_out,
        "overall_rejection_pct": rejection_rate,
        "rejection_breakdown": {
            "script_filter_pct": round(counts["rejected_script"] / total_in * 100, 2) if total_in else 0,
            "length_filter_pct": round(counts["rejected_length"] / total_in * 100, 2) if total_in else 0,
            "empty_after_cleaning_pct": round(counts["empty_after_cleaning"] / total_in * 100, 2) if total_in else 0,
            "langid_filter_pct": round(counts["rejected_langid"] / total_in * 100, 2) if total_in else 0,
            "exact_dedup_pct": round(counts["rejected_exact_dedup"] / total_in * 100, 2) if total_in else 0,
            "near_dedup_pct": round(counts["rejected_near_dedup"] / total_in * 100, 2) if total_in else 0,
        },
        "outputs": {
            "clean": str(out / "clean.jsonl"),
            "clean_by_language": {lang: str(out / f"clean_{lang}.jsonl") for lang in sorted(accepted_langs)},
            "rejected_script": str(out / "rejected_script.jsonl"),
            "rejected_length": str(out / "rejected_length.jsonl"),
            "rejected_langid": str(out / "rejected_langid.jsonl"),
            "rejected_dedup": str(out / "rejected_dedup.jsonl"),
            "rejected_parse": str(out / "rejected_parse.jsonl"),
            "report": str(out / "cleaning_report.json"),
        },
    }

    report_path = out / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("CLEANING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Input docs           : {total_in:,}")
    print(f"  Clean output         : {total_out:,} ({100 - rejection_rate:.1f}% pass rate)")
    print(f"  Language split       : {dict(language_counts)}")
    print(f"  Rejected - script    : {counts['rejected_script']:,}")
    print(f"  Rejected - length    : {counts['rejected_length']:,}")
    print(f"  Rejected - lang ID   : {counts['rejected_langid']:,}")
    print(f"  Rejected - exact dup : {counts['rejected_exact_dedup']:,}")
    print(f"  Rejected - near dup  : {counts['rejected_near_dedup']:,}")
    print(f"  Report saved         : {report_path}")
    print(f"  Clean data           : {out / 'clean.jsonl'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Hindi/Bhojpuri crawler JSONL dumps.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL file(s).")
    parser.add_argument("-o", "--output-dir", default="hindi/data/raw/cleaned")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--raw-text-field", default="raw_text")
    parser.add_argument("--min-deva", type=float, default=DEFAULT_MIN_DEVA_RATIO)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--langs", nargs="+", default=sorted(DEFAULT_ACCEPTED_LANGS))
    parser.add_argument("--langid-score", type=float, default=DEFAULT_LANGID_MIN_SCORE)
    parser.add_argument("--bhojpuri-hint-threshold", type=float, default=DEFAULT_BHOJPURI_HINT_THRESHOLD)
    parser.add_argument("--minhash-threshold", type=float, default=DEFAULT_MINHASH_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_MINHASH_NUM_PERM)
    parser.add_argument("--shingle-n", type=int, default=DEFAULT_SHINGLE_N)
    parser.add_argument("--skip-langid", action="store_true")
    parser.add_argument("--skip-neardedup", action="store_true")
    parser.add_argument("--skip-trafilatura", action="store_true")
    args = parser.parse_args()

    if not args.skip_langid and ft_detect is None:
        print("ftlangdetect is not installed. Re-run with --skip-langid or install ftlangdetect.", file=sys.stderr)
        raise SystemExit(2)

    run_pipeline(
        input_files=args.inputs,
        output_dir=args.output_dir,
        min_deva_ratio=args.min_deva,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        accepted_langs=set(args.langs),
        langid_min_score=args.langid_score,
        bhojpuri_hint_threshold=args.bhojpuri_hint_threshold,
        minhash_threshold=args.minhash_threshold,
        num_perm=args.num_perm,
        shingle_n=args.shingle_n,
        text_field=args.text_field,
        raw_text_field=args.raw_text_field,
        skip_langid=args.skip_langid,
        skip_neardedup=args.skip_neardedup,
        skip_trafilatura=args.skip_trafilatura,
    )


if __name__ == "__main__":
    main()
