import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict


def format_num(n: float) -> str:
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(int(n) if isinstance(n, int) or (isinstance(n, float) and n.is_integer()) else f"{n:.2f}")


class CorpusStatistics:
    def __init__(self):
        self.documents = 0
        self.cleaned_chars = 0
        self.raw_chars = 0
        self.whitespace_words = 0
        self.by_source = defaultdict(lambda: {"documents": 0, "characters": 0, "words": 0})
        self.by_category = Counter()
        self.by_year = Counter()

    def add(self, row: Dict[str, object]) -> None:
        text = str(row.get("text") or "")
        source = str(row.get("source") or "unknown")
        words = len(text.split())
        self.documents += 1
        self.cleaned_chars += len(text)
        self.raw_chars += int(row.get("raw_char_count") or len(text))
        self.whitespace_words += words
        self.by_source[source]["documents"] += 1
        self.by_source[source]["characters"] += len(text)
        self.by_source[source]["words"] += words
        if row.get("category"):
            self.by_category[str(row["category"])] += 1
        published = str(row.get("published_at") or "")
        if len(published) >= 4 and published[:4].isdigit():
            self.by_year[published[:4]] += 1

    def load_jsonl(self, path: str) -> None:
        p = Path(path)
        paths = list(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for file_path in paths:
            with file_path.open(encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        self.add(json.loads(line))

    def report(self) -> str:
        lines = [
            f"{'source':<18} {'docs':>8} {'chars':>10} {'words':>10} {'tokens (1.5x)':>14} {'tokens (2.0x)':>14}",
            "-" * 80,
        ]
        for source, vals in sorted(self.by_source.items()):
            docs_str = format_num(vals["documents"])
            chars_str = format_num(vals["characters"])
            words_str = format_num(vals["words"])
            tok_15 = format_num(vals["words"] * 1.5)
            tok_20 = format_num(vals["words"] * 2.0)
            lines.append(f"{source:<18} {docs_str:>8} {chars_str:>10} {words_str:>10} {tok_15:>14} {tok_20:>14}")
        lines.append("-" * 80)
        total_docs = format_num(self.documents)
        total_chars = format_num(self.cleaned_chars)
        total_words = format_num(self.whitespace_words)
        total_tok_15 = format_num(self.whitespace_words * 1.5)
        total_tok_20 = format_num(self.whitespace_words * 2.0)
        lines.append(f"{'TOTAL':<18} {total_docs:>8} {total_chars:>10} {total_words:>10} {total_tok_15:>14} {total_tok_20:>14}")
        lines.append("-" * 80)
        lines.append(f"Raw totals: {self.documents:,} docs | {self.cleaned_chars:,} chars | {self.whitespace_words:,} words")
        lines.append(f"Estimated token range: {total_tok_15} – {total_tok_20} ({int(self.whitespace_words * 1.5):,} to {int(self.whitespace_words * 2.0):,} tokens @ 1.5–2.0 tok/word)")
        return "\n".join(lines)
