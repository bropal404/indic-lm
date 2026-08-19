#!/usr/bin/env python3
"""Generate Phase 1 report statistics, plots, and LaTeX snippets."""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LANGS = {
    "Hindi": {
        "code": "hi",
        "data_dir": ROOT / "hindi" / "data",
        "tokenizer_eval": ROOT / "hindi" / "tokenizer" / "eval_results.json",
        "tokenizer_model": ROOT / "hindi" / "tokenizer" / "artifacts" / "bpe_hi_32000.pkl",
    },
    "Nepali": {
        "code": "ne",
        "data_dir": ROOT / "nepali" / "data",
        "tokenizer_eval": ROOT / "nepali" / "tokenizer" / "eval_results.json",
        "tokenizer_model": ROOT / "nepali" / "tokenizer" / "artifacts" / "bpe_ne_32000.pkl",
    },
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path, limit: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def estimated_total_tokens(stats: dict) -> int:
    if "total_tokens_bpe" in stats:
        return int(stats["total_tokens_bpe"])
    splits = stats.get("splits", {})
    return sum(int(v.get("estimated_tokens_bpe", v.get("tokens_bpe", 0))) for v in splits.values())


def collection_token_split(stats: dict) -> tuple[int, int, float]:
    total_tokens = estimated_total_tokens(stats)
    token_split = stats.get("token_split", {})
    if "self_crawled_tokens" in token_split:
        manual_tokens = int(token_split["self_crawled_tokens"])
        downloaded_tokens = total_tokens - manual_tokens
        manual_pct = (manual_tokens / total_tokens * 100) if total_tokens else 0.0
        return manual_tokens, downloaded_tokens, manual_pct
    methods = stats.get("collection_methods", {})
    total_docs = max(int(stats.get("total_documents", 0)), 1)
    manual_doc_share = methods.get("self_crawled", 0) / total_docs
    manual_tokens = int(total_tokens * manual_doc_share)
    downloaded_tokens = total_tokens - manual_tokens
    manual_pct = (manual_tokens / total_tokens * 100) if total_tokens else 0.0
    return manual_tokens, downloaded_tokens, manual_pct


def split_tokens(stats: dict, split_name: str) -> int:
    info = stats["splits"][split_name]
    return int(info.get("estimated_tokens_bpe", info.get("tokens_bpe", 0)))


def split_info(stats: dict, split_name: str) -> dict:
    splits = stats["splits"]
    if split_name in splits:
        return splits[split_name]
    if split_name == "validation" and "val" in splits:
        return splits["val"]
    if split_name == "val" and "validation" in splits:
        return splits["validation"]
    raise KeyError(split_name)


def collect_doc_length_stats(data_dir: Path) -> dict:
    lengths = []
    for split in ("train", "val", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            words = row.get("word_count")
            if words is None:
                words = len((row.get("text") or "").split())
            lengths.append(int(words))
    lengths.sort()
    if not lengths:
        return {}
    pct = lambda q: lengths[min(len(lengths) - 1, math.floor((len(lengths) - 1) * q))]
    return {
        "min": lengths[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": lengths[-1],
        "mean": round(statistics.mean(lengths), 2),
    }


def tokenizer_rows(eval_json: dict) -> list[dict]:
    return [
        row
        for row in eval_json.get("models_evaluated", [])
        if str(row.get("name", "")).startswith("BPE-")
    ]


def load_bpe_tokenizer(model_path: Path):
    sys.path.insert(0, str(ROOT))
    from tokenizer.tokenizers import customBPE

    tok = customBPE()
    tok.load(str(model_path))
    return tok


def compute_token_frequency(lang_cfg: dict, sample_docs: int | None) -> Counter:
    tok = load_bpe_tokenizer(lang_cfg["tokenizer_model"])
    counter = Counter()
    for row in iter_jsonl(lang_cfg["data_dir"] / "test.jsonl", limit=sample_docs):
        counter.update(tok.tokenize(row.get("text", "")))
    return counter


def plot_corpus_totals(stats_by_lang: dict, figures_dir: Path) -> Path:
    labels = list(stats_by_lang)
    docs = [stats_by_lang[name]["total_documents"] for name in labels]
    words = [stats_by_lang[name]["total_words"] / 1_000_000 for name in labels]
    tokens = [estimated_total_tokens(stats_by_lang[name]) / 1_000_000 for name in labels]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, values, title, ylabel in [
        (axes[0], docs, "Documents", "documents"),
        (axes[1], words, "Words", "million words"),
        (axes[2], tokens, "Estimated BPE Tokens", "million tokens"),
    ]:
        ax.bar(labels, values, color=["#3366aa", "#dd8844"], label=title)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Language")
        ax.legend()
    fig.tight_layout()
    path = figures_dir / "corpus_totals.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_source_distribution(stats_by_lang: dict, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (lang, stats) in zip(axes, stats_by_lang.items()):
        top = stats["source_distribution"][:10]
        labels = [item["source"] for item in reversed(top)]
        values = [item["document_count"] for item in reversed(top)]
        ax.barh(labels, values, color="#4f8a8b", label="documents")
        ax.set_title(f"Top {len(top)} {lang} Sources")
        ax.set_xlabel("Documents")
        ax.set_ylabel("Source")
        ax.legend()
    fig.tight_layout()
    path = figures_dir / "source_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_tokenizer_sweep(eval_by_lang: dict, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lang, eval_json in eval_by_lang.items():
        rows = tokenizer_rows(eval_json)
        xs = [int(str(r["name"]).replace("BPE-", "").replace("K", "")) for r in rows]
        fertility = [r["fertility"] for r in rows]
        unk = [r["unk_rate_pct"] for r in rows]
        axes[0].plot(xs, fertility, marker="o", label=lang)
        axes[1].plot(xs, unk, marker="o", label=lang)
    axes[0].set_title("Vocabulary Sweep: Fertility")
    axes[0].set_xlabel("BPE merge budget (K)")
    axes[0].set_ylabel("Tokens per word")
    axes[0].legend()
    axes[1].set_title("Vocabulary Sweep: UNK Rate")
    axes[1].set_xlabel("BPE merge budget (K)")
    axes[1].set_ylabel("UNK rate (%)")
    axes[1].legend()
    fig.tight_layout()
    path = figures_dir / "tokenizer_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_zipf(freqs_by_lang: dict[str, Counter], figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for lang, freqs in freqs_by_lang.items():
        counts = sorted(freqs.values(), reverse=True)
        ranks = range(1, len(counts) + 1)
        ax.loglog(ranks, counts, label=lang)
    ax.set_title("Token Frequency Distribution on Test Splits")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "token_frequency_zipf.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def latex_table_corpus(stats_by_lang: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Language & Docs & Words & Est. BPE tokens & UTF-8 MB & Avg. words/doc & Avg. Devanagari \\",
        r"\midrule",
    ]
    for lang, stats in stats_by_lang.items():
        lines.append(
            f"{lang} & {fmt_int(stats['total_documents'])} & {fmt_int(stats['total_words'])} & "
            f"{fmt_int(estimated_total_tokens(stats))} & {fmt_float(stats['total_size_mb'])} & "
            f"{fmt_float(stats['average_words_per_doc'])} & {fmt_float(stats['average_devanagari_ratio'] * 100)}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_table_collection(stats_by_lang: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Language & Total tokens & Manual tokens & Downloaded tokens & Manual share \\",
        r"\midrule",
    ]
    for lang, stats in stats_by_lang.items():
        manual, downloaded, pct = collection_token_split(stats)
        lines.append(
            f"{lang} & {fmt_int(estimated_total_tokens(stats))} & {fmt_int(manual)} & "
            f"{fmt_int(downloaded)} & {fmt_float(pct)}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_table_splits(stats_by_lang: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Language & Split & Docs & Words & Est. BPE tokens \\",
        r"\midrule",
    ]
    for lang, stats in stats_by_lang.items():
        for split in ("train", "validation", "test"):
            s = split_info(stats, split)
            lines.append(
                f"{lang} & {split.capitalize()} & {fmt_int(s['documents'])} & "
                f"{fmt_int(s['words'])} & {fmt_int(int(s.get('estimated_tokens_bpe', s.get('tokens_bpe', 0))))} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_table_tokenizers(eval_by_lang: dict) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Language & Tokenizer & Vocab & Fertility & Chars/token & UNK \% & Tok/s \\",
        r"\midrule",
    ]
    for lang, eval_json in eval_by_lang.items():
        for row in tokenizer_rows(eval_json):
            lines.append(
                f"{lang} & {row['name']} & {row['vocab_size']} & {row['fertility']} & "
                f"{row['chars_per_token']} & {row['unk_rate_pct']} & {fmt_int(row['tokens_per_sec'])} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_table_lengths(length_stats: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Language & Min & P50 & Mean & P90 & P95 & P99 & Max \\",
        r"\midrule",
    ]
    for lang, stats in length_stats.items():
        lines.append(
            f"{lang} & {stats['min']} & {stats['p50']} & {stats['mean']} & "
            f"{stats['p90']} & {stats['p95']} & {stats['p99']} & {stats['max']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_snippets(path: Path, tables: dict[str, str], figures: dict[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name, table in tables.items():
            f.write(f"% {name}\n{table}\n\n")
        f.write("% figures\n")
        for name, figure_path in figures.items():
            f.write(f"% {name}: {figure_path.relative_to(ROOT)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 report stats and plots.")
    parser.add_argument("--figures-dir", default=str(ROOT / "report" / "figures"))
    parser.add_argument("--snippets", default=str(ROOT / "report" / "generated_stats_tables.tex"))
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=None,
        help="Limit test documents for token-frequency plots. Default: use full test split.",
    )
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats_by_lang = {
        lang: load_json(cfg["data_dir"] / "dataset_stats.json") for lang, cfg in LANGS.items()
    }
    eval_by_lang = {lang: load_json(cfg["tokenizer_eval"]) for lang, cfg in LANGS.items()}
    length_stats = {lang: collect_doc_length_stats(cfg["data_dir"]) for lang, cfg in LANGS.items()}

    print("\nCorpus totals")
    for lang, stats in stats_by_lang.items():
        manual, downloaded, pct = collection_token_split(stats)
        print(
            f"{lang}: {fmt_int(stats['total_documents'])} docs, "
            f"{fmt_int(stats['total_words'])} words, "
            f"{fmt_int(estimated_total_tokens(stats))} est. BPE tokens, "
            f"{fmt_float(pct)}% manual"
        )

    print("\nDocument length percentiles")
    for lang, stats in length_stats.items():
        print(
            f"{lang}: p50={stats['p50']}, mean={stats['mean']}, "
            f"p95={stats['p95']}, max={stats['max']} words"
        )

    print("\nTokenizer sweep")
    for lang, eval_json in eval_by_lang.items():
        for row in tokenizer_rows(eval_json):
            print(
                f"{lang} {row['name']}: vocab={row['vocab_size']}, "
                f"fertility={row['fertility']}, chars/token={row['chars_per_token']}, "
                f"unk={row['unk_rate_pct']}%"
            )

    freqs_by_lang = {
        lang: compute_token_frequency(cfg, args.sample_docs) for lang, cfg in LANGS.items()
    }
    for lang, freqs in freqs_by_lang.items():
        total = sum(freqs.values())
        top_100 = sum(count for _, count in freqs.most_common(100))
        coverage = (top_100 / total * 100) if total else 0.0
        print(f"{lang}: {fmt_int(len(freqs))} unique test tokens; top-100 coverage={coverage:.2f}%")

    figures = {
        "corpus_totals": plot_corpus_totals(stats_by_lang, figures_dir),
        "source_distribution": plot_source_distribution(stats_by_lang, figures_dir),
        "tokenizer_sweep": plot_tokenizer_sweep(eval_by_lang, figures_dir),
        "token_frequency_zipf": plot_zipf(freqs_by_lang, figures_dir),
    }
    tables = {
        "corpus_table": latex_table_corpus(stats_by_lang),
        "collection_table": latex_table_collection(stats_by_lang),
        "split_table": latex_table_splits(stats_by_lang),
        "tokenizer_table": latex_table_tokenizers(eval_by_lang),
        "length_table": latex_table_lengths(length_stats),
    }
    write_snippets(Path(args.snippets), tables, figures)

    print("\nGenerated plots")
    for path in figures.values():
        print(f"- {path.relative_to(ROOT)}")
    print(f"\nLaTeX snippets: {Path(args.snippets).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
