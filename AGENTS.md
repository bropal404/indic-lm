# AGENTS.md

Monolingual decoder-only Transformer LMs in **Hindi (Model H)** and **Nepali (Model L)**.
Pipeline has two phases: data + tokenizers (Phase 1, mostly complete), then LM training (Phase 2).

## Layout

- `crawler/` — async crawler. Entrypoint `crawler/cli.py` → `crawler` console script.
- `tokenizer/` — BPE / whitespace / regex tokenizers. Entrypoint `tokenizer/cli.py` → `tokenizer` console script.
- `scripts/` — pipeline scripts (clean, split, evaluate, report stats). Entrypoints `clean-hindi` and `report-stats`.
- `hindi/` and `nepali/` — per-language working trees with their own `configs/`, `data/`, `tokenizer/`, `model/`, `train/`, `eval/`, `checkpoints/`, `logs/`, `results/`. Both follow the same shape.
- `tests/` — pytest suite (only `test_nepali_crawler_components.py` is present).
- `report/` — LaTeX/PDF phase reports and figures.
- Root artifacts: `LMA_Individual_Project_v1.pdf`, `corpora_dataset_bundle.tar.gz` (raw corpora bundle, gitignored).

The README's "Phase 2" training entrypoint `phase2/train.py` referenced in `monitor.sh` does **not** exist — training/eval live at `hindi/train/training.py` and `hindi/eval/evaluation.py`. `monitor.sh` is stale; don't trust its `phase2/` paths.

## Environment

- Python 3.14 pinned via `.python-version`. `uv` is the package manager.
- `uv sync` then `source .venv/bin/activate` to set up.
- Heavy artifacts (datasets, checkpoints, raw html, `.db` crawler state) are `.gitignore`d — they are produced by the pipeline, not checked in.

## Developer commands

```bash
# Run all tests
uv run pytest

# Crawl (Phase 1 step 2) — needs the corpus bundle unpacked first
python3 crawler/cli.py crawl --config nepali/configs/sources.yaml --output nepali/data/raw/clean/nepali_corpus.jsonl --db crawl_state_nepali.db
python3 crawler/cli.py crawl --config hindi/configs/sources.yaml  --output hindi/data/raw/clean/hindi_corpus.jsonl  --db crawl_state_hindi.db

# Clean + dedupe
python3 scripts/clean_hindi.py hindi/data/raw/clean/hindi_corpus.jsonl  -o hindi/data/raw/clean/
python3 scripts/clean_hindi.py nepali/data/raw/clean/nepali_corpus.jsonl -o nepali/data/raw/clean/ --langs ne

# Split + dataset stats
python3 hindi/data/preprocess.py    # also reads nepali/data/clean/final.jsonl
python3 nepali/data/preprocess.py

# Train BPE tokenizers (5K/10K/20K merges for each language)
python3 hindi/tokenizer/train.py
python3 nepali/tokenizer/train.py

# Single tokenizer via CLI
uv run tokenizer train --lang hi --input hindi/data/train.jsonl --merges 20000 --output-dir hindi/tokenizer/artifacts_20000 --name bpe_hi_20000

# Evaluate tokenizers
python3 scripts/evaluate_tokenizers.py
uv run tokenizer evaluate --lang hi --samples 1000
uv run tokenizer evaluate --lang ne --samples 1000

# Report stats + figures
python3 scripts/generate_report_stats.py

# Bundle / unbundle corpora (large, gitignored)
./scripts/zip_corpora.sh corpora_dataset_bundle.tar.gz
./scripts/unzip_corpora.sh corpora_dataset_bundle.tar.gz
```

`scripts/clean_hindi.py` is the cleaner for both languages despite its name — pass `--langs ne` for Nepali. The Hindi cleaner is also exposed as the `clean-hindi` console script.

## Pipeline order (do not skip)

1. Unpack `corpora_dataset_bundle.tar.gz` → `hindi/data/`, `nepali/data/`.
2. (Optional) Crawl + clean → `*/data/raw/clean/*_corpus.jsonl`.
3. `preprocess.py` → `*/data/{train,val,test}.jsonl` + `dataset_stats.json|.md`. Seed is `RANDOM_SEED = 42`; splits are 90/5/5 with a hard `devanagari_ratio >= 0.3` and `len(text) >= 50` filter.
4. `*/tokenizer/train.py` → `*/tokenizer/artifacts_{N}/`.
5. Evaluation / report stats.

Train/val/test JSONLs are gitignored — regenerate them; do not assume they exist after a fresh clone.

## Tokenizer artifacts and configs

- Trained tokenizers land in `hindi/tokenizer/artifacts_{5000,10000,20000}/` and `nepali/tokenizer/artifacts_{5000,10000,20000}/`.
- `tokenizer/cli.py` defaults `--input` to `hindi_data/clean/final.jsonl` / `data/clean/final.jsonl` (legacy paths). Use the per-language `hindi/data/train.jsonl` / `nepali/data/train.jsonl` instead — the legacy defaults will not resolve under the current layout.
- `tokenizer evaluate` auto-picks the most recently modified `bpe_<lang>_*.pkl` under `--model-dir` if `--model` is omitted.

## Conventions / gotchas

- Devanagari ratio is the dominant signal everywhere — crawler (`min_devanagari_ratio` per source in `*/configs/sources.yaml`), preprocessing (`>= 0.3`), and tokenizer inputs all assume it.
- `tokenizer/tokenizers.py` is the canonical tokenizer implementation; `tokenizer/train_bpe_tokenizer.py` is a separate script (not the default entry path).
- `scripts/clean_hindi.py` requires `datasketch`, `tqdm`, and optionally `trafilatura` and `ftlangdetect` — the latter two are imported with soft-fail fallbacks, so missing them just degrades stages.
- `pyproject.toml` sets `pythonpath = ["."]` for pytest, so tests can import `crawler.*` directly from repo root.
- `pyproject.toml` declares `setuptools` packages `["crawler", "crawler.extractors", "tokenizer", "scripts"]` — `hindi` and `nepali` are not packages.
- `crawler/cli.py` raises the soft FD limit (`RLIMIT_NOFILE`) toward 65535 at import time.
- Random seed `42` is used by `preprocess.py` and both model configs — keep it stable for reproducible splits.
- Generated content: `dataset_stats.json`, `dataset_stats.md`, `eval_report.md`, `eval_results.json`, `tokenized_sample_sentences.json`, `report/figures/`, `report/tables/`, `report/plots/`, `report/heatmaps/`, `report/generations/` are pipeline outputs (most are not gitignored — review before committing).
- `reports/reconnaissance.md` is a stale top-level reconnaissance report from earlier in the project; superseded by `report/phase1_report.md`.

## LM training (Phase 2)

- `hindi/configs/model_h.yaml` and `nepali/configs/model_l.yaml` are the source of truth for hyperparameters.
- `d_model=384, n_layers=6, n_heads=6, d_ff=1536, context_length=256`, weight tying on. CPU-optimized (no mixed precision).
- Tokenizer paths in configs reference `bpe_hi_32000.pkl` / `bpe_ne_32000.pkl`, but only 5K/10K/20K tokenizers are actually trained by `*/tokenizer/train.py`. Either retrain at 32K merges or update the configs to point at an existing artifact before training.
- `hindi/train/training.py` defines a `Trainer` class driven by these configs; `hindi/eval/evaluation.py` defines loss/perplexity + BLEU/ROUGE helpers. There is no top-level `python -m ...` driver — invoke via your own entrypoint or update `monitor.sh`'s `phase2/train.py` path.
