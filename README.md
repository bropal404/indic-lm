# GOPAL KATARIA

# 2023 112 006 



# Language Models and Agents (LMA) — Phase 1: Data & Tokenizers

This repository contains the complete implementation, data pipelines, and tokenizer models for building two independent decoder-only Transformer language models from scratch in two Indian languages:
- **Model H (Higher-Resource)**: **Hindi (`hi`)**
- **Model L (Lower-Resource)**: **Nepali (`ne`)**

---

## Reproduction & Running Steps

### 1. Environment Setup

```bash
# Sync dependencies and activate environment
uv sync
source .venv/bin/activate
```

---

### 2. Run Web Crawler

The asynchronous, robots-compliant crawler is located in `crawler/`.

```bash
# Preview sources and inspect robots.txt / sitemaps
python3 crawler/cli.py recon --config nepali/configs/sources.yaml
python3 crawler/cli.py recon --config hindi/configs/sources.yaml

# Run crawler for Nepali
python3 crawler/cli.py crawl --config nepali/configs/sources.yaml --output nepali/data/raw/clean/nepali_corpus.jsonl --db crawl_state_nepali.db

# Run crawler for Hindi
python3 crawler/cli.py crawl --config hindi/configs/sources.yaml --output hindi/data/raw/clean/hindi_corpus.jsonl --db crawl_state_hindi.db
```

---

### 3. Clean Dataset

Run the multi-stage cleaning pipeline (Unicode NFC normalization, boilerplate removal, Devanagari ratio filtering, SHA-256 exact dedup, and MinHash/LSH near-dedup):

```bash
# Clean raw Hindi crawler dumps
python3 scripts/clean_hindi.py hindi/data/raw/clean/hindi_corpus.jsonl -o hindi/data/raw/clean/

# Clean Nepali crawler dumps
python3 scripts/clean_hindi.py nepali/data/raw/clean/nepali_corpus.jsonl -o nepali/data/raw/clean/ --langs ne
```

---

### 4. Combine Datasets & Generate 90 / 5 / 5 Splits

Combine self-crawled and curated public datasets into final train, validation, and test splits with complete corpus metadata:

```bash
# Generate document-level 90/5/5 splits and dataset_stats.json for both languages
python3 scripts/split_and_stats.py

# Or run language-specific preprocessing pipelines
python3 hindi/data/preprocess.py
python3 nepali/data/preprocess.py
```
Outputs produced:
- `hindi/data/{train,val,test}.jsonl` & `hindi/data/dataset_stats.json`
- `nepali/data/{train,val,test}.jsonl` & `nepali/data/dataset_stats.json`

---

### 5. Train Tokenizers

Train custom BPE tokenizers from scratch (5K, 10K, 20K merges) using the unified tokenizer CLI:

```bash
# Train Hindi BPE tokenizers
python3 hindi/tokenizer/train.py
# Or via CLI directly:
uv run tokenizer train --lang hi --input hindi/data/train.jsonl --merges 20000 --output-dir hindi/tokenizer/artifacts_20000 --name bpe_hi_20000

# Train Nepali BPE tokenizers
python3 nepali/tokenizer/train.py
# Or via CLI directly:
uv run tokenizer train --lang ne --input nepali/data/train.jsonl --merges 20000 --output-dir nepali/tokenizer/artifacts_20000 --name bpe_ne_20000
```

---

### 6. Evaluate Tokenizers

Benchmark tokenizers on held-out test splits (measuring Fertility, Bytes/Token, Compression Ratio, Continuation Rate, UNK Rate, and Throughput):

```bash
# Run comprehensive evaluation across both languages
python3 scripts/evaluate_tokenizers.py

# Or evaluate individually
python3 hindi/tokenizer/evaluate.py
python3 nepali/tokenizer/evaluate.py

# Interactive CLI evaluation with live terminal metrics & sample outputs
uv run tokenizer evaluate --lang hi --samples 1000
uv run tokenizer evaluate --lang ne --samples 1000
```

---

### 7. Zip & Unzip Datasets

Scripts to bundle all heavy dataset JSONL files into a single archive or unpack them to their target directories:

```bash
# Package all corpus files (train, val, test, scraped, public) into an archive
./scripts/zip_corpora.sh corpora_dataset_bundle.tar.gz

# Unpack archive into hindi/data/ and nepali/data/
./scripts/unzip_corpora.sh corpora_dataset_bundle.tar.gz
```

---

### 8. Generate Report Stats & Figures


```bash
python3 scripts/generate_report_stats.py
```
