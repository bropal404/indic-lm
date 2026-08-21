# Language Models and Agents (LMA), Phase 1: Data and Tokenizers

## **Gopal Kataria, 2023112006**

This repository contains the implementation, data pipelines, and tokenizer models for building two decoder-only Transformer language models from scratch, in two Indian languages:

- Model H (Higher-Resource): Hindi (`hi`)
- Model L (Lower-Resource): Nepali (`ne`)

Drive link for corpus tarzip: [https://drive.google.com/drive/folders/1W5IcGslCwV7zTKIDFokKLg28qfrp4uGY?usp=sharing](https://drive.google.com/drive/folders/1W5IcGslCwV7zTKIDFokKLg28qfrp4uGY?usp=sharing)

---

## Reproduction and Running Steps

### 0. Download and Extract the Data

If you have no local dataset yet, download the corpus archive from the Drive link above and unpack it first.

```bash
./scripts/unzip_corpora.sh corpora_dataset_bundle.tar.gz
```

This populates `hindi/data/` and `nepali/data/` with the raw and cleaned corpus files referenced by every step below. Once extraction finishes, continue with Step 1.

---

### 1. Environment Setup

```bash
uv sync
source .venv/bin/activate
```

---

### 2. Run the Web Crawler

The crawler is located in `crawler/`.

```bash

# Run crawler for Nepali
python3 crawler/cli.py crawl \
  --config nepali/configs/sources.yaml \
  --output nepali/data/raw/clean/nepali_corpus.jsonl \
  --db crawl_state_nepali.db

# Run crawler for Hindi
python3 crawler/cli.py crawl \
  --config hindi/configs/sources.yaml \
  --output hindi/data/raw/clean/hindi_corpus.jsonl \
  --db crawl_state_hindi.db
```

---

### 3. Clean the Dataset

Runs the multi-stage cleaning pipeline: Unicode NFC normalization, boilerplate removal, Devanagari ratio filtering, SHA-256 exact dedup, and MinHash/LSH near-dedup.

```bash
python3 scripts/clean_hindi.py hindi/data/raw/clean/hindi_corpus.jsonl -o hindi/data/raw/clean/

python3 scripts/clean_hindi.py nepali/data/raw/clean/nepali_corpus.jsonl -o nepali/data/raw/clean/ --langs ne
```

---

### 4. Combine Datasets and Generate Splits

Combines self-crawled and curated public datasets into train, validation, and test splits (90/5/5) with corpus metadata.

```bash
python3 scripts/split_and_stats.py

# Or run language-specific preproc essing pipelines
python3 hindi/data/preprocess.py
python3 nepali/data/preprocess.py
```

Outputs produced:

```
hindi/data/{train,val,test}.jsonl
hindi/data/dataset_stats.json
nepali/data/{train,val,test}.jsonl
nepali/data/dataset_stats.json
```

---

### 5. Train Tokenizers

Trains custom BPE tokenizers from scratch (5K, 10K, 20K merges) using the unified tokenizer CLI.

```bash
# Train all 3 Hindi BPE tokenizers
python3 hindi/tokenizer/train.py
# Or a specific one:
uv run tokenizer train --lang hi --input hindi/data/train.jsonl --merges 20000 \
  --output-dir hindi/tokenizer/artifacts_20000 --name bpe_hi_20000

# Train all 3 Nepali BPE tokenizers
python3 nepali/tokenizer/train.py
# Or a specific one:
uv run tokenizer train --lang ne --input nepali/data/train.jsonl --merges 20000 \
  --output-dir nepali/tokenizer/artifacts_20000 --name bpe_ne_20000
```

---

### 6. Evaluate Tokenizers

Benchmarks tokenizers on held-out test splits, measuring fertility, bytes/token, compression ratio, continuation rate, UNK rate, and throughput.

```bash
# Run comprehensive evaluation across both languages
python3 scripts/evaluate_tokenizers.py

# Or evaluate individually
python3 hindi/tokenizer/evaluate.py
python3 nepali/tokenizer/evaluate.py

# Interactive CLI evaluation with live terminal metrics and sample outputs
uv run tokenizer evaluate --lang hi --samples 1000
uv run tokenizer evaluate --lang ne --samples 1000
```

---

### 7. Zip and Unzip Datasets

Scripts to bundle all dataset JSONL files into a single archive, or unpack an existing archive into place.

```bash
# Package all corpus files (train, val, test, scraped, public) into an archive
./scripts/zip_corpora.sh corpora_dataset_bundle.tar.gz

# Unpack archive into hindi/data/ and nepali/data/
./scripts/unzip_corpora.sh corpora_dataset_bundle.tar.gz
```

---

### 8. Generate  Stats and Figures

```bash
python3 scripts/generate_report_stats.py
```