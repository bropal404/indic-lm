#!/usr/bin/env python3
"""
Tokenize Hindi dataset and push directly to Hugging Face Hub (bropal404/lma-project).

Splits processed:
  - train.jsonl -> train
  - val.jsonl   -> validation
  - test.jsonl  -> test

Each sample is chunked into context_length token sequences:
  - input_ids: list of int32 token IDs of length context_length
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.tokenizers import RegexTokenizer


class FastBPETokenizer:
    """Fast tokenizer using trained merge ranks and vocabulary dict."""

    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or "vocab" not in data or "merges" not in data:
            raise ValueError(f"Invalid tokenizer pickle at {pkl_path}")

        self.vocab: Dict[str, int] = data["vocab"]
        self.merges: Dict[tuple, str] = data["merges"]
        self.unk_id: int = self.vocab.get("<unk>", 0)
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges.keys())}
        self.pre_tokenizer = RegexTokenizer()
        self._token_cache = {}

    def _tokenize_word(self, word: str) -> List[str]:
        cached = self._token_cache.get(word)
        if cached is not None:
            return cached

        tokens = list(word) + ["</w>"]
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = None
            earliest_rank = float("inf")
            for pair in pairs:
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < earliest_rank:
                    earliest_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            left, right = best_pair
            merged = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == left and tokens[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        cleaned = [t.replace("</w>", "") for t in tokens if t != "</w>"]
        cleaned = [t for t in cleaned if t]
        if len(self._token_cache) < 100000:
            self._token_cache[word] = cleaned
        return cleaned

    def encode(self, text: str) -> List[int]:
        words = self.pre_tokenizer.tokenize(text)
        token_ids = []
        for word in words:
            for piece in self._tokenize_word(word):
                token_ids.append(self.vocab.get(piece, self.unk_id))
        return token_ids


def process_jsonl_to_arrow_chunks(
    jsonl_path: str,
    tokenizer: FastBPETokenizer,
    context_length: int = 256,
    max_samples: int = None,
    output_parquet: str = None,
    batch_size: int = 10000,
):
    """Tokenizes JSONL and writes directly to Parquet to keep RAM low."""
    schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
    ])

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    writer = pq.ParquetWriter(output_parquet, schema, compression="snappy")

    current_chunk = []
    batch_records = []
    total_written = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc=f"Processing {Path(jsonl_path).name}")):
            if max_samples and idx >= max_samples:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            text = item.get("text", "")
            if not text:
                continue

            ids = tokenizer.encode(text)
            if not ids:
                continue

            # Concatenate into continuous token stream chunked to context_length
            current_chunk.extend(ids)

            while len(current_chunk) >= context_length:
                seq = current_chunk[:context_length]
                current_chunk = current_chunk[context_length:]
                batch_records.append(seq)

                if len(batch_records) >= batch_size:
                    table = pa.Table.from_arrays([pa.array(batch_records, type=pa.list_(pa.int32()))], schema=schema)
                    writer.write_table(table)
                    total_written += len(batch_records)
                    batch_records = []

    # Remainder padding if at least 16 tokens remain
    if len(current_chunk) >= 16:
        pad_len = context_length - len(current_chunk)
        seq = current_chunk + [0] * pad_len
        batch_records.append(seq)

    if batch_records:
        table = pa.Table.from_arrays([pa.array(batch_records, type=pa.list_(pa.int32()))], schema=schema)
        writer.write_table(table)
        total_written += len(batch_records)

    writer.close()
    print(f"Finished {jsonl_path} -> {output_parquet} ({total_written:,} sequences)")
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Tokenize and upload Hindi dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="bropal404/lma-project", help="HF Repo ID")
    parser.add_argument("--tokenizer-path", type=str, default="hindi/tokenizer/artifacts/bpe_hi_32000.pkl", help="Path to tokenizer pkl")
    parser.add_argument("--data-dir", type=str, default="hindi/data", help="Path to hindi/data dir containing train/val/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/tokenized_hf", help="Temporary dir for parquet files")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API Token")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Debug option: limit train lines")
    parser.add_argument("--upload-only", action="store_true", help="Skip tokenization and only upload existing parquet files")
    args = parser.parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        print("[WARNING] HF_TOKEN is not set. You can set HF_TOKEN environment variable or pass --hf-token.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl"),
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }

    if not args.upload_only:
        print(f"Loading tokenizer from: {args.tokenizer_path}")
        tokenizer = FastBPETokenizer(args.tokenizer_path)

        for split_name, jsonl_file in splits.items():
            if not os.path.exists(jsonl_file):
                print(f"File {jsonl_file} not found, skipping split {split_name}.")
                continue
            max_s = args.max_train_samples if split_name == "train" else None
            pq_out = str(out_dir / f"{split_name}.parquet")
            process_jsonl_to_arrow_chunks(
                jsonl_path=jsonl_file,
                tokenizer=tokenizer,
                context_length=args.context_length,
                max_samples=max_s,
                output_parquet=pq_out,
            )

    print("\nLoading Parquet files into DatasetDict...")
    ds_dict = {}
    for split_name in ["train", "validation", "test"]:
        pq_path = out_dir / f"{split_name}.parquet"
        if pq_path.exists():
            ds_dict[split_name] = Dataset.from_parquet(str(pq_path))

    dataset = DatasetDict(ds_dict)
    print(f"Prepared DatasetDict: {dataset}")

    print(f"\nPushing dataset to Hugging Face Hub: {args.repo_id}...")
    dataset.push_to_hub(
        repo_id=args.repo_id,
        token=token,
        private=False,
    )
    print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
