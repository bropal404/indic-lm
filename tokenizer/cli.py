import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.tokenizers import (
    WhitespaceTokenizer,
    RegexTokenizer,
    customBPE,
    load_data,
    split_data,
)

DEFAULT_HINDI_DATA = "hindi_data/clean/final.jsonl"
DEFAULT_NEPALI_DATA = "data/clean/final.jsonl"


def run_train(args):
    print("=" * 60)
    print("TOKENIZER TRAINING")
    print("=" * 60)

    input_file = args.input
    if not input_file:
        if args.lang == "hi":
            input_file = DEFAULT_HINDI_DATA
        elif args.lang == "ne":
            input_file = DEFAULT_NEPALI_DATA
        else:
            raise ValueError("Must specify --input or a valid --lang ('hi' or 'ne').")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from: {input_file}")
    t0 = time.time()
    data = load_data(input_file, text_field=args.text_field, limit=args.limit)
    print(f"Loaded {len(data):,} documents in {time.time() - t0:.2f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.name or f"bpe_{args.lang or 'custom'}_{args.merges}"

    if args.type == "bpe":
        tokenizer = customBPE(num_merges=args.merges, min_pair_frequency=args.min_pair_freq)
        t0 = time.time()
        tokenizer.train(data)
        train_time = time.time() - t0
        print(f"Training completed in {train_time:.2f}s")
        artifacts = tokenizer.save_artifacts(output_dir, model_name)
        print("\nSaved artifacts:")
        for k, v in artifacts.items():
            print(f"  {k:<10}: {v}")
    else:
        print(f"Tokenizer type '{args.type}' requires no training. Saving config only...")
        config = {
            "type": args.type,
            "lang": args.lang,
            "documents_processed": len(data),
        }
        cfg_path = output_dir / f"{model_name}_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"  config    : {cfg_path}")


def run_evaluate(args):
    print("=" * 105)
    print("TOKENIZER EVALUATION & BENCHMARKING")
    print("=" * 105)

    input_file = args.input
    if not input_file:
        if args.lang == "hi":
            input_file = DEFAULT_HINDI_DATA
        elif args.lang == "ne":
            input_file = DEFAULT_NEPALI_DATA
        else:
            raise ValueError("Must specify --input or a valid --lang ('hi' or 'ne').")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading evaluation dataset: {input_file}")
    t0 = time.time()
    data = load_data(input_file, text_field=args.text_field, limit=args.limit)
    print(f"Loaded {len(data):,} documents in {time.time() - t0:.2f}s")

    train_data, val_data, test_data = split_data(data)
    eval_set = test_data[: args.samples] if args.samples else test_data
    print(f"Using {len(eval_set):,} test samples for evaluation.")

    # Calculate dataset aggregate ground truths
    total_raw_chars = sum(len(doc) for doc in eval_set)
    total_raw_bytes = sum(len(doc.encode("utf-8")) for doc in eval_set)
    total_raw_words = sum(len(doc.split()) for doc in eval_set)
    mb_raw = total_raw_bytes / (1024 * 1024)

    print(f"Test split volume: {total_raw_words:,} words | {total_raw_chars:,} chars | {total_raw_bytes:,} bytes ({mb_raw:.2f} MB)")

    # Initialize Tokenizers
    tokenizers = {
    }

    # Search for BPE model artifacts
    model_dir = Path(args.model_dir)
    bpe_model_path = args.model
    if not bpe_model_path and args.lang:
        # First check specific merges if requested
        if args.merges:
            candidate = model_dir / f"bpe_{args.lang}_{args.merges}.pkl"
            if candidate.exists():
                bpe_model_path = str(candidate)

        # If not found or merges was default, auto-find any trained model for this language
        if not bpe_model_path:
            matches = sorted(
                model_dir.glob(f"bpe_{args.lang}_*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                bpe_model_path = str(matches[0])

    if bpe_model_path and os.path.exists(bpe_model_path):
        print(f"Loading BPE model: {bpe_model_path}")
        bpe_tok = customBPE()
        bpe_tok.load(bpe_model_path)
        tokenizers["BPE"] = bpe_tok
    else:
        print("Note: No pre-trained BPE model found or provided. Evaluating Whitespace & Regex.")

    print("\n" + "=" * 105)
    header = (
        f"{'Tokenizer':<12} | "
        f"{'Fertility (↓)':<13} | "
        f"{'Comp (B/tok)':<12} | "
        f"{'Comp (C/tok)':<12} | "
        f"{'Cont. Rate':<11} | "
        f"{'Tok/sec (↑)':<13} | "
        f"{'Throughput':<12}"
    )
    print(header)
    print("=" * 105)

    results = {}
    for name, tok in tokenizers.items():
        t0 = time.time()
        total_tokens = 0
        multi_subword_words = 0
        total_word_count = 0

        for doc in tqdm(eval_set):
            tokens = tok.tokenize(doc)
            total_tokens += len(tokens)

            # Continuation rate calculation: check each whitespace word
            words = doc.split()
            total_word_count += len(words)
            for w in words:
                w_tokens = tok.tokenize(w)
                if len(w_tokens) > 1:
                    multi_subword_words += 1

        elapsed = time.time() - t0

        fertility = total_tokens / total_raw_words if total_raw_words > 0 else 0
        bytes_per_tok = total_raw_bytes / total_tokens if total_tokens > 0 else 0
        chars_per_tok = total_raw_chars / total_tokens if total_tokens > 0 else 0
        continuation_rate = (multi_subword_words / total_word_count * 100.0) if total_word_count > 0 else 0.0
        tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        mb_per_sec = mb_raw / elapsed if elapsed > 0 else 0

        results[name] = {
            "total_tokens": total_tokens,
            "fertility": fertility,
            "bytes_per_tok": bytes_per_tok,
            "chars_per_tok": chars_per_tok,
            "continuation_rate": continuation_rate,
            "tok_per_sec": tok_per_sec,
            "mb_per_sec": mb_per_sec,
            "elapsed": elapsed,
        }

        row = (
            f"{name:<12} | "
            f"{fertility:>10.2f} w | "
            f"{bytes_per_tok:>10.2f} B | "
            f"{chars_per_tok:>10.2f} C | "
            f"{continuation_rate:>9.1f} % | "
            f"{tok_per_sec:>11,.0f} | "
            f"{mb_per_sec:>8.2f} MB/s"
        )
        print(row)


    print("\n--- SAMPLE TOKENIZATION ---")
    sample_docs = eval_set[: min(3, len(eval_set))]
    for idx, doc in enumerate(sample_docs, 1):
        print(f"\n[Sample {idx}] Document (first 120 chars): {doc[:120]}...")
        for name, tok in tokenizers.items():
            sample_tokens = tok.tokenize(doc[:120])
            joined = " | ".join(sample_tokens[:20])
            if len(sample_tokens) > 20:
                joined += " ... [truncated]"
            print(f"  {name:<12}: {joined}")

    print("\nEvaluation complete.")



def main():
    parser = argparse.ArgumentParser(description="CLI for Tokenizer Training & Evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train BPE tokenizer on dataset")
    train_parser.add_argument("--lang", choices=["hi", "ne", "en"], help="Language code ('hi' for Hindi, 'ne' for Nepali)")
    train_parser.add_argument("--input", help="Path to input jsonl/txt dataset")
    train_parser.add_argument("--text-field", default="text", help="JSONL field containing text")
    train_parser.add_argument("--type", choices=["bpe", "whitespace", "regex"], default="bpe", help="Tokenizer type")
    train_parser.add_argument("--merges", type=int, default=10000, help="Number of BPE merges")
    train_parser.add_argument("--min-pair-freq", type=int, default=2, help="Minimum pair frequency to merge")
    train_parser.add_argument("--limit", type=int, help="Limit number of documents to read")
    train_parser.add_argument("--output-dir", default="tokenizer/artifacts", help="Directory to save tokenizer artifacts")
    train_parser.add_argument("--name", help="Custom model artifact name")

    # EVALUATE command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate tokenizers on dataset")
    eval_parser.add_argument("--lang", choices=["hi", "ne", "en"], help="Language code ('hi' for Hindi, 'ne' for Nepali)")
    eval_parser.add_argument("--input", help="Path to input jsonl/txt dataset")
    eval_parser.add_argument("--text-field", default="text", help="JSONL field containing text")
    eval_parser.add_argument("--model", help="Path to pre-trained BPE model (.pkl)")
    eval_parser.add_argument("--model-dir", default="tokenizer/artifacts", help="Directory containing BPE models")
    eval_parser.add_argument("--merges", type=int, default=None, help="Specific merges of model to look for (default: auto-detect latest)")
    eval_parser.add_argument("--samples", type=int, default=1000, help="Number of test samples to evaluate")
    eval_parser.add_argument("--limit", type=int, help="Limit total dataset loading")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
