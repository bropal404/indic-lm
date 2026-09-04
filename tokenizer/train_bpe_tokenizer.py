#!/usr/bin/env -S uv run python
import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from tokenizers import BPE_MERGES, clean_text, customBPE


def read_documents(input_paths, language, text_field):
    documents = []

    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        print(f"Reading {path}...")
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in tqdm(f, miniters=100):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = item.get(text_field, "")
                    if not text:
                        continue

                    cleaned = clean_text(text, language)
                    if cleaned:
                        documents.append(cleaned)
        else:
            with open(path, "r", encoding="utf-8") as f:
                cleaned = clean_text(f.read(), language)
                if cleaned:
                    documents.append(cleaned)

    return documents


def resolve_input_paths(args):
    if args.input:
        return args.input

    if args.data_dir and args.lang:
        return [Path(args.data_dir) / f"cc100_{args.lang}.jsonl"]

    raise ValueError("Provide --input, or provide both --data-dir and --lang.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a fast indexed BPE tokenizer and save reusable artifacts."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="One or more .jsonl or .txt files. JSONL files use --text-field.",
    )
    parser.add_argument(
        "--data-dir",
        help="Dataset directory containing cc100_<lang>.jsonl. Used when --input is not set.",
    )
    parser.add_argument("--lang", choices=["en", "mn", "hi"], help="Language code for naming and cleaning.")
    parser.add_argument("--text-field", default="text", help="JSONL field containing document text.")
    parser.add_argument("--merges", type=int, default=BPE_MERGES, help="Maximum number of BPE merges.")
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=2,
        help="Stop when the best pair appears fewer than this many times.",
    )
    parser.add_argument("--train-limit", type=int, help="Maximum number of documents to train on.")
    parser.add_argument("--seed", type=int, default=69, help="Shuffle seed used with --train-limit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tokenizer",
        help="Directory for tokenizer artifacts.",
    )
    parser.add_argument(
        "--name",
        help="Artifact basename. Defaults to bpe_<lang>_<merges>, or bpe_custom_<merges>.",
    )
    args = parser.parse_args()

    input_paths = resolve_input_paths(args)
    language = "english" if args.lang == "en" else "non-english"
    documents = read_documents(input_paths, language, args.text_field)

    if not documents:
        raise ValueError("No training documents were loaded.")

    if args.train_limit and len(documents) > args.train_limit:
        random.seed(args.seed)
        random.shuffle(documents)
        documents = documents[: args.train_limit]

    model_name = args.name or f"bpe_{args.lang or 'custom'}_{args.merges}"
    tokenizer = customBPE(num_merges=args.merges, min_pair_frequency=args.min_pair_frequency)

    print(f"Training on {len(documents)} documents...")
    tokenizer.train(documents)
    artifacts = tokenizer.save_artifacts(args.output_dir, model_name)

    print("\nSaved tokenizer artifacts:")
    for artifact_type, path in artifacts.items():
        print(f"  {artifact_type}: {path}")


if __name__ == "__main__":
    main()
