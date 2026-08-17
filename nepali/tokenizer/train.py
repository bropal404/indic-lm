#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tokenizer.cli import main

if __name__ == "__main__":
    for merges in [5000, 10000, 20000]:
        print(f"\n{'=' * 60}")
        print(f"Training Nepali tokenizer with {merges} merges")
        print(f"{'=' * 60}\n")

        sys.argv = [
            sys.argv[0],
            "train",
            "--lang", "ne",
            "--input", "nepali/data/train.jsonl",
            "--output-dir", f"nepali/tokenizer/artifacts_{merges}",
            "--merges", str(merges),
            "--name", f"bpe_ne_{merges}",
        ]

        main()