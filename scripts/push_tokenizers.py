#!/usr/bin/env python3
"""
Push 20k BPE tokenizer artifacts to Hugging Face Hub.

Supports both Hindi (hi) and Nepali (ne) languages via --lang flag.
Uses HF_TOKEN_GOPAL for authentication.

Script ensures 20k tokenizer is trained/pushed to HF Hub.
Uses the fixed 20k merges which is sure to exist for both languages
and works best for the transformer model.

Usage:
    python3 scripts/push_tokenizers.py --lang hi    # For Hindi
    python3 scripts/push_tokenizers.py --lang ne    # For Nepali
    python3 scripts/push_tokenizers.py              # Default: Nepali

Requires:
    - HF_TOKEN_GOPAL environment variable set for Hugging Face authentication
    - Training data available at hindi/data/train.jsonl / nepali/data/train.jsonl
"""

import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_LANG = "ne"  # Change to "hi" for Hindi


def push_tokenizer_to_hf(tokenizer_dir: Path, repo_id: str, hf_token: str):
    """Push a single tokenizer artifact directory to HF Hub."""
    print(f"Pushing tokenizer from {tokenizer_dir} to {repo_id}...")

    # Upload the entire directory as artifacts
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    # Upload each file in the directory
    for file_path in tokenizer_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(tokenizer_dir)
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"artifacts/{relative_path}",
                repo_id=repo_id,
                token=hf_token,
            )
            print(f"  Uploaded: {relative_path}")


def main():
    """Main function to push 20k tokenizer artifacts to HF Hub."""
    hf_token = os.getenv("HF_TOKEN_GOPAL")
    if not hf_token:
        print("[ERROR] HF_TOKEN_GOPAL environment variable not set!")
        print("Please set it export HF_TOKEN_GOPAL='your_token_here'")
        return 1

    print(f"Using HF token: {hf_token[:10]}...")

    # Determine language
    import argparse
    parser = argparse.ArgumentParser(description="Push 20k tokenizer artifacts to HF Hub")
    parser.add_argument("--lang", choices=["hi", "ne"], default=DEFAULT_LANG,
                        help="Language: hi for Hindi, ne for Nepali")
    args = parser.parse_args()

    lang = args.lang
    repo_id_map = {
        "hi": "bropal/lma-project",
        "ne": "bropal/lma-project-nepali",
    }
    repo_id = repo_id_map[lang]

    print(f"Target HF repo: {repo_id}")

    # Create repo if it doesn't exist
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Ensured repo exists: {repo_id}")
    except Exception as e:
        print(f"[WARNING] Could not create repo: {e}")

    # Define the 20k merge size (fixed for both languages)
    merges = 20000
    tokenizer_dir = Path(f"hindi/tokenizer/artifacts_{merges}") if lang == "hi" else Path(f"nepali/tokenizer/artifacts_{merges}")

    if not tokenizer_dir.exists():
        print(f"[WARNING] Tokenizer directory not found: {tokenizer_dir}")
        print(f"  Training 20k tokenizer first...")

        # Train the tokenizer using the CLI
        print(f"\nTraining BPE tokenizer with {merges} merges...")
        sys.argv = [
            "tokenizer",
            "train",
            "--lang", lang,
            "--input", f"{lang}_data/train.jsonl",
            "--output-dir", f"{lang}_tokenizer/artifacts_{merges}",
            "--merges", str(merges),
            "--name", f"bpe_{lang}_{merges}",
        ]
        from tokenizer.cli import main as tokenizer_main
        tokenizer_main()
        print(f"20k Tokenizer {merges}K trained successfully!")
    else:
        print(f"Using existing 20k tokenizer: {tokenizer_dir}")

    # Push to HF Hub
    print(f"\n{'='*60}")
    print(f"Pushing bpe_{lang}_{merges} tokenizer to HF Hub...")
    print(f"{'='*60}")

    try:
        push_tokenizer_to_hf(tokenizer_dir, repo_id, hf_token)
        print(f"[OK] Successfully pushed bpe_{lang}_{merges} tokenizer!")
    except Exception as e:
        print(f"[ERROR] Failed to push tokenizer: {e}")
        return 1

    print()

    print("="*60)
    print(f"20k tokenizer pushed to HF Hub successfully!")
    print(f"Repo: {repo_id}")
    print(f"Tokenizer: bpe_{lang}_{merges}.pkl (artifacts_{merges}/)")
    print(f"Use this tokenizer with: '{lang}_tokenizer/artifacts_{merges}/bpe_{lang}_{merges}.pkl'")
    print("="*60)
    print()

    print(f"You can load the 20k tokenizer in your notebook/code with:")
    print(f"  tokenizer_path = '{lang}_tokenizer/artifacts_{merges}/bpe_{lang}_{merges}.pkl'")
    print(f"  tokenizer = BPETokenizer.load(tokenizer_path)")


if __name__ == "__main__":
    exit(main())