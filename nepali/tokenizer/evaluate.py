#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.evaluate_tokenizers import run_language_evaluation, SAMPLE_SENTENCES_NE
if __name__ == "__main__":
    run_language_evaluation(
        lang_name="Nepali",
        lang_code="ne",
        test_path="nepali/data/test.jsonl",
        artifacts_dir="nepali/tokenizer/artifacts",
        output_dir="nepali/tokenizer",
        sample_sentences=SAMPLE_SENTENCES_NE
    )
