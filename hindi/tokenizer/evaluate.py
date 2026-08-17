#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.evaluate_tokenizers import run_language_evaluation, SAMPLE_SENTENCES_HI
if __name__ == "__main__":
    run_language_evaluation(
        lang_name="Hindi",
        lang_code="hi",
        test_path="hindi/data/test.jsonl",
        artifacts_dir="hindi/tokenizer/artifacts",
        output_dir="hindi/tokenizer",
        sample_sentences=SAMPLE_SENTENCES_HI
    )
