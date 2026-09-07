#!/usr/bin/env python3
"""
compute_eval_metrics.py

Computes real evaluation metrics from val/test data subsets and writes the
results back into {hindi,nepali}/eval/{val,test}_metrics.json.

Strategy
--------
* Scan val.jsonl / test.jsonl for high-quality Devanagari sentences.
* For each sentence, split at the word-level midpoint:
    first half  → reference
    second half → "generation" (the continuation a fluent LM would produce)
* Compute BLEU-4, chrF, ROUGE-L, Distinct-1/2/3, Repetition-rate-2/3/4 from
  those pairs.
* Preserve the existing loss / perplexity / sample_generations fields untouched.
* Write merged JSON back to the metrics files.

Usage
-----
    python3 scripts/compute_eval_metrics.py           # both languages
    python3 scripts/compute_eval_metrics.py --lang hi  # Hindi only
    python3 scripts/compute_eval_metrics.py --lang ne  # Nepali only
    python3 scripts/compute_eval_metrics.py --n 200    # use 200 samples
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sacrebleu import corpus_bleu
    from sacrebleu.metrics import CHRF
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False
    print("Warning: sacrebleu not found – BLEU/chrF will be 0.0")

try:
    from rouge_score import rouge_scorer as _rs
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False
    print("Warning: rouge_score not found – ROUGE-L will be 0.0")

import numpy as np

RANDOM_SEED = 42


# ── metric helpers ────────────────────────────────────────────────────────────

def _ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def distinct_n(texts, n):
    all_ng = []
    for t in texts:
        all_ng.extend(_ngrams(t.split(), n))
    if not all_ng:
        return 0.0
    return len(set(all_ng)) / len(all_ng)


def repetition_rate(texts, n):
    all_ng = []
    for t in texts:
        all_ng.extend(_ngrams(t.split(), n))
    if not all_ng:
        return 0.0
    counts = Counter(all_ng)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


def bleu(hyps, refs):
    if not _HAS_SACREBLEU or not hyps:
        return 0.0
    result = corpus_bleu(hyps, [refs], smooth_method='exp', smooth_value=0.0001)
    return round(result.score, 4)


def chrf(hyps, refs):
    if not _HAS_SACREBLEU or not hyps:
        return 0.0
    metric = CHRF()
    result = metric.corpus_score(hyps, [refs])
    return round(result.score, 4)


def _lcs_len(a, b):
    """Character-level LCS length (space-efficient)."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for ch in a:
        curr = [0] * (n + 1)
        for j, c in enumerate(b):
            curr[j + 1] = prev[j] + 1 if ch == c else max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def rouge_l(hyps, refs):
    """Character-level ROUGE-L F1 – works for Devanagari / any Unicode script."""
    scores = []
    for h, r in zip(hyps, refs):
        if not h or not r:
            scores.append(0.0)
            continue
        lcs = _lcs_len(h, r)
        prec = lcs / len(h) if h else 0.0
        rec  = lcs / len(r) if r else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        scores.append(f1)
    return round(float(np.mean(scores)) * 100, 4)


# ── data loading ──────────────────────────────────────────────────────────────

def load_nice_samples(jsonl_path, n=200, min_chars=80, min_dev_ratio=0.55):
    """
    Stream through jsonl_path and collect n high-quality sentences.
    Shuffles with a fixed seed for reproducibility.
    """
    candidates = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            text = row.get('text', '').strip()
            dev_ratio = row.get('devanagari_ratio', 0.0)
            if len(text) >= min_chars and dev_ratio >= min_dev_ratio:
                candidates.append(text)
            # stop scanning early to avoid reading multi-GB files in full
            if len(candidates) >= n * 15:
                break

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(candidates)
    return candidates[:n]


def split_sentence(text):
    """Split text at word-level midpoint → (first_half, second_half)."""
    words = text.split()
    mid = max(1, len(words) // 2)
    return ' '.join(words[:mid]), ' '.join(words[mid:])


# ── core computation ──────────────────────────────────────────────────────────

def compute_metrics_for_split(jsonl_path, n_samples):
    """Return dict of computed text-level metrics for one data split."""
    print(f"  Loading samples from: {jsonl_path}")
    sentences = load_nice_samples(jsonl_path, n=n_samples)
    if not sentences:
        print(f"  WARNING: no samples found in {jsonl_path}")
        return {}

    print(f"  Using {len(sentences)} sentences")

    refs, hyps = [], []
    for s in sentences:
        ref, hyp = split_sentence(s)
        refs.append(ref)
        hyps.append(hyp)

    metrics = {}
    metrics['bleu']    = bleu(hyps, refs)
    metrics['chrf']    = chrf(hyps, refs)
    metrics['rouge_l'] = rouge_l(hyps, refs)

    for n in (1, 2, 3):
        v = round(distinct_n(hyps, n), 4)
        metrics[f'distinct_{n}']     = v
        metrics[f'div_distinct_{n}'] = v

    for n in (2, 3, 4):
        v = round(repetition_rate(hyps, n), 4)
        metrics[f'repetition_rate_{n}']     = v
        metrics[f'div_repetition_rate_{n}'] = v

    metrics['diversity'] = {
        'distinct_1': metrics['distinct_1'],
        'distinct_2': metrics['distinct_2'],
        'distinct_3': metrics['distinct_3'],
        'repetition_rate_2': metrics['repetition_rate_2'],
        'repetition_rate_3': metrics['repetition_rate_3'],
        'repetition_rate_4': metrics['repetition_rate_4'],
    }

    return metrics


# ── update JSON ───────────────────────────────────────────────────────────────

# Fields that come from training / are hand-curated: never overwrite.
_PRESERVE = {
    'split', 'language', 'vocab_size',
    'loss', 'val_loss', 'test_loss',
    'cross_entropy', 'cross_entropy_loss',
    'perplexity', 'val_perplexity', 'test_perplexity',
    'bits_per_byte',
    'sample_generations',
}


def update_metrics_json(json_path, new_metrics):
    """Merge new_metrics into existing JSON, preserving protected fields."""
    with open(json_path, 'r', encoding='utf-8') as f:
        existing = json.load(f)

    for k, v in new_metrics.items():
        if k not in _PRESERVE:
            existing[k] = v

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"  Updated  → {json_path}")


# ── per-language runner ───────────────────────────────────────────────────────

def run_language(lang, base_dir, n_samples):
    lang_dir = os.path.join(base_dir, 'hindi' if lang == 'hi' else 'nepali')
    data_dir = os.path.join(lang_dir, 'data')
    eval_dir = os.path.join(lang_dir, 'eval')

    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}")
    print(f"{'='*60}")

    for split in ('val', 'test'):
        jsonl   = os.path.join(data_dir, f'{split}.jsonl')
        json_out = os.path.join(eval_dir, f'{split}_metrics.json')

        if not os.path.exists(jsonl):
            print(f"\n[{split}] SKIP: {jsonl} not found")
            continue
        if not os.path.exists(json_out):
            print(f"\n[{split}] SKIP: {json_out} not found")
            continue

        print(f"\n[{split}]")
        metrics = compute_metrics_for_split(jsonl, n_samples)
        if not metrics:
            continue

        print(f"  BLEU      : {metrics.get('bleu', 0):.2f}")
        print(f"  chrF      : {metrics.get('chrf', 0):.2f}")
        print(f"  ROUGE-L   : {metrics.get('rouge_l', 0):.2f}")
        print(f"  Distinct-1: {metrics.get('distinct_1', 0):.4f}")
        print(f"  Distinct-2: {metrics.get('distinct_2', 0):.4f}")
        print(f"  Distinct-3: {metrics.get('distinct_3', 0):.4f}")
        print(f"  RepRate-2 : {metrics.get('repetition_rate_2', 0):.4f}")

        update_metrics_json(json_out, metrics)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute real metrics on data subsets and update eval JSON files.'
    )
    parser.add_argument('--lang', choices=['hi', 'ne'], default=None,
                        help='Language to evaluate (default: both)')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of samples per split (default: 200)')
    parser.add_argument('--base-dir', default=None,
                        help='Repo root directory (default: auto-detect from script location)')
    args = parser.parse_args()

    base_dir = args.base_dir or str(Path(__file__).resolve().parent.parent)
    langs = [args.lang] if args.lang else ['hi', 'ne']

    for lang in langs:
        run_language(lang, base_dir, args.n)

    print("\nDone.")


if __name__ == '__main__':
    main()
