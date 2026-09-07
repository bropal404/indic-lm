#!/usr/bin/env python3
"""
compute_perplexity.py

Loads model_final_hindi.pt / model_final_nepali.pt (trained on the 16k JSON
tokenizers) and computes REAL cross-entropy loss, perplexity, and bits-per-byte
on val/test splits. Updates {hindi,nepali}/eval/{val,test}_metrics.json.

Usage:
    python3 scripts/compute_perplexity.py            # both languages
    python3 scripts/compute_perplexity.py --lang hi
    python3 scripts/compute_perplexity.py --lang ne
    python3 scripts/compute_perplexity.py --max-batches 30
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

# Root model.py — GPTLanguageModel (token_emb, pos_emb, W_q, ln_f, head)
from model import GPTLanguageModel


# ── tokenizer ─────────────────────────────────────────────────────────────────

def load_tokenizer_json(json_path):
    """Load byte-level BPE tokenizer from *_tokenizer_16000.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = data.get('vocab', {})          # chr(byte) → id
    raw   = data.get('merges', [])
    if isinstance(raw, list):
        merges      = {tuple(p): p[0] + p[1] for p in raw}
        merge_ranks = {tuple(p): i           for i, p in enumerate(raw)}
    else:
        merges      = raw
        merge_ranks = {(tuple(k) if not isinstance(k, tuple) else k): i
                       for i, k in enumerate(raw.keys())}
    return vocab, merges, merge_ranks


def tokenize(text, vocab, merges, merge_ranks, unk_id=0):
    """Byte-level BPE: encode UTF-8 bytes as chr(b), apply merges → token ids."""
    token_ids = []
    for word in text.split():
        byte_chars = [chr(b) for b in word.encode('utf-8')] + ['</w>']
        while len(byte_chars) > 1:
            pairs = [(byte_chars[i], byte_chars[i+1])
                     for i in range(len(byte_chars) - 1)]
            best = min(pairs, key=lambda p: merge_ranks.get(p, float('inf')))
            if best not in merge_ranks:
                break
            l, r = best
            merged, i, out = l + r, 0, []
            while i < len(byte_chars):
                if i < len(byte_chars)-1 and byte_chars[i] == l and byte_chars[i+1] == r:
                    out.append(merged); i += 2
                else:
                    out.append(byte_chars[i]); i += 1
            byte_chars = out
        for t in byte_chars:
            if t != '</w>':
                token_ids.append(vocab.get(t, unk_id))
    return token_ids


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path):
    """Load GPTLanguageModel from a flat-config checkpoint (model_final_*.pt)."""
    ck  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ck['config']          # flat dict, not nested under 'model'
    model = GPTLanguageModel(
        vocab_size  = cfg['vocab_size'],
        d_model     = cfg['d_model'],
        n_heads     = cfg['n_heads'],
        n_layers    = cfg['n_layers'],
        d_ff        = cfg['d_ff'],
        context_len = cfg['context_len'],
        dropout     = 0.0,
        pad_token_id= cfg.get('pad_token_id', 0),
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, cfg


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_loss_and_ppl(model, jsonl_path, vocab, merges, merge_ranks,
                          context_len=512, max_batches=50, batch_size=8,
                          min_chars=50, min_dev_ratio=0.3):
    """
    Stream JSONL, tokenize with byte-level BPE, run model forward on
    context_len-length windows, accumulate NLL.
    Returns (avg_loss, perplexity, bits_per_byte).
    """
    device = next(model.parameters()).device
    total_nll, total_tokens, total_chars, batches_done = 0.0, 0, 0, 0
    batch_in, batch_tgt = [], []

    def flush(bi, bt):
        nonlocal total_nll, total_tokens, batches_done
        x = torch.tensor(bi, dtype=torch.long, device=device)
        y = torch.tensor(bt, dtype=torch.long, device=device)
        out    = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        nll    = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum'
        )
        total_nll    += nll.item()
        total_tokens += y.numel()
        batches_done += 1

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if batches_done >= max_batches:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            text      = row.get('text', '').strip()
            dev_ratio = row.get('devanagari_ratio', 0.0)
            if len(text) < min_chars or dev_ratio < min_dev_ratio:
                continue

            ids = tokenize(text, vocab, merges, merge_ranks)
            if len(ids) < context_len + 1:
                continue
            total_chars += len(text.encode('utf-8'))

            for start in range(0, len(ids) - context_len, context_len):
                batch_in.append(ids[start: start + context_len])
                batch_tgt.append(ids[start + 1: start + context_len + 1])
                if len(batch_in) >= batch_size:
                    flush(batch_in, batch_tgt)
                    batch_in, batch_tgt = [], []
                    if batches_done >= max_batches:
                        break

    if batch_in and batches_done < max_batches:
        flush(batch_in, batch_tgt)

    if total_tokens == 0:
        return float('inf'), float('inf'), float('inf')

    avg_loss = total_nll / total_tokens
    ppl      = math.exp(min(avg_loss, 20))
    bpb      = total_nll / (max(total_chars, 1) * math.log(2))
    return round(avg_loss, 4), round(ppl, 4), round(bpb, 4)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def update_json(path, updates):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.update(updates)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {path}")


# ── config ────────────────────────────────────────────────────────────────────

LANG_CONFIG = {
    'hi': {
        'checkpoint': 'model_final_hindi.pt',
        'tokenizer':  'hindi/tokenizer/hindi_tokenizer_16000.json',
        'data_dir':   'hindi/data',
        'eval_dir':   'hindi/eval',
        'splits': {
            'val':  ('val.jsonl',  'val_metrics.json'),
            'test': ('test.jsonl', 'test_metrics.json'),
        },
    },
    'ne': {
        'checkpoint': 'model_final_nepali.pt',
        'tokenizer':  'nepali/tokenizer/nepali_tokenizer_16000.json',
        'data_dir':   'nepali/data',
        'eval_dir':   'nepali/eval',
        'splits': {
            'val':  ('val.jsonl',  'val_metrics.json'),
            'test': ('test.jsonl', 'test_metrics.json'),
        },
    },
}


# ── runner ────────────────────────────────────────────────────────────────────

def run_language(lang, base_dir, max_batches):
    cfg  = LANG_CONFIG[lang]
    base = Path(base_dir)

    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}")
    print(f"{'='*60}")

    ckpt_path = base / cfg['checkpoint']
    if not ckpt_path.exists():
        print(f"  SKIP: {ckpt_path} not found")
        return

    print(f"  Loading: {ckpt_path.name}")
    model, model_cfg = load_model(str(ckpt_path))
    context_len = model_cfg['context_len']
    print(f"  vocab={model_cfg['vocab_size']}  d={model_cfg['d_model']}  "
          f"L={model_cfg['n_layers']}  H={model_cfg['n_heads']}  ctx={context_len}  "
          f"step={torch.load(str(ckpt_path), map_location='cpu', weights_only=False).get('step','?')}")

    tok_path = base / cfg['tokenizer']
    if not tok_path.exists():
        print(f"  SKIP: tokenizer not found at {tok_path}")
        return
    vocab, merges, merge_ranks = load_tokenizer_json(str(tok_path))
    print(f"  Tokenizer: {tok_path.name}  vocab_size={len(vocab)}")

    for split, (jsonl_name, json_name) in cfg['splits'].items():
        jsonl_path = base / cfg['data_dir'] / jsonl_name
        json_path  = base / cfg['eval_dir'] / json_name
        if not jsonl_path.exists():
            print(f"\n  [{split}] SKIP: {jsonl_path} not found")
            continue
        if not json_path.exists():
            print(f"\n  [{split}] SKIP: {json_path} not found")
            continue

        print(f"\n  [{split}] computing on {jsonl_name} ...")
        loss, ppl, bpb = compute_loss_and_ppl(
            model, str(jsonl_path), vocab, merges, merge_ranks,
            context_len=context_len, max_batches=max_batches,
        )
        print(f"  loss={loss:.4f}  perplexity={ppl:.4f}  bpb={bpb:.4f}")

        update_json(str(json_path), {
            'loss':                loss,
            f'{split}_loss':       loss,
            'cross_entropy':       loss,
            'cross_entropy_loss':  loss,
            'perplexity':          ppl,
            f'{split}_perplexity': ppl,
            'bits_per_byte':       bpb,
            'vocab_size':          len(vocab),
        })


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute real loss/perplexity from final checkpoints.'
    )
    parser.add_argument('--lang', choices=['hi', 'ne'], default=None)
    parser.add_argument('--max-batches', type=int, default=50,
                        help='Batches per split (default: 50)')
    parser.add_argument('--base-dir', default=None)
    args = parser.parse_args()

    base_dir = args.base_dir or str(Path(__file__).resolve().parent.parent)
    langs    = [args.lang] if args.lang else ['hi', 'ne']

    for lang in langs:
        run_language(lang, base_dir, args.max_batches)

    print("\nDone.")


if __name__ == '__main__':
    main()
