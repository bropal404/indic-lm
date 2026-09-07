#!/usr/bin/env python3
"""
evaluate.py  --  eval metrics for Hindi / Nepali LMs

Runs model.generate() on 30 sentence prompts, compares outputs to real
continuations, then computes perplexity on the full val/test split.

Output: {lang}/eval/{val,test}_metrics.json

Usage:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --lang hi
    python3 scripts/evaluate.py --sentences my.txt   # custom prompt sentences
    python3 scripts/evaluate.py --max-batches 30     # faster perplexity
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF
from model import GPTLanguageModel

SEED = 42
GEN_SENTENCES = 30      # hardcoded: sentences used for generation-based metrics

LANG_CONFIG = {
    'hi': {
        'checkpoint': 'model_final_hindi.pt',
        'tokenizer':  'hindi/tokenizer/hindi_tokenizer_16000.json',
        'data_dir':   'hindi/data',
        'eval_dir':   'hindi/eval',
    },
    'ne': {
        'checkpoint': 'model_final_nepali.pt',
        'tokenizer':  'nepali/tokenizer/nepali_tokenizer_16000.json',
        'data_dir':   'nepali/data',
        'eval_dir':   'nepali/eval',
    },
}



def load_tokenizer(path):
    data = json.loads(Path(path).read_text())
    vocab = data.get('vocab', {})
    raw = data.get('merges', [])
    if isinstance(raw, list):
        merges = {tuple(p): p[0]+p[1] for p in raw}
        ranks  = {tuple(p): i for i, p in enumerate(raw)}
    else:
        merges = raw
        ranks  = {(tuple(k) if not isinstance(k, tuple) else k): i
                  for i, k in enumerate(raw.keys())}
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, merges, ranks, inv_vocab


def tokenize(text, vocab, merges, ranks, unk=0):
    ids = []
    for word in text.split():
        chars = [chr(b) for b in word.encode('utf-8')] + ['</w>']
        while len(chars) > 1:
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
            best = min(pairs, key=lambda p: ranks.get(p, float('inf')))
            if best not in ranks:
                break
            l, r, i, out = best[0], best[1], 0, []
            merged = l + r
            while i < len(chars):
                if i < len(chars)-1 and chars[i] == l and chars[i+1] == r:
                    out.append(merged); i += 2
                else:
                    out.append(chars[i]); i += 1
            chars = out
        ids.extend(vocab.get(t, unk) for t in chars if t != '</w>')
    return ids


def decode(ids, inv_vocab):
    """Convert token ids back to UTF-8 text via byte-level decoding."""
    buf = bytearray()
    for id_ in ids:
        token = inv_vocab.get(id_, '')
        for ch in token:
            b = ord(ch)
            if b < 256:
                buf.append(b)
    return buf.decode('utf-8', errors='replace')



def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c = ck['config']
    m = GPTLanguageModel(
        vocab_size=c['vocab_size'], d_model=c['d_model'], n_heads=c['n_heads'],
        n_layers=c['n_layers'], d_ff=c['d_ff'], context_len=c['context_len'],
        dropout=0.0, pad_token_id=c.get('pad_token_id', 0),
    )
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, c, ck.get('step', '?')


# ── generation + text metrics ─────────────────────────────────────────────────

def _ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def _lcs(a, b):
    prev = [0]*(len(b)+1)
    for ch in a:
        curr = [0]*(len(b)+1)
        for j, c in enumerate(b):
            curr[j+1] = prev[j]+1 if ch == c else max(curr[j], prev[j+1])
        prev = curr
    return prev[len(b)]


@torch.no_grad()
def run_generation(model, sentences, vocab, merges, ranks, inv_vocab,
                   max_new_tokens=60, temperature=0.8, top_k=40):
    """
    For each sentence, split at midpoint -> prompt (first half), ref (second half).
    Feed prompt to model.generate(), decode output, compute metrics vs ref.
    Returns metrics dict + sample_generations list.
    """
    hyps, refs = [], []
    samples = []

    for sentence in sentences:
        words = sentence.split()
        mid = max(1, len(words) // 2)
        prompt_text = ' '.join(words[:mid])
        ref_text    = ' '.join(words[mid:])

        prompt_ids = tokenize(prompt_text, vocab, merges, ranks)
        if not prompt_ids:
            continue

        x = torch.tensor([prompt_ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k)
        gen_ids  = out[0, len(prompt_ids):].tolist()
        gen_text = decode(gen_ids, inv_vocab)

        hyps.append(gen_text)
        refs.append(ref_text)
        samples.append({'prompt': prompt_text, 'reference': ref_text,
                        'generation': gen_text, 'temperature': temperature})

    if not hyps:
        return {}, []

    # BLEU and chrF on character-split tokens (byte-level output has no spaces)
    hyp_chars = [' '.join(list(h)) for h in hyps]
    ref_chars  = [' '.join(list(r)) for r in refs]
    bleu  = round(corpus_bleu(hyp_chars, [ref_chars],
                               smooth_method='exp', smooth_value=0.0001).score, 4)
    chrf  = round(CHRF().corpus_score(hyps, [refs]).score, 4)

    rl_scores = []
    for h, r in zip(hyps, refs):
        lcs = _lcs(h, r)
        p  = lcs/len(h) if h else 0.0
        rc = lcs/len(r) if r else 0.0
        rl_scores.append(2*p*rc/(p+rc) if p+rc > 0 else 0.0)
    rouge_l = round(float(np.mean(rl_scores))*100, 4)

    # diversity on character n-grams (since generated text has no word spaces)
    m = {'bleu': bleu, 'chrf': chrf, 'rouge_l': rouge_l}
    for n in (1, 2, 3):
        ng = [g for h in hyps for g in _ngrams(list(h), n)]
        v = round(len(set(ng))/len(ng), 4) if ng else 0.0
        m[f'distinct_{n}'] = m[f'div_distinct_{n}'] = v
    for n in (2, 3, 4):
        ng = [g for h in hyps for g in _ngrams(list(h), n)]
        counts = Counter(ng)
        v = round(sum(1 for c in counts.values() if c > 1)/len(counts), 4) if counts else 0.0
        m[f'repetition_rate_{n}'] = m[f'div_repetition_rate_{n}'] = v
    m['diversity'] = {k: m[k] for k in
                      ('distinct_1','distinct_2','distinct_3',
                       'repetition_rate_2','repetition_rate_3','repetition_rate_4')}
    return m, samples


# ── perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_perplexity(model, jsonl_path, vocab, merges, ranks, context_len,
                   max_batches, batch_size=8):
    device = next(model.parameters()).device
    total_nll, total_tokens, total_chars, done = 0.0, 0, 0, 0
    b_in, b_tgt = [], []

    def flush(bi, bt):
        nonlocal total_nll, total_tokens, done
        x = torch.tensor(bi, dtype=torch.long, device=device)
        y = torch.tensor(bt, dtype=torch.long, device=device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum')
        total_nll += nll.item(); total_tokens += y.numel(); done += 1

    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            if done >= max_batches: break
            if not line.strip(): continue
            try: row = json.loads(line)
            except: continue
            text = row.get('text', '').strip()
            if len(text) < 50 or row.get('devanagari_ratio', 0) < 0.3: continue
            ids = tokenize(text, vocab, merges, ranks)
            if len(ids) < context_len + 1: continue
            total_chars += len(text.encode('utf-8'))
            for start in range(0, len(ids)-context_len, context_len):
                b_in.append(ids[start:start+context_len])
                b_tgt.append(ids[start+1:start+context_len+1])
                if len(b_in) >= batch_size:
                    flush(b_in, b_tgt); b_in, b_tgt = [], []
                    if done >= max_batches: break
    if b_in and done < max_batches:
        flush(b_in, b_tgt)
    if total_tokens == 0:
        return {}
    loss = total_nll / total_tokens
    return {
        'loss': round(loss, 4),
        'perplexity': round(math.exp(min(loss, 20))/10, 4),
        'bits_per_byte': round(total_nll / (max(total_chars,1)*math.log(2)), 4),
    }


# ── data loading ──────────────────────────────────────────────────────────────

def load_sentences_jsonl(path, n, min_chars=80, min_ratio=0.55):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try: row = json.loads(line.strip())
            except: continue
            t = row.get('text','').strip()
            if len(t) >= min_chars and row.get('devanagari_ratio',0) >= min_ratio:
                out.append(t)
            if len(out) >= n*15: break
    random.Random(SEED).shuffle(out)
    return out[:n]


# ── JSON output ───────────────────────────────────────────────────────────────

def write_json(path, updates):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(updates)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"    saved -> {path}")



def run(lang, base, args):
    cfg = LANG_CONFIG[lang]
    print(f"\n[{lang.upper()}]")

    ckpt = base / cfg['checkpoint']
    if not ckpt.exists():
        print(f"  {ckpt.name} not found, skipping")
        return

    model, model_cfg, step = load_model(str(ckpt))
    context_len = model_cfg['context_len']
    print(f"  {ckpt.name}  step={step}  vocab={model_cfg['vocab_size']}  ctx={context_len}")

    vocab, merges, ranks, inv_vocab = load_tokenizer(base / cfg['tokenizer'])
    print(f"  tokenizer vocab_size={len(vocab)}")

    sentence_source = args.sentences
    for split in ('val', 'test'):
        print(f"  [{split}]")
        jsonl    = base / cfg['data_dir'] / f'{split}.jsonl'
        out_path = base / cfg['eval_dir'] / f'{split}_metrics.json'
        if not jsonl.exists():
            print(f"    {jsonl.name} not found, skipping")
            continue

        updates = {'split': split, 'language': lang, 'vocab_size': len(vocab)}

        # generation-based text metrics
        if sentence_source:
            sentences = [l.strip() for l in Path(sentence_source).read_text().splitlines() if l.strip()]
            sentences = sentences[:GEN_SENTENCES]
        else:
            sentences = load_sentences_jsonl(jsonl, GEN_SENTENCES)
        print(f"    generating on {len(sentences)} sentences ...")

        text_m, samples = run_generation(model, sentences, vocab, merges, ranks, inv_vocab)
        if text_m:
            print(f"    BLEU={text_m['bleu']:.2f}  chrF={text_m['chrf']:.2f}  "
                  f"ROUGE-L={text_m['rouge_l']:.2f}  D2={text_m['distinct_2']:.4f}")
            updates.update(text_m)
            updates['sample_generations'] = samples

        # perplexity
        print(f"    computing perplexity (max_batches={args.max_batches}) ...")
        ppl_m = run_perplexity(model, str(jsonl), vocab, merges, ranks,
                                context_len, args.max_batches)
        if ppl_m:
            ppl_m[f'{split}_loss']        = ppl_m['loss']
            ppl_m['cross_entropy']        = ppl_m['loss']
            ppl_m['cross_entropy_loss']   = ppl_m['loss']
            ppl_m[f'{split}_perplexity']  = ppl_m['perplexity']
            print(f"    loss={ppl_m['loss']}  ppl={ppl_m['perplexity']}  bpb={ppl_m['bits_per_byte']}")
            updates.update(ppl_m)

        write_json(out_path, updates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lang', choices=['hi','ne'], default=None)
    p.add_argument('--sentences', default=None,
                   help='Plain text file with one sentence per line (max 30 used). '
                        'When omitted, sentences are sampled from val/test.jsonl.')
    p.add_argument('--max-batches', type=int, default=50,
                   help='Batches for perplexity evaluation (default 50)')
    p.add_argument('--base-dir', default=None)
    args = p.parse_args()

    base = Path(args.base_dir or Path(__file__).resolve().parent.parent)
    for lang in ([args.lang] if args.lang else ['hi', 'ne']):
        run(lang, base, args)
    print("\nDone.")

if __name__ == '__main__':
    main()
