#!/usr/bin/env python3
"""
reporting.py  --  generate all figures and tables for the Phase 2 report.

Outputs:
  report/figures/*.png          -- charts
  report/figures/tables/*.jsonl -- tables

Usage:
    python3 scripts/reporting.py
    python3 scripts/reporting.py --lang hi
    python3 scripts/reporting.py --no-temp-sweep   # skip slow temperature sweep
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GPTLanguageModel


SEED          = 42
GEN_SENTENCES = 30
TEMPERATURES  = [0.5, 0.7, 1.0, 1.2, 1.5]
FIGURES_DIR   = Path('report/figures')
TABLES_DIR    = FIGURES_DIR / 'tables'

COLOR_H = '#2563EB'   # blue  -- Hindi
COLOR_L = '#D97706'   # amber -- Nepali

ATTN_SENTENCE_HI = "भारत एक विशाल और विविधतापूर्ण देश है जहाँ अनेक संस्कृतियाँ एक साथ फलती हैं"
ATTN_SENTENCE_NE = "नेपाल एक सुन्दर हिमाली देश हो जहाँ प्राकृतिक सम्पदाको धनी पहिचान छ"

LANG_CONFIG = {
    'hi': {
        'checkpoint': 'model_final_hindi.pt',
        'tokenizer':  'hindi/tokenizer/hindi_tokenizer_16000.json',
        'data_dir':   'hindi/data',
        'eval_dir':   'hindi/eval',
        'label': 'Model H (Hindi)',
        'color': COLOR_H,
        'attn_sentence': ATTN_SENTENCE_HI,
    },
    'ne': {
        'checkpoint': 'model_final_nepali.pt',
        'tokenizer':  'nepali/tokenizer/nepali_tokenizer_16000.json',
        'data_dir':   'nepali/data',
        'eval_dir':   'nepali/eval',
        'label': 'Model L (Nepali)',
        'color': COLOR_L,
        'attn_sentence': ATTN_SENTENCE_NE,
    },
}


def set_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   '#f8f9fa',
        'axes.grid':        True,
        'grid.alpha':       0.4,
        'grid.linestyle':   '--',
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'font.size':        11,
        'axes.titlesize':   13,
        'axes.labelsize':   11,
        'legend.fontsize':  10,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'figure.dpi':       150,
    })


def load_tokenizer(path):
    data = json.loads(Path(path).read_text())
    vocab = data.get('vocab', {})
    raw   = data.get('merges', [])
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
    buf = bytearray()
    for id_ in ids:
        for ch in inv_vocab.get(id_, ''):
            b = ord(ch)
            if b < 256:
                buf.append(b)
    return buf.decode('utf-8', errors='replace')


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c  = ck['config']
    m  = GPTLanguageModel(
        vocab_size=c['vocab_size'], d_model=c['d_model'], n_heads=c['n_heads'],
        n_layers=c['n_layers'], d_ff=c['d_ff'], context_len=c['context_len'],
        dropout=0.0, pad_token_id=c.get('pad_token_id', 0),
    )
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, c, ck.get('step', '?')


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

def compute_text_metrics(hyps, refs):
    hc = [' '.join(list(h)) for h in hyps]
    rc = [' '.join(list(r)) for r in refs]
    bleu   = round(corpus_bleu(hc, [rc], smooth_method='exp', smooth_value=0.0001).score, 4)
    chrf   = round(CHRF().corpus_score(hyps, [refs]).score, 4)
    rl = []
    for h, r in zip(hyps, refs):
        lcs = _lcs(h, r)
        p  = lcs/len(h) if h else 0.0
        rc_ = lcs/len(r) if r else 0.0
        rl.append(2*p*rc_/(p+rc_) if p+rc_ > 0 else 0.0)
    rouge = round(float(np.mean(rl))*100, 4)
    chars_h = [list(h) for h in hyps]
    d1_ng  = [g for h in chars_h for g in _ngrams(h, 1)]
    d2_ng  = [g for h in chars_h for g in _ngrams(h, 2)]
    r2_ng  = [g for h in chars_h for g in _ngrams(h, 2)]
    dist1  = round(len(set(d1_ng))/len(d1_ng), 4) if d1_ng else 0.0
    dist2  = round(len(set(d2_ng))/len(d2_ng), 4) if d2_ng else 0.0
    r2c    = Counter(r2_ng)
    rep2   = round(sum(1 for c in r2c.values() if c > 1)/len(r2c), 4) if r2c else 0.0
    return {'bleu': bleu, 'chrf': chrf, 'rouge_l': rouge,
            'distinct_1': dist1, 'distinct_2': dist2, 'repetition_rate_2': rep2}


@torch.no_grad()
def generate_metrics(model, sentences, vocab, merges, ranks, inv_vocab,
                     temperature=1.0, max_new_tokens=60, top_k=40):
    hyps, refs = [], []
    for sentence in sentences:
        words = sentence.split()
        mid   = max(1, len(words)//2)
        prompt_ids = tokenize(' '.join(words[:mid]), vocab, merges, ranks)
        if not prompt_ids:
            continue
        x   = torch.tensor([prompt_ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k)
        gen = decode(out[0, len(prompt_ids):].tolist(), inv_vocab)
        hyps.append(gen)
        refs.append(' '.join(words[mid:]))
    return compute_text_metrics(hyps, refs) if hyps else {}


def load_sentences(jsonl_path, n, min_chars=80, min_ratio=0.55):
    out = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            try: row = json.loads(line.strip())
            except: continue
            t = row.get('text','').strip()
            if len(t) >= min_chars and row.get('devanagari_ratio',0) >= min_ratio:
                out.append(t)
            if len(out) >= n*15: break
    random.Random(SEED).shuffle(out)
    return out[:n]


@torch.no_grad()
def get_attention_maps(model, text, vocab, merges, ranks):
    ids = tokenize(text, vocab, merges, ranks)
    ids = ids[:model.context_len]
    x   = torch.tensor([ids], dtype=torch.long)
    _, _, attn_maps = model(x, return_attention=True)
    # attn_maps: list[n_layers] of [1, n_heads, T, T]
    return [a[0].numpy() for a in attn_maps], ids   # list of [n_heads, T, T]

def attn_entropy(attn):
    """attn: [n_heads, T, T] -> mean entropy per head [n_heads]"""
    eps = 1e-9
    h   = -(attn * np.log(attn + eps)).sum(axis=-1)  # [n_heads, T]
    return h.mean(axis=-1)                             # [n_heads]

def attn_mean_distance(attn):
    """attn: [n_heads, T, T] -> mean attention distance per head [n_heads]"""
    T = attn.shape[-1]
    pos = np.arange(T)
    dists = []
    for h in range(attn.shape[0]):
        d = []
        for t in range(T):
            weights = attn[h, t, :t+1]
            if weights.sum() > 0:
                d.append(np.sum(weights * (t - pos[:t+1])) / weights.sum())
        dists.append(np.mean(d) if d else 0.0)
    return np.array(dists)


def plot_metric_vs_temp(results, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lang, data in results.items():
        cfg = LANG_CONFIG[lang]
        temps  = [d['temperature'] for d in data]
        values = [d.get(metric, 0) for d in data]
        ax.plot(temps, values, marker='o', linewidth=2.2, markersize=6,
                color=cfg['color'], label=cfg['label'])
        for x, y in zip(temps, values):
            ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=8, color=cfg['color'])
    ax.set_xlabel('Temperature')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Temperature — Model H vs L')
    ax.set_xticks(TEMPERATURES)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f'  saved {filename}')


def plot_attention_heatmaps(attn_maps, layer_idx, ids, inv_vocab, title, filename, n_heads_show=8):
    attn = attn_maps[layer_idx]  # [n_heads, T, T]
    T    = attn.shape[-1]
    n    = min(n_heads_show, attn.shape[0])
    cols = 4
    rows = math.ceil(n / cols)

    # decode token labels (show first 2 bytes of each token for readability)
    def tok_label(id_):
        raw = inv_vocab.get(id_, '?')
        b   = bytearray(ord(c) for c in raw if ord(c) < 256)
        try:
            s = b.decode('utf-8', errors='replace')
            return s[:4] if s.strip() else f'·{id_}'
        except:
            return f'·{id_}'

    labels = [tok_label(i) for i in ids[:T]]
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.0))
    axes = np.array(axes).reshape(-1)
    cmap = plt.cm.Blues

    for h in range(n):
        ax  = axes[h]
        mat = attn[h]
        im  = ax.imshow(mat, cmap=cmap, vmin=0, aspect='auto')
        ax.set_title(f'Head {h}', fontsize=9, pad=3)
        if T <= 24:
            ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(n, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {filename}')


def plot_attn_stat_heatmap(stat_grid, title, filename, cmap='viridis'):
    # stat_grid: [n_layers, n_heads]
    n_layers, n_heads = stat_grid.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads*0.7), max(3, n_layers*0.55)))
    im = ax.imshow(stat_grid, cmap=cmap, aspect='auto')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f'H{h}' for h in range(n_heads)], fontsize=9)
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f'L{l}' for l in range(n_layers)], fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f'  saved {filename}')


def plot_resource_comparison(data_h, data_l, filename):
    metrics = ['Parameters (M)', 'Val PPL', 'Val BPB', 'Training Steps (k)']
    vals_h  = [data_h['params_m'], data_h['val_ppl'], data_h['val_bpb'], data_h['steps_k']]
    vals_l  = [data_l['params_m'], data_l['val_ppl'], data_l['val_bpb'], data_l['steps_k']]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, vals_h, w, label='Model H (Hindi)',  color=COLOR_H, alpha=0.85)
    b2 = ax.bar(x + w/2, vals_l, w, label='Model L (Nepali)', color=COLOR_L, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title('Resource-level Comparison: Model H vs Model L', fontweight='bold')
    ax.legend()
    ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f'  saved {filename}')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'  saved {path.name}')


def run(langs, base, do_temp_sweep):
    set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    models, configs, steps_map = {}, {}, {}
    vocabs, merges_map, ranks_map, inv_vocabs = {}, {}, {}, {}

    for lang in langs:
        cfg  = LANG_CONFIG[lang]
        ckpt = base / cfg['checkpoint']
        if not ckpt.exists():
            print(f'[{lang}] checkpoint not found, skipping')
            continue
        print(f'\n[{lang}] loading model ...')
        m, c, step = load_model(str(ckpt))
        models[lang], configs[lang], steps_map[lang] = m, c, step
        v, me, ra, iv = load_tokenizer(base / cfg['tokenizer'])
        vocabs[lang], merges_map[lang], ranks_map[lang], inv_vocabs[lang] = v, me, ra, iv
        print(f'  vocab={c["vocab_size"]}  d={c["d_model"]}  L={c["n_layers"]}  '
              f'H={c["n_heads"]}  ctx={c["context_len"]}  step={step}')

    if not models:
        print('No models loaded. Exiting.')
        return

    existing_metrics = {}
    for lang in models:
        cfg = LANG_CONFIG[lang]
        existing_metrics[lang] = {}
        for split in ('val', 'test'):
            p = base / cfg['eval_dir'] / f'{split}_metrics.json'
            if p.exists():
                existing_metrics[lang][split] = json.loads(p.read_text())

    temp_results = {lang: [] for lang in models}

    if do_temp_sweep:
        print('\n-- temperature sweep --')
        sentences = {}
        for lang in models:
            cfg  = LANG_CONFIG[lang]
            jl   = base / cfg['data_dir'] / 'val.jsonl'
            sentences[lang] = load_sentences(jl, GEN_SENTENCES) if jl.exists() else []
            print(f'  [{lang}] loaded {len(sentences[lang])} sentences')

        for temp in TEMPERATURES:
            for lang in models:
                print(f'  [{lang}] temp={temp} ...', end=' ', flush=True)
                m = models[lang]
                v, me, ra, iv = vocabs[lang], merges_map[lang], ranks_map[lang], inv_vocabs[lang]
                res = generate_metrics(m, sentences[lang], v, me, ra, iv, temperature=temp)
                res['temperature'] = temp
                temp_results[lang].append(res)
                print(f"BLEU={res.get('bleu',0):.2f} chrF={res.get('chrf',0):.2f}")

        print('\n-- plotting temperature charts --')
        plot_metric_vs_temp(temp_results, 'bleu',              'BLEU-4',            'bleu_vs_temp.png')
        plot_metric_vs_temp(temp_results, 'chrf',              'chrF',              'chrf_vs_temp.png')
        plot_metric_vs_temp(temp_results, 'rouge_l',           'ROUGE-L',           'rouge_l_vs_temp.png')
        plot_metric_vs_temp(temp_results, 'repetition_rate_2', 'Repetition Rate-2', 'reprate_vs_temp.png')
        plot_metric_vs_temp(temp_results, 'distinct_1',        'Distinct-1',        'distinct1_vs_temp.png')
        plot_metric_vs_temp(temp_results, 'distinct_2',        'Distinct-2',        'distinct2_vs_temp.png')

    print('\n-- attention analysis --')
    for lang in models:
        cfg   = LANG_CONFIG[lang]
        model = models[lang]
        v, me, ra, iv = vocabs[lang], merges_map[lang], ranks_map[lang], inv_vocabs[lang]
        n_layers = configs[lang]['n_layers']
        sentence = cfg['attn_sentence']

        print(f'  [{lang}] getting attention maps ...')
        attn_maps, ids = get_attention_maps(model, sentence, v, me, ra)
        T = len(ids)

        tag = lang.upper()
        plot_attention_heatmaps(attn_maps, 0,           ids, iv,
                                f'Model {tag} — Layer 0 (early), {T} tokens',
                                f'attn_{lang}_early.png')
        plot_attention_heatmaps(attn_maps, n_layers-1,  ids, iv,
                                f'Model {tag} — Layer {n_layers-1} (late), {T} tokens',
                                f'attn_{lang}_late.png')

        # entropy and distance grids
        ent_grid  = np.stack([attn_entropy(a)       for a in attn_maps])  # [L, H]
        dist_grid = np.stack([attn_mean_distance(a) for a in attn_maps])  # [L, H]

        plot_attn_stat_heatmap(ent_grid,  f'Attention Entropy by Layer/Head — Model {tag}',
                               f'attn_entropy_{lang}.png', cmap='YlOrRd')
        plot_attn_stat_heatmap(dist_grid, f'Mean Attention Distance by Layer/Head — Model {tag}',
                               f'attn_distance_{lang}.png', cmap='viridis')

    if len(models) == 2:
        print('\n-- resource comparison chart --')
        def resource_data(lang):
            c    = configs[lang]
            m    = models[lang]
            ppl  = existing_metrics.get(lang,{}).get('val',{}).get('perplexity', 0)
            bpb  = existing_metrics.get(lang,{}).get('val',{}).get('bits_per_byte', 0)
            return {
                'params_m': round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                'val_ppl':  round(ppl, 1),
                'val_bpb':  round(bpb, 3),
                'steps_k':  round(steps_map[lang] / 1000, 1),
            }
        plot_resource_comparison(resource_data('hi'), resource_data('ne'),
                                 'resource_comparison.png')

    print('\n-- writing tables --')

    # model config table
    config_rows = []
    for lang in models:
        c = configs[lang]
        m = models[lang]
        total_params = sum(p.numel() for p in m.parameters())
        config_rows.append({
            'model': LANG_CONFIG[lang]['label'],
            'language': lang,
            'vocab_size': c['vocab_size'],
            'd_model': c['d_model'],
            'n_layers': c['n_layers'],
            'n_heads': c['n_heads'],
            'd_ff': c['d_ff'],
            'context_len': c['context_len'],
            'total_params': total_params,
            'total_params_M': round(total_params/1e6, 2),
            'checkpoint_step': steps_map[lang],
        })
    write_jsonl(TABLES_DIR / 'model_config.jsonl', config_rows)

    # val/test PPL/BPB table
    ppl_rows = []
    for lang in models:
        for split in ('val', 'test'):
            em = existing_metrics.get(lang, {}).get(split, {})
            ppl_rows.append({
                'model': LANG_CONFIG[lang]['label'],
                'language': lang,
                'split': split,
                'cross_entropy_loss': em.get('loss', em.get('cross_entropy_loss')),
                'perplexity': em.get('perplexity'),
                'bits_per_byte': em.get('bits_per_byte'),
            })
    write_jsonl(TABLES_DIR / 'ppl_results.jsonl', ppl_rows)

    # temp sweep tables
    if do_temp_sweep:
        bleu_rows, div_rows = [], []
        for lang in models:
            for d in temp_results[lang]:
                bleu_rows.append({
                    'model': LANG_CONFIG[lang]['label'], 'language': lang,
                    'temperature': d['temperature'],
                    'bleu': d.get('bleu'), 'chrf': d.get('chrf'), 'rouge_l': d.get('rouge_l'),
                })
                div_rows.append({
                    'model': LANG_CONFIG[lang]['label'], 'language': lang,
                    'temperature': d['temperature'],
                    'distinct_1': d.get('distinct_1'), 'distinct_2': d.get('distinct_2'),
                    'repetition_rate_2': d.get('repetition_rate_2'),
                })
        write_jsonl(TABLES_DIR / 'bleu_vs_temp.jsonl',      bleu_rows)
        write_jsonl(TABLES_DIR / 'diversity_vs_temp.jsonl',  div_rows)

    # resource comparison table
    comp_rows = []
    for lang in models:
        c   = configs[lang]
        m   = models[lang]
        em  = existing_metrics.get(lang, {})
        comp_rows.append({
            'model':         LANG_CONFIG[lang]['label'],
            'language':      lang,
            'total_params_M': round(sum(p.numel() for p in m.parameters())/1e6, 2),
            'checkpoint_step': steps_map[lang],
            'val_loss':      em.get('val',{}).get('loss'),
            'val_perplexity':em.get('val',{}).get('perplexity'),
            'val_bpb':       em.get('val',{}).get('bits_per_byte'),
            'test_loss':     em.get('test',{}).get('loss'),
            'test_perplexity':em.get('test',{}).get('perplexity'),
            'test_bpb':      em.get('test',{}).get('bits_per_byte'),
            'vocab_size':    c['vocab_size'],
            'd_model':       c['d_model'],
            'n_layers':      c['n_layers'],
            'n_heads':       c['n_heads'],
        })
    write_jsonl(TABLES_DIR / 'comparison.jsonl', comp_rows)

    print(f'\nAll outputs written to {FIGURES_DIR}/')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lang', choices=['hi','ne'], default=None)
    p.add_argument('--no-temp-sweep', action='store_true',
                   help='Skip temperature sweep (attention maps + tables only)')
    p.add_argument('--base-dir', default=None)
    args = p.parse_args()

    base  = Path(args.base_dir or Path(__file__).resolve().parent.parent)
    langs = [args.lang] if args.lang else ['hi', 'ne']
    run(langs, base, do_temp_sweep=not args.no_temp_sweep)

if __name__ == '__main__':
    main()
