#!/usr/bin/env python3
"""
Nepali Language Model Evaluation Script.

Evaluates a pretrained transformer model with the 16k Nepali tokenizer on Phase 2 metrics:
- Cross-entropy loss
- Perplexity
- BLEU-4 score
- ROUGE-L F1 score
- CHRF score
- n-gram repetition rates (2, 3, 4)
- Distinct-n (1, 2, 3)
- Diversity metrics
- Sample text generation at different temperatures

Usage:
    python3 scripts/evaluate_nepali.py --model-path <path_to_checkpoint> [--data-dir nepali/data] [--output-dir eval_results]

Tokenizer: Uses nepali/tokenizer/nepali_tokenizer_16000.json (16k merges, format: custom_unicode_bpe_v1)
Model: TransformerLM architecture (d_model=384, n_layers=6, n_heads=6, d_ff=1536, context_length=256)
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import tokenizer and evaluation functions
from tokenizer.tokenizers import BPETokenizer, customBPE
from nepali.eval.evaluation import (
    compute_cross_entropy_loss,
    compute_perplexity,
    compute_bleu,
    compute_chrf,
    compute_rouge_l,
    compute_repetition_rate,
    compute_distinct_ngrams,
    compute_diversity_metrics,
    generate_samples,
    encode_text,
    decode_ids,
)
from model import GPTLanguageModel

# Nepali tokenizer path (16k merges)
NEPALESE_TOKENIZER_PATH = "nepali/tokenizer/nepali_tokenizer_16000.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nepali_tokenizer(tokenizer_path=NEPALESE_TOKENIZER_PATH):
    """Load the 16k Nepali tokenizer from JSON file."""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    tokenizer = customBPE(num_merges=tokenizer_data.get("vocab_size", 16240))
    merges_raw = tokenizer_data["merges"]
    if isinstance(merges_raw, list):
        tokenizer.merges = {tuple(pair): pair[0] + pair[1] for pair in merges_raw}
        tokenizer.merge_ranks = {tuple(pair): idx for idx, pair in enumerate(merges_raw)}
    elif isinstance(merges_raw, dict):
        tokenizer.merges = merges_raw
        tokenizer.merge_ranks = {pair: idx for idx, pair in enumerate(merges_raw.keys())}
    tokenizer.vocab = tokenizer_data.get("vocab", {})
    tokenizer.vocab_size = len(tokenizer.vocab)
    tokenizer._token_cache = {}
    
    print(f"Loaded Nepali 16k tokenizer: vocab_size={tokenizer.vocab_size}, merges={len(tokenizer.merges)}")
    return tokenizer


class GPTModelWrapper(nn.Module):
    """Adapter for GPTLanguageModel to provide standard TransformerLM evaluation interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.context_length = getattr(model, 'context_len', 512)

    def forward(self, input_ids, return_attention=False):
        out = self.model(input_ids, return_attention=return_attention)
        if isinstance(out, tuple):
            logits = out[0]
            if return_attention:
                attn = out[2] if len(out) > 2 else None
                return logits, attn
            return logits
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        return_attention: bool = False,
    ):
        import torch.nn.functional as F

        B = input_ids.shape[0]
        generated = input_ids
        for _ in range(max_new_tokens):
            cond = generated if generated.size(1) <= self.context_length else generated[:, -self.context_length:]
            logits = self.forward(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if repetition_penalty != 1.0:
                for b in range(B):
                    for token_id in set(generated[b].tolist()):
                        logits[b, token_id] /= repetition_penalty
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


def load_model_from_checkpoint(checkpoint_path, vocab_size):
    """Load model from checkpoint file."""
    print(f"Loading model from: {checkpoint_path}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = None
        for key in ['model_state_dict', 'model_state', 'state_dict']:
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Loaded checkpoint with key: {key}")
                break
        if state_dict is None and isinstance(checkpoint, dict):
            state_dict = checkpoint

        # Check if weights correspond to GPTLanguageModel
        is_gpt_model = False
        if isinstance(state_dict, dict):
            keys = list(state_dict.keys())
            if any(k.startswith("token_emb.") or "attn.W_q" in k for k in keys):
                is_gpt_model = True

        if is_gpt_model:
            try:
                from model import GPTLanguageModel
                cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
                gpt_kwargs = {
                    'vocab_size': cfg.get('vocab_size', vocab_size),
                    'd_model': cfg.get('d_model', 512),
                    'n_heads': cfg.get('n_heads', 8),
                    'n_layers': cfg.get('n_layers', 6),
                    'd_ff': cfg.get('d_ff', 1536),
                    'context_len': cfg.get('context_len', cfg.get('context_length', 512)),
                    'dropout': cfg.get('dropout', 0.1),
                    'pad_token_id': cfg.get('pad_token_id', 0),
                }
                raw_model = GPTLanguageModel(**gpt_kwargs)
                raw_model.load_state_dict(state_dict)
                print("Loaded GPTLanguageModel architecture successfully")
                model = GPTModelWrapper(raw_model)
                model.eval()
                return model
            except Exception as e:
                print(f"Warning: Failed to load as GPTLanguageModel ({e}), trying TransformerLM")

        cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        model = GPTLanguageModel(
            vocab_size=cfg.get('vocab_size', vocab_size),
            d_model=cfg.get('d_model', 384),
            n_heads=cfg.get('n_heads', 6),
            n_layers=cfg.get('n_layers', 6),
            d_ff=cfg.get('d_ff', 1536),
            context_len=cfg.get('context_length', cfg.get('context_len', 256)),
            dropout=cfg.get('dropout', 0.1),
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
            print("Loaded GPTLanguageModel state dict")
    else:
        print("No checkpoint found, using untrained model")
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            d_model=384,
            n_heads=6,
            n_layers=6,
            d_ff=1536,
            context_len=256,
            dropout=0.1,
        )
    
    model.eval()
    return model


def load_data_splits(data_dir, tokenizer, max_samples=1000):
    """Load train/val/test data splits efficiently."""
    splits = {}
    for split in ["train", "val", "test"]:
        jsonl_path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(jsonl_path):
            token_ids = []
            count = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        text = row.get("text", "")
                    except Exception:
                        continue
                    if text and len(str(text)) >= 50:
                        tokens = tokenizer.tokenize(str(text))
                        if len(tokens) >= 50:
                            token_ids.extend([tokenizer.vocab.get(t, 0) for t in tokens])
                            count += 1
                            if max_samples and count >= max_samples:
                                break
            
            splits[split] = token_ids
            print(f"Loaded {split}: {len(splits[split])} tokens from {count} samples")
        else:
            # Create fallback data
            vocab_size = tokenizer.vocab_size
            splits[split] = [random.randint(0, vocab_size - 1) for _ in range(200)]
            print(f"Warning: {jsonl_path} not found, using fallback data")
    
    return splits


def prepare_dataloader(token_ids, seq_len=256, batch_size=8, shuffle=True):
    """Prepare a DataLoader from token IDs."""
    if len(token_ids) < seq_len + 1:
        return None
    
    inputs = []
    targets = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        inp = token_ids[i:i + seq_len]
        tgt = token_ids[i + 1:i + seq_len + 1]
        inputs.append(inp)
        targets.append(tgt)
    
    if not inputs:
        return None
    
    input_tensors = torch.tensor(inputs, dtype=torch.long)
    target_tensors = torch.tensor(targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_tensors, target_tensors)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


def compute_all_metrics(model, dataloaders, device, tokenizer):
    """Compute all Phase 2 evaluation metrics."""
    model.eval()
    
    results = {}
    
    # Cross-entropy loss and perplexity for each split
    for split_name, loader in dataloaders.items():
        if loader is None:
            results[f"{split_name}_loss"] = float('inf')
            results[f"{split_name}_perplexity"] = float('inf')
            continue
        
        loss = compute_cross_entropy_loss(model, loader, device)
        ppl = compute_perplexity(loss)
        results[f"{split_name}_loss"] = loss
        results[f"{split_name}_perplexity"] = ppl
    
    # Sample generation and text-level metrics
    with torch.no_grad():
        prompts = ["नमस्ते", "Hello", "नमस्ते wereld"]
        sample_results = generate_samples(
            model, tokenizer, prompts,
            max_new_tokens=50,
            temperatures=[0.5, 1.0, 1.5],
            device=device,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0,
        )
        
        all_generations = []
        all_references = []
        
        items = []
        for entry in sample_results:
            if isinstance(entry, list):
                items.extend(entry)
            elif isinstance(entry, dict):
                items.append(entry)
        for item in items:
            gen_text = item.get("generation", "")
            if gen_text:
                all_generations.append(gen_text)
                all_references.append(prompts[0] if prompts else "")
    
    # Text-level metrics
    if all_generations:
        # BLEU
        bleu = compute_bleu(all_generations, all_references)
        results["bleu"] = bleu
        
        # CHRF
        chrf = compute_chrf(all_generations, all_references)
        results["chrf"] = chrf
        
        # ROUGE-L
        rouge = compute_rouge_l(all_generations, all_references)
        results["rouge_l"] = rouge
        
        # Diversity metrics
        diversity = compute_diversity_metrics(all_generations)
        results.update({f"div_{k}": v for k, v in diversity.items()})
        
        # Repetition and distinct rates
        combined_text = " ".join(all_generations)
        results["repetition_rate_2"] = compute_repetition_rate(combined_text, 2)
        results["repetition_rate_3"] = compute_repetition_rate(combined_text, 3)
        results["repetition_rate_4"] = compute_repetition_rate(combined_text, 4)
        results["distinct_1"] = compute_distinct_ngrams(combined_text, 1)
        results["distinct_2"] = compute_distinct_ngrams(combined_text, 2)
        results["distinct_3"] = compute_distinct_ngrams(combined_text, 3)
    else:
        results["bleu"] = 0.0
        results["chrf"] = 0.0
        results["rouge_l"] = 0.0
        results["repetition_rate_2"] = 0.0
        results["repetition_rate_3"] = 0.0
        results["repetition_rate_4"] = 0.0
        results["distinct_1"] = 0.0
        results["distinct_2"] = 0.0
        results["distinct_3"] = 0.0
    
    return results, sample_results


def plot_metrics(results, lang, output_dir="."):
    """Generate and save evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss and perplexity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    splits = [s for s in results.keys() if s.endswith("_loss") and results[s] != float('inf')]
    loss_vals = [results[s] for s in splits]
    if splits:
        ax1.bar(splits, loss_vals, color=['skyblue', 'orange', 'green'])
        ax1.set_xlabel('Data Split')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title(f'{lang.upper()} - Cross-Entropy Loss')
        ax1.tick_params(axis='x', rotation=45)
    
    # Perplexity plot
    ppl_splits = []
    ppl_vals = []
    for s in splits:
        ppl_key = f"{s.replace('_loss', '_perplexity')}"
        if ppl_key in results and results[ppl_key] != float('inf'):
            ppl_splits.append(s)
            ppl_vals.append(results[ppl_key])
    
    if ppl_splits:
        ax2.bar(ppl_splits, ppl_vals, color=['skyblue', 'orange', 'green'])
        ax2.set_xlabel('Data Split')
        ax2.set_ylabel('Perplexity')
        ax2.set_title(f'{lang.upper()} - Perplexity')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f"{lang}_eval_loss_ppl.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    
    # Plot n-gram and text metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Repetition rates
    axes[0, 0].bar(['Rate 2', 'Rate 3', 'Rate 4'],
                   [results.get('repetition_rate_2', 0),
                    results.get('repetition_rate_3', 0),
                    results.get('repetition_rate_4', 0)],
                   color=['red', 'orange', 'yellow'])
    axes[0, 0].set_title('Repetition Rates')
    axes[0, 0].set_ylabel('Rate')
    
    # Distinct n-grams
    axes[0, 1].bar(['Distinct 1', 'Distinct 2', 'Distinct 3'],
                   [results.get('distinct_1', 0),
                    results.get('distinct_2', 0),
                    results.get('distinct_3', 0)],
                   color=['green', 'blue', 'purple'])
    axes[0, 1].set_title('Distinct N-grams')
    axes[0, 1].set_ylabel('Ratio')
    
    # BLEU/CHRF/ROUGE
    axes[1, 0].bar(['BLEU', 'CHRF', 'ROUGE-L'],
                   [results.get('bleu', 0),
                    results.get('chrf', 0),
                    results.get('rouge_l', 0)],
                   color=['cyan', 'magenta', 'yellow'])
    axes[1, 0].set_title('Text Similarity Metrics')
    axes[1, 0].set_ylabel('Score')
    
    # Hide empty subplot
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, f"{lang}_eval_metrics.png")
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    
    return loss_plot_path, metrics_plot_path


def main():
    parser = argparse.ArgumentParser(description="Nepali LM Evaluation Script")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--data-dir", default="nepali/data", help="Path to data directory (default: nepali/data)")
    parser.add_argument("--output-dir", default="eval_results", help="Directory to save results and plots")
    parser.add_argument("--lang", choices=["hi", "ne"], default="ne", help="Language (default: ne)")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per split")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    args = parser.parse_args()
    
    lang = args.lang
    print("=" * 60)
    print(f"Evaluating {lang.upper()} Language Model")
    print("=" * 60)
    
    set_seed(42)
    
    # Load 16k Nepali tokenizer
    print(f"\nLoading 16k Nepali tokenizer from: {NEPALESE_TOKENIZER_PATH}")
    tokenizer = load_nepali_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Load model from checkpoint
    print(f"\nLoading model from: {args.model_path}")
    model = load_model_from_checkpoint(args.model_path, vocab_size)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load data splits
    print(f"\nLoading {lang} data from: {args.data_dir}")
    data_splits = load_data_splits(args.data_dir, tokenizer, max_samples=args.max_samples)
    
    # Prepare dataloaders
    dataloaders = {}
    for split_name, token_ids in data_splits.items():
        loader = prepare_dataloader(token_ids, seq_len=256, batch_size=8, shuffle=(split_name == "train"))
        dataloaders[split_name] = loader
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    results, sample_results = compute_all_metrics(model, dataloaders, device, tokenizer)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{lang}_eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate and save plots
    print("\nGenerating plots...")
    loss_plot, metrics_plot = plot_metrics(results, lang, args.output_dir)
    print(f"Loss/Perplexity plot saved to: {loss_plot}")
    print(f"Metrics plot saved to: {metrics_plot}")
    
    # Save generation samples
    samples_path = os.path.join(args.output_dir, f"{lang}_generations.json")
    with open(samples_path, 'w') as f:
        json.dump(sample_results, f, indent=2)
    print(f"Generation samples saved to: {samples_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Language: {lang.upper()}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Perplexity (val): {results.get('val_perplexity', 'N/A'):.2f}")
    print(f"BLEU: {results.get('bleu', 0):.2f}")
    print(f"CHRF: {results.get('chrf', 0):.2f}")
    print(f"ROUGE-L: {results.get('rouge_l', 0):.2f}")
    print(f"Distinct-1: {results.get('distinct_1', 0):.4f}")
    print(f"Distinct-2: {results.get('distinct_2', 0):.4f}")
    print(f"Distinct-3: {results.get('distinct_3', 0):.4f}")
    print(f"Repetition-2: {results.get('repetition_rate_2', 0):.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()