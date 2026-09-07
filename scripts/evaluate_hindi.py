#!/usr/bin/env python3
"""
Hindi Language Model Evaluation Script.

Evaluates a pretrained transformer model with the 16k Hindi tokenizer on Phase 2 metrics:
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
    python3 scripts/evaluate_hindi.py --model-path <path_to_checkpoint> [--data-dir hindi/data] [--output-dir eval_results]

Tokenizer: Uses hindi/tokenizer/hindi_tokenizer_16000.json (16k merges)
Model: TransformerLM architecture (d_model=384, n_layers=6, n_heads=6, d_ff=1536, context_length=256)
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import tokenizer and evaluation functions
from tokenizer.tokenizers import BPETokenizer, customBPE
from hindi.eval.evaluation import (
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
from hindi.train.training import TransformerLM

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hindi tokenizer path (16k merges)
HINDI_TOKENIZER_PATH = "hindi/tokenizer/hindi_tokenizer_16000.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_hindi_tokenizer(tokenizer_path=HINDI_TOKENIZER_PATH):
    """Load the 16k Hindi tokenizer from JSON file."""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    tokenizer = customBPE(num_merges=tokenizer_data.get("vocab_size", 16000))
    tokenizer.merges = tokenizer_data["merges"]
    tokenizer.vocab = tokenizer_data.get("vocab", {})
    tokenizer.merge_ranks = {pair: idx for idx, pair in enumerate(tokenizer.merges.keys())}
    tokenizer._token_cache = {}
    
    print(f"Loaded Hindi 16k tokenizer: vocab_size={tokenizer.vocab_size}, merges={len(tokenizer.merges)}")
    return tokenizer


def load_model_from_checkpoint(checkpoint_path, vocab_size):
    """Load model from checkpoint file."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model with the right architecture
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1536,
        context_length=256,
        dropout=0.1,
    )
    
    # Load checkpoint if it exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Handle different checkpoint formats
        state_key = None
        for key in ['model_state_dict', 'model_state', 'state_dict']:
            if key in checkpoint:
                state_key = key
                break
        
        if state_key:
            model.load_state_dict(checkpoint[state_key])
            print(f"Loaded checkpoint with key: {state_key}")
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint directly")
    else:
        print("No checkpoint found, using untrained model")
    
    model.eval()
    return model


def load_data_splits(data_dir, tokenizer, max_samples=1000):
    """Load train/val/test data splits."""
    from datasets import Dataset
    
    splits = {}
    for split in ["train", "val", "test"]:
        jsonl_path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(jsonl_path):
            dataset = Dataset.from_json(jsonl_path)
            texts = dataset["text"] if "text" in dataset.column_names else []
            
            token_ids = []
            for text in texts[:max_samples]:
                if text and len(str(text)) >= 50:
                    tokens = tokenizer.tokenize(str(text))
                    if len(tokens) >= 50:
                        token_ids.extend([tokenizer.vocab.get(t, 0) for t in tokens])
            
            splits[split] = token_ids
            print(f"Loaded {split}: {len(splits[split])} tokens")
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
    
    from datasets import Dataset as DS
    dataset = DS.from_dict({
        "input_ids": inputs,
        "target_ids": targets,
    })
    
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
        prompts = ["नमस्ते", "नमस्ते दुनिया", "Hello world"]
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
        
        for sample_batch in sample_results:
            for item in sample_batch:
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
    parser = argparse.ArgumentParser(description="Hindi LM Evaluation Script")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--data-dir", default="hindi/data", help="Path to data directory (default: hindi/data)")
    parser.add_argument("--output-dir", default="eval_results", help="Directory to save results and plots")
    parser.add_argument("--lang", choices=["hi", "ne"], default="hi", help="Language (default: hi)")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per split")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    args = parser.parse_args()
    
    lang = args.lang
    print("=" * 60)
    print(f"Evaluating {lang.upper()} Language Model")
    print("=" * 60)
    
    set_seed(42)
    
    # Load 16k Hindi tokenizer
    print(f"\nLoading 16k Hindi tokenizer from: {HINDI_TOKENIZER_PATH}")
    tokenizer = load_hindi_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Load model from checkpoint
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