#!/usr/bin/env python
"""
Evaluation script for Phase 2 models.
"""
import sys
import os
import yaml
import torch
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer_utils import load_tokenizer
from dataset import create_dataloaders
from training import create_model_from_config, Trainer
from evaluation import (
    compute_cross_entropy_loss,
    compute_perplexity,
    compute_bits_per_byte,
    generate_samples,
    compute_bleu,
    compute_chrf,
    compute_rouge_l,
    compute_diversity_metrics,
)
from attention_analysis import analyze_attention_heads, identify_head_types
from model import TransformerLM


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 2 Transformer LM')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output-dir', type=str, default='phase2/results', help='Output directory')
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading tokenizer from {config['data']['tokenizer_path']}...")
    tokenizer = load_tokenizer(config['data']['tokenizer_path'])
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    print("Creating model...")
    model = create_model_from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    
    print("Creating validation dataloader...")
    _, val_loader = create_dataloaders(
        config['data']['train_file'],
        config['data']['val_file'],
        tokenizer,
        context_length=config['model']['context_length'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        text_field=config['data']['text_field'],
        streaming=True
    )
    
    print("Computing cross-entropy loss...")
    ce_loss = compute_cross_entropy_loss(model, val_loader, device, max_batches=config['eval']['eval_steps'])
    print(f"Cross-Entropy Loss: {ce_loss:.4f}")
    
    print("Computing perplexity...")
    ppl = compute_perplexity(ce_loss)
    print(f"Perplexity: {ppl:.2f}")
    
    print("Computing bits per byte...")
    bpb = compute_bits_per_byte(model, val_loader, tokenizer, device, max_batches=config['eval']['eval_steps'])
    print(f"Bits per Byte: {bpb:.4f}")
    
    print("Generating samples...")
    test_prompts = [
        "आज मौसम बहुत",
        "मैं कल स्कूल",
        "यह किताब बहुत",
        "हम सब मिलकर",
        "सरकार ने घोषणा की",
    ]
    
    generations = generate_samples(
        model, tokenizer, test_prompts,
        max_new_tokens=config['eval']['max_new_tokens'],
        temperatures=config['eval']['generation_temperatures'],
        device=device,
        top_k=config['eval'].get('generation_top_k', 50),
        top_p=config['eval'].get('generation_top_p', 0.9),
        repetition_penalty=config['eval'].get('generation_repetition_penalty', 1.0)
    )
    
    gen_texts = [g['generation'] for g in generations]
    ref_texts = [""] * len(gen_texts)
    
    print("Computing BLEU...")
    bleu_score = compute_bleu(gen_texts, ref_texts)
    print(f"BLEU-4: {bleu_score:.2f}")
    
    print("Computing chrF...")
    chrf_score = compute_chrf(gen_texts, ref_texts)
    print(f"chrF: {chrf_score:.2f}")
    
    print("Computing ROUGE-L...")
    rouge_l = compute_rouge_l(gen_texts, ref_texts)
    print(f"ROUGE-L: {rouge_l:.2f}")
    
    print("Computing diversity metrics...")
    div_metrics = compute_diversity_metrics(gen_texts)
    for k, v in div_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("Analyzing attention...")
    test_sentences = [
        "आज मौसम बहुत अच्छा है।",
        "मैं कल स्कूल जाऊंगा।",
        "यह किताब बहुत रोचक है।",
    ]
    
    n_layers = config['model']['n_layers']
    n_heads = config['model']['n_heads']
    layer_indices = [0, n_layers // 2, n_layers - 1]
    head_indices = [0, 1, 2, n_heads // 2, n_heads - 1]
    head_indices = [h for h in head_indices if h < n_heads]
    
    attn_results = analyze_attention_heads(
        model, tokenizer, test_sentences,
        layer_indices, head_indices,
        device=str(device),
        save_dir=str(output_dir / 'attention')
    )
    
    print("Identifying head types...")
    head_types = identify_head_types(
        attn_results['avg_entropy'],
        attn_results['avg_distance']
    )
    for head_type, heads in head_types.items():
        print(f"  {head_type}: {heads}")
    
    results = {
        'cross_entropy': ce_loss,
        'perplexity': ppl,
        'bits_per_byte': bpb,
        'bleu': bleu_score,
        'chrf': chrf_score,
        'rouge_l': rouge_l,
        'diversity': div_metrics,
        'head_types': {k: v for k, v in head_types.items()},
        'generations': generations
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()