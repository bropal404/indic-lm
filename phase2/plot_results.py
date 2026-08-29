#!/usr/bin/env python
"""
Plotting script for Phase 2 results.
"""
import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def parse_training_log(log_path):
    """Parse training log to extract loss and learning rate data."""
    steps = []
    losses = []
    lrs = []
    val_steps = []
    val_losses = []
    val_ppls = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for validation loss lines: "Step 1000 | Val Loss: 7.5842 | Val PPL: 1966.88"
            if 'Val Loss:' in line:
                parts = line.strip().split('|')
                step_part = parts[0].strip()
                step = int(step_part.split()[1])
                val_loss = float(parts[1].split(':')[1].strip())
                val_ppl = float(parts[2].split(':')[1].strip())
                val_steps.append(step)
                val_losses.append(val_loss)
                val_ppls.append(val_ppl)
            # Try to parse training lines with loss (if any exist in format "Step X | Loss: Y | LR: Z")
            elif 'Step' in line and 'Loss:' in line and 'LR:' in line:
                parts = line.strip().split('|')
                step_part = parts[0].strip()
                step = int(step_part.split()[1])
                loss = float(parts[1].split(':')[1].strip())
                lr = float(parts[2].split(':')[1].strip())
                steps.append(step)
                losses.append(loss)
                lrs.append(lr)
    
    return steps, losses, lrs, val_steps, val_losses, val_ppls


def plot_training_curves(log_path, save_dir):
    """Plot training and validation curves."""
    steps, losses, lrs, val_steps, val_losses, val_ppls = parse_training_log(log_path)
    
    if not steps and not val_steps:
        print(f"No training data found in {log_path}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    if steps:
        axes[0, 0].plot(steps, losses, 'b-', alpha=0.7, label='Train Loss')
    if val_steps:
        axes[0, 0].plot(val_steps, val_losses, 'r-', alpha=0.7, label='Val Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if steps and lrs:
        axes[0, 1].plot(steps, lrs, 'g-', alpha=0.7)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].set_title('Learning Rate Schedule (not logged)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Validation perplexity
    if val_steps:
        axes[1, 0].plot(val_steps, val_ppls, 'm-', alpha=0.7)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].set_title('Validation Perplexity')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].set_title('Validation Perplexity (not available)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss comparison (smoothed)
    if len(losses) > 10:
        window = 50
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        axes[1, 1].plot(steps, losses, 'b-', alpha=0.3, label='Raw')
        axes[1, 1].plot(smooth_steps, smoothed, 'b-', alpha=0.9, label=f'Smoothed (window={window})')
        if val_steps:
            axes[1, 1].plot(val_steps, val_losses, 'r-', alpha=0.7, label='Val Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss (Smoothed)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_dir / 'training_curves.png'}")


def plot_comparison_table(results_h, results_l, save_dir):
    """Create comparison table visualization."""
    metrics = [
        ('Parameters', 'parameters', ''),
        ('Validation CE', 'cross_entropy', ''),
        ('Perplexity', 'perplexity', ''),
        ('Bits per Byte', 'bits_per_byte', ''),
        ('BLEU-4', 'bleu', ''),
        ('chrF', 'chrf', ''),
        ('ROUGE-L', 'rouge_l', ''),
        ('Repetition Rate (2)', 'diversity.repetition_rate_2', ''),
        ('Distinct-1', 'diversity.distinct_1', ''),
        ('Distinct-2', 'diversity.distinct_2', ''),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table_data = [['Metric', 'Model H (Hindi)', 'Model L (Nepali)']]
    for name, key, _ in metrics:
        try:
            keys = key.split('.')
            val_h = results_h
            val_l = results_l
            for k in keys:
                val_h = val_h.get(k, 'N/A')
                val_l = val_l.get(k, 'N/A')
            
            if isinstance(val_h, float):
                val_h = f"{val_h:.4f}"
            if isinstance(val_l, float):
                val_l = f"{val_l:.4f}"
            
            table_data.append([name, str(val_h), str(val_l)])
        except:
            table_data.append([name, 'N/A', 'N/A'])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Model H vs Model L Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison table to {save_dir / 'comparison_table.png'}")


def plot_generation_samples(results, save_dir, model_name):
    """Plot generation samples."""
    generations = results.get('generations', [])
    
    if not generations:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    text = f"{model_name} Generation Samples\n\n"
    for gen in generations[:15]:
        temp = gen.get('temperature', 1.0)
        prompt = gen.get('prompt', '')[:50]
        generation = gen.get('generation', '')[:100]
        text += f"Temp {temp}: {prompt}... → {generation}...\n\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(save_dir / f'generations_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved generations to {save_dir / f'generations_{model_name.lower()}.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot Phase 2 results')
    parser.add_argument('--log-h', type=str, default='phase2/logs/training_log.txt', help='Hindi training log')
    parser.add_argument('--log-l', type=str, default='phase2/logs/training_log.txt', help='Nepali training log')
    parser.add_argument('--results-h', type=str, default='phase2/results/model_h/evaluation_results.json', help='Hindi results')
    parser.add_argument('--results-l', type=str, default='phase2/results/model_l/evaluation_results.json', help='Nepali results')
    parser.add_argument('--output-dir', type=str, default='phase2/results/plots', help='Output directory')
    args = parser.parse_args()
    
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Plotting training curves for Model H...")
    plot_training_curves(args.log_h, save_dir)
    
    if Path(args.results_h).exists() and Path(args.results_l).exists():
        with open(args.results_h) as f:
            results_h = json.load(f)
        with open(args.results_l) as f:
            results_l = json.load(f)
        
        print("Plotting comparison table...")
        plot_comparison_table(results_h, results_l, save_dir)
        
        print("Plotting generation samples...")
        plot_generation_samples(results_h, save_dir, 'Model_H')
        plot_generation_samples(results_l, save_dir, 'Model_L')
    
    print("Plotting complete!")


if __name__ == "__main__":
    main()