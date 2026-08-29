import sys
sys.path.insert(0, 'phase2/src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


def plot_attention_heatmap(attention: torch.Tensor, tokens: List[str], 
                           layer_idx: int, head_idx: int, 
                           save_path: str, title: str = None):
    """
    Plot attention heatmap for a specific layer and head.
    
    Args:
        attention: (T, T) attention matrix
        tokens: List of token strings
        layer_idx: Layer index
        head_idx: Head index
        save_path: Path to save the plot
        title: Optional title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    attn_np = attention.cpu().numpy() if isinstance(attention, torch.Tensor) else attention
    
    im = ax.imshow(attn_np, cmap='Blues', aspect='auto', vmin=0, vmax=attn_np.max())
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=14)
    
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_entropy(entropy_per_layer_head: Dict[Tuple[int, int], float],
                           save_path: str):
    """
    Plot attention entropy across layers and heads.
    
    Args:
        entropy_per_layer_head: Dict mapping (layer, head) to mean entropy
        save_path: Path to save the plot
    """
    layers = sorted(set(k[0] for k in entropy_per_layer_head.keys()))
    heads = sorted(set(k[1] for k in entropy_per_layer_head.keys()))
    
    entropy_matrix = np.zeros((len(layers), len(heads)))
    for (l, h), val in entropy_per_layer_head.items():
        entropy_matrix[layers.index(l), heads.index(h)] = val
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(entropy_matrix, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(heads)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f'Head {h}' for h in heads])
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    
    ax.set_title('Mean Attention Entropy per Layer/Head', fontsize=14)
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Entropy (nats)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mean_attention_distance(distance_per_layer_head: Dict[Tuple[int, int], float],
                                  save_path: str):
    """
    Plot mean attention distance across layers and heads.
    
    Args:
        distance_per_layer_head: Dict mapping (layer, head) to mean distance
        save_path: Path to save the plot
    """
    layers = sorted(set(k[0] for k in distance_per_layer_head.keys()))
    heads = sorted(set(k[1] for k in distance_per_layer_head.keys()))
    
    dist_matrix = np.zeros((len(layers), len(heads)))
    for (l, h), val in distance_per_layer_head.items():
        dist_matrix[layers.index(l), heads.index(h)] = val
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, cmap='plasma', aspect='auto')
    
    ax.set_xticks(range(len(heads)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f'Head {h}' for h in heads])
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    
    ax.set_title('Mean Attention Distance per Layer/Head', fontsize=14)
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Distance (positions)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_attention_heads(model, tokenizer, test_sentences: List[str], 
                           layer_indices: List[int], head_indices: List[int],
                           device: str = 'cpu', save_dir: str = 'phase2/results/attention'):
    """
    Comprehensive attention analysis for specified layers and heads.
    
    Returns:
        Dictionary with entropy, distance, and attention maps
    """
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'entropy': {},
        'distance': {},
        'attention_maps': {},
        'sentences': []
    }
    
    with torch.no_grad():
        for sent_idx, sentence in enumerate(test_sentences):
            input_ids = torch.tensor([encode_text(sentence, tokenizer)], dtype=torch.long).to(device)
            tokens = tokenizer.tokenize(sentence)
            
            logits, all_attention = model(input_ids, return_attention=True)
            
            results['sentences'].append({
                'text': sentence,
                'tokens': tokens
            })
            
            for layer_idx in layer_indices:
                if layer_idx >= len(all_attention):
                    continue
                    
                attn = all_attention[layer_idx].squeeze(0)
                
                for head_idx in head_indices:
                    if head_idx >= attn.size(0):
                        continue
                    
                    head_attn = attn[head_idx]
                    
                    entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-10), dim=-1).mean().item()
                    results['entropy'][(layer_idx, head_idx)] = results['entropy'].get((layer_idx, head_idx), []) + [entropy]
                    
                    T = head_attn.size(0)
                    distances = torch.arange(T, device=device).float()
                    distances = distances.unsqueeze(1).expand(T, T)
                    distances = torch.abs(distances - distances.t())
                    
                    causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                    distances = distances.masked_fill(causal_mask, 0)
                    
                    mean_dist = torch.sum(head_attn * distances).item() / T
                    results['distance'][(layer_idx, head_idx)] = results['distance'].get((layer_idx, head_idx), []) + [mean_dist]
                    
                    key = f"layer_{layer_idx}_head_{head_idx}"
                    if key not in results['attention_maps']:
                        results['attention_maps'][key] = []
                    results['attention_maps'][key].append(head_attn.cpu().numpy())
    
    avg_entropy = {k: np.mean(v) for k, v in results['entropy'].items()}
    avg_distance = {k: np.mean(v) for k, v in results['distance'].items()}
    
    results['avg_entropy'] = avg_entropy
    results['avg_distance'] = avg_distance
    
    with open(save_path / 'attention_analysis.json', 'w') as f:
        json.dump({
            'entropy': {f"{l}_{h}": v for (l, h), v in avg_entropy.items()},
            'distance': {f"{l}_{h}": v for (l, h), v in avg_distance.items()},
            'sentences': results['sentences']
        }, f, indent=2)
    
    plot_attention_entropy(avg_entropy, str(save_path / 'entropy_heatmap.png'))
    plot_mean_attention_distance(avg_distance, str(save_path / 'distance_heatmap.png'))
    
    for key, attn_maps in results['attention_maps'].items():
        if attn_maps:
            # Handle variable sequence lengths - pad to max
            max_len = max(a.shape[0] for a in attn_maps)
            padded = []
            for a in attn_maps:
                if a.shape[0] < max_len:
                    # Pad rows
                    pad_rows = np.zeros((max_len - a.shape[0], a.shape[1]))
                    a_padded = np.vstack([a, pad_rows])
                    # Pad cols
                    if a_padded.shape[1] < max_len:
                        pad_cols = np.zeros((max_len, max_len - a_padded.shape[1]))
                        a_padded = np.hstack([a_padded, pad_cols])
                    padded.append(a_padded)
                else:
                    if a.shape[1] < max_len:
                        pad_cols = np.zeros((a.shape[0], max_len - a.shape[1]))
                        a = np.hstack([a, pad_cols])
                    padded.append(a)
            avg_attn = np.mean(padded, axis=0)
            layer_idx = int(key.split('_')[1])
            head_idx = int(key.split('_')[3])
            tokens = results['sentences'][0]['tokens'] if results['sentences'] else []
            plot_attention_heatmap(
                avg_attn, tokens[:avg_attn.shape[0]], 
                layer_idx, head_idx,
                str(save_path / f'{key}_avg.png'),
                f'Average {key}'
            )
    
    return results


def identify_head_types(entropy: Dict[Tuple[int, int], float], 
                        distance: Dict[Tuple[int, int], float],
                        entropy_threshold_low: float = 1.0,
                        entropy_threshold_high: float = 2.5,
                        distance_threshold_local: float = 3.0,
                        distance_threshold_long: float = 10.0) -> Dict:
    """
    Classify attention heads by their behavior.
    
    Returns:
        Dict mapping head types to list of (layer, head) tuples
    """
    head_types = {
        'local_concentrated': [],
        'local_diffuse': [],
        'long_range_concentrated': [],
        'long_range_diffuse': [],
        'positional': [],
        'content_dependent': []
    }
    
    for (layer, head), ent in entropy.items():
        dist = distance.get((layer, head), 0)
        
        if dist <= distance_threshold_local:
            if ent <= entropy_threshold_low:
                head_types['local_concentrated'].append((layer, head))
            elif ent >= entropy_threshold_high:
                head_types['local_diffuse'].append((layer, head))
        elif dist >= distance_threshold_long:
            if ent <= entropy_threshold_low:
                head_types['long_range_concentrated'].append((layer, head))
            elif ent >= entropy_threshold_high:
                head_types['long_range_diffuse'].append((layer, head))
    
    return head_types


def encode_text(text: str, tokenizer) -> List[int]:
    """Encode text to token IDs."""
    tokens = tokenizer.tokenize(text)
    return [tokenizer.vocab.get(token, 0) for token in tokens]


if __name__ == "__main__":
    print("Attention analysis module loaded successfully")
    print("Available functions:")
    print("  - plot_attention_heatmap")
    print("  - plot_attention_entropy")
    print("  - plot_mean_attention_distance")
    print("  - analyze_attention_heads")
    print("  - identify_head_types")