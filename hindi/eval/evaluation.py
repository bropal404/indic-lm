import sys
sys.path.insert(0, 'phase2/src')

import torch
import torch.nn as nn
import math
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
import os


def compute_cross_entropy_loss(model, dataloader, device, max_batches=None):
    """Compute cross-entropy loss on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_batches and batches >= max_batches:
                break

            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-1,
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += target_ids.numel()
            batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss


def compute_perplexity(cross_entropy_loss):
    """Compute perplexity from cross-entropy loss (natural log)."""
    if cross_entropy_loss >= 20:
        return float('inf')
    return math.exp(cross_entropy_loss)


def compute_bits_per_byte(model, dataloader, tokenizer, device, max_batches=None):
    """
    Compute bits per byte (BPB) on held-out raw text.
    This is more comparable across different tokenizers than bits per token.
    """
    model.eval()
    total_nll = 0.0  # negative log likelihood in nats
    total_bytes = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_batches and batches >= max_batches:
                break

            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            log_probs = nn.functional.log_softmax(logits, dim=-1)

            target_flat = target_ids.view(-1)
            log_probs_flat = log_probs.view(-1, log_probs.size(-1))

            nll = -log_probs_flat[torch.arange(target_flat.size(0)), target_flat].sum().item()
            total_nll += nll

            for i in range(input_ids.size(0)):
                tokens = input_ids[i].tolist()
                text = ''.join(chr(b) for b in tokens)
                total_bytes += len(text.encode('utf-8'))

            batches += 1

    if total_bytes == 0:
        return float('inf')

    bpb = total_nll / (total_bytes * math.log(2))
    return bpb


def decode_ids(ids: List[int], tokenizer) -> str:
    """Decode token IDs back to text."""
    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokens = [inv_vocab.get(id, '') for id in ids]
    # Join with space for word-level tokens
    return ' '.join(tokens).replace('</w>', '').strip()


def encode_text(text: str, tokenizer) -> List[int]:
    """Encode text to token IDs."""
    tokens = tokenizer.tokenize(text)
    return [tokenizer.vocab.get(token, 0) for token in tokens]


def compute_bleu(generations: List[str], references: List[str]) -> float:
    """Compute corpus-level BLEU-4."""
    if not generations or not references:
        return 0.0

    refs = [[ref] for ref in references]
    bleu = corpus_bleu(generations, refs, smooth_method='exp', smooth_value=0.0001)
    return bleu.score


def compute_chrf(generations: List[str], references: List[str]) -> float:
    """Compute chrF score."""
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
        score = chrf.corpus_score(generations, [[ref] for ref in references])
        return score.score
    except ImportError:
        return 0.0


def compute_rouge_l(generations: List[str], references: List[str]) -> float:
    """Compute ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []

    for gen, ref in zip(generations, references):
        score = scorer.score(ref, gen)
        scores.append(score['rougeL'].fmeasure)

    return np.mean(scores) * 100 if scores else 0.0


def compute_repetition_rate(text: str, n: int = 2) -> float:
    """Compute n-gram repetition rate."""
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0

    ngram_counts = Counter(ngrams)
    repeated = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated / len(ngram_counts)


def compute_distinct_ngrams(text: str, n: int) -> float:
    """Compute distinct-n (unique n-grams / total n-grams)."""
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0

    unique = len(set(ngrams))
    return unique / len(ngrams)


def compute_diversity_metrics(generations: List[str]) -> Dict[str, float]:
    """Compute all diversity metrics for a list of generations."""
    all_text = ' '.join(generations)

    return {
        'repetition_rate_2': compute_repetition_rate(all_text, 2),
        'repetition_rate_3': compute_repetition_rate(all_text, 3),
        'repetition_rate_4': compute_repetition_rate(all_text, 4),
        'distinct_1': compute_distinct_ngrams(all_text, 1),
        'distinct_2': compute_distinct_ngrams(all_text, 2),
        'distinct_3': compute_distinct_ngrams(all_text, 3),
    }


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute attention entropy for each head.
    attention_weights: (B, n_heads, T, T)
    Returns: (n_heads,) mean entropy per head
    """
    eps = 1e-10
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
    mean_entropy = entropy.mean(dim=(0, 2))
    return mean_entropy


def compute_mean_attention_distance(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute mean attention distance for each head.
    attention_weights: (B, n_heads, T, T)
    Returns: (n_heads,) mean distance per head
    """
    B, n_heads, T, _ = attention_weights.shape

    distances = torch.arange(T, device=attention_weights.device).float()
    distances = distances.unsqueeze(0).unsqueeze(0).expand(B, n_heads, T, T)
    distances = torch.abs(distances - distances.transpose(-2, -1))

    causal_mask = torch.triu(torch.ones(T, T, device=attention_weights.device), diagonal=1).bool()
    distances = distances.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0)

    mean_dist = torch.sum(attention_weights * distances, dim=-1)
    mean_dist = mean_dist.mean(dim=(0, 2))

    return mean_dist


def extract_attention_maps(model, input_ids: torch.Tensor, layer_indices: List[int],
                           head_indices: List[int]) -> Dict:
    """Extract attention maps for specified layers and heads."""
    model.eval()
    with torch.no_grad():
        logits, all_attention = model(input_ids, return_attention=True)

    results = {}
    for layer_idx in layer_indices:
        if layer_idx < len(all_attention):
            attn = all_attention[layer_idx]
            for head_idx in head_indices:
                if head_idx < attn.size(1):
                    key = f"layer_{layer_idx}_head_{head_idx}"
                    results[key] = attn[:, head_idx, :, :].cpu()

    return results


def generate_samples(model, tokenizer, prompts: List[str], max_new_tokens: int = 100,
                     temperatures: List[float] = [0.5, 1.0, 1.5],
                     device='cpu', top_k: int = 50, top_p: float = 0.9,
                     repetition_penalty: float = 1.0):
    """Generate text samples at different temperatures."""
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = torch.tensor([encode_text(prompt, tokenizer)], dtype=torch.long).to(device)

        for temp in temperatures:
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )

            generated_text = decode_ids(generated[0].tolist(), tokenizer)
            results.append({
                'prompt': prompt,
                'temperature': temp,
                'generation': generated_text[len(prompt):] if len(generated_text) > len(prompt) else generated_text,
                'full_text': generated_text
            })

    return results


def main():
    """CLI entry point for Hindi evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Hindi LM evaluation CLI")
    parser.add_argument("--config", default="phase2/configs/model_h.yaml", help="Path to model config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--prompts", nargs="+", default=["नमस्ते", "hello"], help="Generation prompts")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    from model import TransformerLM
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        context_length=config['model']['context_length'],
        dropout=config['model']['dropout'],
        weight_tying=config['model'].get('weight_tying', True)
    )
    model.to(args.device)
    model.eval()

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Load tokenizer
    tokenizer_path = config['data']['tokenizer_path']
    import pickle
    with open(tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)

    class _Tokenizer:
        def __init__(self, data):
            self.num_merges = data.get("num_merges", 0)
            self.min_pair_frequency = data.get("min_pair_frequency", 2)
            self.merges = data["merges"]
            self.vocab = data.get("vocab", {})
            self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges.keys())}
            self._token_cache = {}

        def tokenize(self, text):
            from tokenizer.tokenizers import RegexTokenizer
            pre = RegexTokenizer()
            words = pre.tokenize(text)
            result = []
            for word in words:
                tokens = list(word) + ["</w>"]
                while len(tokens) > 1:
                    pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
                    best_pair = None
                    earliest_rank = float('inf')
                    for pair in pairs:
                        rank = self.merge_ranks.get(pair)
                        if rank is not None and rank < earliest_rank:
                            earliest_rank = rank
                            best_pair = pair
                    if best_pair is None:
                        break
                    left, right = best_pair
                    merged_tokens = []
                    i = 0
                    n = len(tokens)
                    while i < n:
                        if i < n - 1 and tokens[i] == left and tokens[i+1] == right:
                            merged_tokens.append(left + right)
                            i += 2
                        else:
                            merged_tokens.append(tokens[i])
                            i += 1
                    tokens = merged_tokens
                cleaned_tokens = [t.replace('</w>', '') for t in tokens if t != '</w>']
                cleaned_tokens = [token for token in cleaned_tokens if token]
                result.extend(cleaned_tokens)
            return result

        def vocab_size(self):
            return len(self.vocab)

    tokenizer = _Tokenizer(tokenizer_data)

    # Generate samples
    with torch.no_grad():
        for prompt in args.prompts:
            input_ids = torch.tensor([encode_text(prompt, tokenizer)], dtype=torch.long).to(args.device)
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=1.0
            )
            gen_text = decode_ids(generated[0].tolist(), tokenizer)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {gen_text}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()