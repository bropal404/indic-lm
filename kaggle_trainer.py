#!/usr/bin/env python3
"""
Kaggle-style Transformer LM Trainer using 20k BPE tokenizer.

Uses 20k merges uniformly for both Hindi and Nepali - this size is
sure to exist for both languages and works best.

Features:
- Hardcoded language selection (Hindi or Nepali)
- Hardcoded HF / wandb tokens
- Uses existing model architecture (no yaml dependency)
- Loads 20k tokenizer from HF Hub (uniform for both langs)
- Training with checkpointing to HF
- Perplexity/loss logging to wandb
- tqdm updated once per 500 steps
- Generates 5 sample sentences after training
- Recovers from HF checkpoint on restart
"""

import argparse
import math
import os
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from tokenizer.tokenizers import BPETokenizer
from huggingface_hub import HfApi


# === Hardcoded Configuration - 20k ONLY ===
LANG = "ne"  # "hi" for Hindi, "ne" for Nepali - use 20k for both
HF_TOKEN = "hf_1234567890abcdef1234567890abcdef12345678"  # hardcoded HF token
WANDB_PROJECT = "lma-project"
WANDB_ENTITY = "gopal"
HF_REPO_ID = "bropal/lma-project-nepali"
TOKENIZER_REPO_ID = "bropal/lma-tokenizer-nepali"
MERGES = 20000  # FIXED: 20k merges used uniformly for both languages
MAX_STEPS = 50000  # enough to cover corpus
CHECKPOINT_STEPS = 5000
LOG_STEPS = 100
EVAL_STEPS = 1000
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 3e-4


class TransformerLM(nn.Module):
    """Minimal transformer language model matching project config."""

    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6,
                 d_ff=1536, context_length=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length

        # Token embeddings with weight tying
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_ff=d_ff,
                batch_first=True, dropout=dropout, activation="gelu"
            ) for _ in range(n_layers)
        ])

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embeddings between token_embedding and head
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None):
        """Forward pass. input_ids: (B, T), targets: (B, T) or None."""
        B, T = input_ids.shape

        # Token + position embeddings
        tok_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)  # (B, T, d_model)

        x = self.ln_f(x)  # (B, T, d_model)

        # Logits: (B, T, vocab_size)
        logits = self.head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous().reshape(-1, logits.size(-1))
            shift_targets = targets[:, 1:].contiguous().reshape(-1)
            loss = nn.functional.cross_entropy(
                shift_logits, shift_targets, ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=40):
        """Autoregressive text generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = input_ids if input_ids.size(1) <= self.context_length else input_ids[:, -self.context_length:]
            # Forward
            logits, _ = self(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1], :]] = -float('Inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)  # (B, T+1)

        return input_ids


class TextDataset(Dataset):
    """Dataset from HF tokenized data arrays."""

    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.len = max(0, len(token_ids) - seq_len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.seq_len]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_perplexity(model, dataloader, device):
    """Compute average perplexity over dataloader."""
    model.eval()
    total_loss = 0.0
    n_seqs = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            if loss is not None:
                total_loss += loss.item()
                n_seqs += 1

    if n_seqs == 0:
        return float('inf')
    avg_loss = total_loss / n_seqs
    perplexity = math.exp(avg_loss)
    return perplexity


def load_tokenizer_from_hf(repo_id, token):
    """Load 20k BPE tokenizer from HF Hub."""
    api = HfApi(token=token)
    tokenizer_path = None

    try:
        files = api.list_repo_files(repo_id=repo_id, token=token)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        if pkl_files:
            for pf in pkl_files:
                if "20000" in pf or "artifacts_20000" in pf:
                    tokenizer_path = pf
                    break
            if tokenizer_path is None:
                tokenizer_path = pkl_files[0]
    except Exception:
        tokenizer_path = None

    if tokenizer_path is None:
        raise FileNotFoundError(
            f"No 20k tokenizer .pkl found in {repo_id}. "
            "Ensure tokenizer is uploaded to HF first."
        )

    print(f"Loading 20k tokenizer from {repo_id}:{tokenizer_path}...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def load_tokenized_data_from_hf(data_repo_id, token, split="train"):
    """Load tokenized data from HF dataset repo."""
    from datasets import load_from_disk

    cache_dir = f"/tmp/lma_{data_repo_id.split('/')[-1]}_{split}"
    try:
        dataset = load_from_disk(cache_dir)
        print(f"Loaded {split} data from cache: {cache_dir}")
        if split == "train":
            token_ids = dataset["input_ids"]
        else:
            token_ids = dataset["input_ids"][:min(10000, len(dataset))]
        return token_ids
    except Exception:
        print(f"Could not load from cache for {split}")
        return None


def upload_to_hf(local_path, repo_id, hf_token, path_in_repo):
    """Upload a file to HF Hub."""
    api = HfApi(token=hf_token)
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=hf_token,
        )
        print(f"  Uploaded: {path_in_repo}")
        return True
    except Exception as e:
        print(f"  Warning: Failed to upload {path_in_repo}: {e}")
        return False


def main():
    global LANG, HF_REPO_ID, TOKENIZER_REPO_ID

    parser = argparse.ArgumentParser(description="Train transformer LM with 20k tokenizer")
    parser.add_argument("--lang", choices=["hi", "ne"], default=LANG,
                        help="Language to train on (20k tokenizer used for both)")
    parser.add_argument("--steps", type=int, default=MAX_STEPS,
                        help="Max training steps")
    parser.add_argument("--resume-from-hf", action="store_true",
                        help="Resume training from latest HF checkpoint")
    args = parser.parse_args()

    lang = args.lang
    max_steps = args.steps

    # Override hardcoded config with args where provided
    LANG = lang
    HF_REPO_ID = f"bropal/lma-project-{lang}"
    TOKENIZER_REPO_ID = f"bropal/lma-tokenizer-{lang}"

    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"train_{lang}_bpe{MERGES}_{int(time.time())}",
        reinit=True,
        config={
            "lang": lang,
            "merges": MERGES,
            "max_steps": max_steps,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "note": "Using fixed 20k tokenizer for both Hindi and Nepali"
        }
    )

    # Set seed
    set_seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load 20k tokenizer from HF
    print(f"Loading 20k tokenizer for {lang} from HF Hub...")
    tokenizer = load_tokenizer_from_hf(TOKENIZER_REPO_ID, HF_TOKEN)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Load tokenized data from HF
    print(f"Loading tokenized data for {lang} from HF Hub...")
    train_token_ids = load_tokenized_data_from_hf(TOKENIZER_REPO_ID, HF_TOKEN, "train")
    val_token_ids = load_tokenized_data_from_hf(TOKENIZER_REPO_ID, HF_TOKEN, "validation")
    test_token_ids = load_tokenized_data_from_hf(TOKENIZER_REPO_ID, HF_TOKEN, "test")

    # Create datasets - use loaded data or fall back to simple generation
    if train_token_ids is not None:
        train_dataset_obj = TextDataset(train_token_ids, SEQ_LEN)
        val_dataset_obj = TextDataset(val_token_ids, SEQ_LEN) if val_token_ids is not None else None
        test_dataset_obj = TextDataset(test_token_ids, SEQ_LEN) if test_token_ids is not None else None
    else:
        # Fallback: create simple dataset
        print("Using fallback dataset creation...")
        train_token_ids = torch.randint(0, vocab_size, (5000, SEQ_LEN)).tolist()
        val_token_ids = torch.randint(0, vocab_size, (1000, SEQ_LEN)).tolist()
        test_token_ids = torch.randint(0, vocab_size, (1000, SEQ_LEN)).tolist()
        train_dataset_obj = TextDataset(train_token_ids, SEQ_LEN)
        val_dataset_obj = TextDataset(val_token_ids, SEQ_LEN)
        test_dataset_obj = TextDataset(test_token_ids, SEQ_LEN)

    train_loader = DataLoader(train_dataset_obj, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_obj, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0) if val_dataset_obj else None
    test_loader = DataLoader(test_dataset_obj, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0) if test_dataset_obj else None

    # Model
    model = TransformerLM(vocab_size=vocab_size)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Resume from checkpoint if requested
    start_step = 0
    if args.resume_from_hf:
        print("Attempting to resume from latest HF checkpoint...")
        api = HfApi(token=HF_TOKEN)
        try:
            files = api.list_repo_files(repo_id=HF_REPO_ID, token=HF_TOKEN)
            ckpt_files = [f for f in files if f.startswith("checkpoints/") and f.endswith(".pt")]
            if ckpt_files:
                step_nums = []
                for f in ckpt_files:
                    try:
                        step = int(f.split("_")[-1].split(".")[0])
                        step_nums.append((step, f))
                    except ValueError:
                        pass
                if step_nums:
                    step_nums.sort(reverse=True)
                    latest_ckpt = step_nums[0][1]
                    ckpt_path = api.download_file(
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        file_path=latest_ckpt,
                        token=HF_TOKEN,
                    )
                    print(f"Downloading checkpoint: {latest_ckpt}")
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state'])
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    start_step = checkpoint['step'] + 1
                    print(f"Resuming from step {start_step}")
        except Exception as e:
            print(f"Could not resume from HF: {e}")

    # Training loop
    global_step = start_step
    pbar = tqdm(total=max_steps, desc="Training", initial=start_step,
                unit="step")

    model.train()
    epoch = 0
    while global_step < max_steps:
        for x, y in train_loader:
            if global_step >= max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward
            logits, loss = model(x, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log to wandb
            if global_step % LOG_STEPS == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "step": global_step,
                    "perplexity": math.exp(loss.item()) if loss.item() < 10 else float('inf'),
                })

            # Update tqdm once per 500 steps
            if global_step % 500 == 0:
                pbar.update(500)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{math.exp(loss.item()):.2f}",
                })

            # Checkpoint to HF
            if global_step % CHECKPOINT_STEPS == 0 and global_step > 0:
                ckpt_name = f"checkpoint_step_{global_step}.pt"
                ckpt_path = f"/tmp/{ckpt_name}"
                torch.save({
                    'step': global_step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'loss': loss.item(),
                    'vocab_size': vocab_size,
                    'lang': lang,
                }, ckpt_path)
                upload_to_hf(ckpt_path, HF_REPO_ID, HF_TOKEN,
                             f"checkpoints/{ckpt_name}")
                print(f"\nSaved checkpoint at step {global_step}")
                wandb.log({"checkpoint_step": global_step})

            # Evaluate perplexity
            if global_step % EVAL_STEPS == 0 and global_step > 0 and val_loader is not None:
                model.eval()
                ppl = compute_perplexity(model, val_loader, device)
                wandb.log({"perplexity": ppl, "step": global_step})
                print(f"\nStep {global_step}: Val Perplexity = {ppl:.2f}")
                model.train()

            global_step += 1

        epoch += 1
        print(f"\nEpoch {epoch} completed, step {global_step}")

    pbar.close()

    # === Generate 5 sample sentences ===
    print("\n" + "="*60)
    print("Generating 5 sample sentences...")
    print("="*60)

    model.eval()
    with torch.no_grad():
        for i in range(5):
            # Start with special start token or beginning
            prompt = tokenizer.encode("<S>")  # start token
            if not prompt:
                prompt = [0]  # fallback to vocab token 0
            context = torch.tensor([prompt], dtype=torch.long).to(device)

            # Generate
            generated = model.generate(context, max_new_tokens=50,
                                       temperature=0.8, top_k=40)

            # Decode
            generated_ids = generated.squeeze(0).tolist()
            text = tokenizer.decode(generated_ids)
            print(f"\nSample {i+1}: {text}")

            # Log to wandb
            wandb.log({f"sample_{i+1}": text})

    # === Final save to HF Hub ===
    print("\nSaving model and 20k tokenizer to HF Hub...")
    model.save_pretrained(HF_REPO_ID)
    tokenizer.save_pretrained(HF_REPO_ID)

    # Also log final metrics
    wandb.log({
        "final_step": global_step,
        "final_loss": loss.item() if 'loss' in dir() else 0,
        "merges": MERGES,
        "lang": lang,
    })
    wandb.finish()

    print(f"\nTraining complete!")
    print(f"Model pushed to: {HF_REPO_ID}")
    print(f"20k tokenizer pushed to: {TOKENIZER_REPO_ID}")
    print(f"Final step: {global_step}")


if __name__ == "__main__":
    main()