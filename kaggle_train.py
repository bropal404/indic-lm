#!/usr/bin/env python3
"""
Kaggle Training Script with Auto-Recovery and HF Checkpoint Sync for Hindi LM.

Features:
- Auto-detects Kaggle GPU (CUDA) and enables AMP (Mixed Precision FP16) for fast training.
- Streams / Loads tokenized dataset directly from Hugging Face Hub (bropal404/lma-project).
- Auto-syncs checkpoints from and to Hugging Face Model Hub (bropal404/lma-project-checkpoints).
- If Kaggle session times out or restarts, it seamlessly pulls the latest checkpoint from HF and continues.
"""

import os
import sys
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    os.system("pip install -q huggingface_hub datasets pyarrow")
    from huggingface_hub import HfApi, snapshot_download

# Kaggle working directories
WORKING_DIR = Path("/kaggle/working" if os.path.exists("/kaggle") else "./kaggle_output")
CHECKPOINT_DIR = WORKING_DIR / "checkpoints"
LOG_DIR = WORKING_DIR / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    "hf_dataset": os.getenv("HF_DATASET", "bropal404/lma-project"),
    "hf_model_repo": os.getenv("HF_MODEL_REPO", "bropal404/lma-project-checkpoints"),
    "model": {
        "vocab_size": 30980,
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 6,
        "d_ff": 1536,
        "context_length": 256,
        "dropout": 0.1,
        "weight_tying": True,
    },
    "training": {
        "batch_size": 32,                 # Scaled for GPU
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "warmup_steps": 2000,
        "max_steps": 50000,
        "gradient_accumulation_steps": 2, # Effective batch size = 64
        "gradient_clip": 1.0,
        "mixed_precision": torch.cuda.is_available(),
        "save_every": 1000,
        "eval_every": 500,
        "eval_steps": 50,
        "log_every": 50,
        "keep_last_n": 3,
    },
    "seed": 42,
}


# --- Transformer Architecture (Self-contained for easy single-file execution) ---
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, context_length: int = 256):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.context_length = context_length

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length) * float("-inf"), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.causal_mask[:T, :T]
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(nn.functional.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, context_length: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 context_length: int = 256, dropout: float = 0.1, weight_tying: bool = True):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, context_length) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.weight_tying = weight_tying
        if weight_tying:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# --- Dataset from Hugging Face Pre-tokenized Parquet ---
class TokenizedHFDataset(Dataset):
    def __init__(self, hf_dataset, context_length: int = 256):
        self.dataset = hf_dataset
        self.context_length = context_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        tokens = torch.tensor(row["input_ids"], dtype=torch.long)
        if tokens.numel() < self.context_length:
            pad = torch.zeros(self.context_length - tokens.numel(), dtype=torch.long)
            tokens = torch.cat([tokens, pad])
        elif tokens.numel() > self.context_length:
            tokens = tokens[:self.context_length]

        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids


# --- HF Checkpointing & Recovery ---
def sync_checkpoints_from_hf(repo_id: str, local_dir: Path, token: str = None):
    token = token or os.getenv("HF_TOKEN")
    if not token:
        print("[HF Sync] No HF_TOKEN provided. Skipping HF download.")
        return
    try:
        api = HfApi(token=token)
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"[HF Sync] Downloading checkpoints from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            allow_patterns=["*.pt"],
            token=token,
        )
        print("[HF Sync] Download finished.")
    except Exception as e:
        print(f"[HF Sync] No existing checkpoints downloaded ({e}). Starting fresh or with local.")


def upload_checkpoint_to_hf(file_path: Path, repo_id: str, token: str = None, msg: str = ""):
    token = token or os.getenv("HF_TOKEN")
    if not token:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg or f"Upload {file_path.name}",
        )
        print(f"[HF Sync] Uploaded {file_path.name} to {repo_id}")
    except Exception as e:
        print(f"[HF Sync] Warning: Upload failed: {e}")


def get_latest_checkpoint(local_dir: Path):
    pts = sorted(local_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


def main():
    hf_token = os.getenv("HF_TOKEN")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Sync any existing checkpoints from Hugging Face Hub for auto-recovery
    sync_checkpoints_from_hf(CONFIG["hf_model_repo"], CHECKPOINT_DIR, token=hf_token)

    # 2. Load dataset from HF Hub
    print(f"\nLoading tokenized dataset from Hugging Face: {CONFIG['hf_dataset']}...")
    ds = load_dataset(CONFIG["hf_dataset"], token=hf_token)
    train_dataset = TokenizedHFDataset(ds["train"], context_length=CONFIG["model"]["context_length"])
    val_dataset = TokenizedHFDataset(ds["validation"], context_length=CONFIG["model"]["context_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Dataset loaded: {len(train_dataset):,} train samples, {len(val_dataset):,} val samples")

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(**CONFIG["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {num_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=float(CONFIG["training"]["learning_rate"]),
        weight_decay=float(CONFIG["training"]["weight_decay"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    max_steps = CONFIG["training"]["max_steps"]
    warmup_steps = CONFIG["training"]["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["training"]["mixed_precision"])

    step = 0
    best_val_loss = float("inf")

    # 4. Resume Checkpoint if available
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"\n[Recovery] Resuming training from: {latest_ckpt.name}")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
        step = ckpt_data["step"]
        best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
        print(f"[Recovery] Resumed at step {step}, best_val_loss: {best_val_loss:.4f}\n")
    else:
        print("\nStarting training from step 0\n")

    # 5. Training Loop
    grad_accum = CONFIG["training"]["gradient_accumulation_steps"]
    pbar = tqdm(total=max_steps, initial=step, desc="Training")

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            input_ids, target_ids = [t.to(device) for t in batch]
            model.train()

            with torch.cuda.amp.autocast(enabled=CONFIG["training"]["mixed_precision"]):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-1)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % CONFIG["training"]["log_every"] == 0:
                cur_loss = loss.item() * grad_accum
                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{cur_loss:.4f}", "lr": f"{cur_lr:.6f}"})

            # Validation and Checkpoint
            if step > 0 and step % CONFIG["training"]["eval_every"] == 0:
                model.eval()
                val_loss = 0.0
                val_tokens = 0
                eval_batches = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        if eval_batches >= CONFIG["training"]["eval_steps"]:
                            break
                        v_inp, v_tgt = [t.to(device) for t in v_batch]
                        with torch.cuda.amp.autocast(enabled=CONFIG["training"]["mixed_precision"]):
                            v_logits = model(v_inp)
                            v_loss = nn.functional.cross_entropy(v_logits.view(-1, v_logits.size(-1)), v_tgt.view(-1), ignore_index=-1)
                        val_loss += v_loss.item() * v_inp.numel()
                        val_tokens += v_inp.numel()
                        eval_batches += 1

                avg_val_loss = val_loss / val_tokens if val_tokens > 0 else float("inf")
                val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
                print(f"\n[Step {step}] Val Loss: {avg_val_loss:.4f} | Perplexity: {val_ppl:.2f}")

                # Save Checkpoint
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss

                ckpt_file = CHECKPOINT_DIR / f"step_{step}.pt"
                ckpt_dict = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": CONFIG,
                }
                torch.save(ckpt_dict, ckpt_file)

                # Sync step checkpoint to HF
                upload_checkpoint_to_hf(
                    ckpt_file,
                    CONFIG["hf_model_repo"],
                    token=hf_token,
                    msg=f"Step {step} checkpoint (val_loss: {avg_val_loss:.4f})"
                )

                if is_best:
                    best_file = CHECKPOINT_DIR / "best_model.pt"
                    torch.save(ckpt_dict, best_file)
                    upload_checkpoint_to_hf(
                        best_file,
                        CONFIG["hf_model_repo"],
                        token=hf_token,
                        msg=f"Best model update (val_loss: {best_val_loss:.4f})"
                    )

                # Prune older local checkpoints to avoid filling Kaggle disk
                all_ckpts = sorted(CHECKPOINT_DIR.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
                keep_n = CONFIG["training"]["keep_last_n"]
                for old in all_ckpts[:-keep_n]:
                    old.unlink()

            step += 1
            pbar.update(1)

    pbar.close()
    final_file = CHECKPOINT_DIR / "final_model.pt"
    torch.save({"step": step, "model_state_dict": model.state_dict(), "config": CONFIG}, final_file)
    upload_checkpoint_to_hf(final_file, CONFIG["hf_model_repo"], token=hf_token, msg="Final model")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
