#!/usr/bin/env python3
"""Hindi Transformer Language Model Trainer.

Entry point: python3 hindi/train/training.py or python3 -m hindi.train

Phase 2 training for Model H (Hindi) - CPU optimized, no mixed precision.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure hindi/ is on the path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.checkpoint import (
    find_latest_checkpoint,
    find_best_checkpoint,
    load_checkpoint_for_resume,
    upload_checkpoint_to_hf,
)
from train.model import create_model_from_config, count_trainable_parameters
from train.loader import create_dataloaders as create_text_dataloaders


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Create a schedule with warmup and cosine decay."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer:
    """Trainer class for Transformer language models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.tokenizer = None

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.tokens_processed = 0

        self.writer = None
        self.log_file = None

    def setup(self, model, tokenizer, train_loader, val_loader):
        """Setup trainer components."""
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            betas=(0.9, 0.95),
            eps=1e-8
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=self.config['training']['max_steps']
        )

        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir / f"run_{int(time.time())}")

        self.log_file = open(log_dir / "training_log.txt", "w")
        self.log_file.write(f"Training started at {time.ctime()}\n")
        self.log_file.write(f"Config: {self.config}\n\n")
        self.log_file.flush()

        self._log_model_info()

    def _log_model_info(self):
        """Log model parameter counts."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.log_file.write(f"Total parameters: {total_params:,}\n")
        self.log_file.write(f"Trainable parameters: {trainable_params:,}\n")
        self.log_file.flush()

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'tokens_processed': self.tokens_processed,
            'config': self.config
        }

        torch.save(checkpoint, path)

        # Upload step checkpoint to Hugging Face Hub if configured
        hf_repo = self.config.get('checkpoint', {}).get('hf_repo_id') or os.getenv("HF_MODEL_REPO")
        if hf_repo:
            upload_checkpoint_to_hf(path, hf_repo, commit_message=f"Step {self.step} checkpoint (loss: {self.best_val_loss:.4f})")

        if is_best:
            best_path = Path(path).parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log_file.write(f"Saved best model at step {self.step}\n")
            self.log_file.flush()
            if hf_repo:
                upload_checkpoint_to_hf(str(best_path), hf_repo, commit_message=f"Best model update at step {self.step}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.tokens_processed = checkpoint['tokens_processed']

        print(f"Loaded checkpoint from step {self.step}")
        self.log_file.write(f"Loaded checkpoint from step {self.step}\n")
        self.log_file.flush()

    def train_step(self, batch):
        """Perform a single training step."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        logits = self.model(input_ids)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-1
        )

        loss = loss / self.config['training']['gradient_accumulation_steps']
        loss.backward()

        self.tokens_processed += input_ids.numel()

        return loss.item() * self.config['training']['gradient_accumulation_steps']

    @torch.no_grad()
    def evaluate(self, num_batches: Optional[int] = None):
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        batches = 0

        eval_steps = num_batches or self.config['eval']['eval_steps']

        for batch in self.val_loader:
            if batches >= eval_steps:
                break

            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits = self.model(input_ids)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-1
            )

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
            batches += 1

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        return avg_loss

    def train(self):
        """Main training loop."""
        max_steps = self.config['training']['max_steps']
        grad_accum = self.config['training']['gradient_accumulation_steps']
        grad_clip = self.config['training']['gradient_clip']
        eval_every = self.config['eval']['eval_every']
        save_every = self.config['checkpoint']['save_every']
        log_every = self.config['logging']['log_every']
        save_dir = self.config['checkpoint']['save_dir']

        # Checkpoint resume logic: auto-detect latest checkpoint (syncing from HF if configured)
        hf_repo = self.config.get('checkpoint', {}).get('hf_repo_id') or os.getenv("HF_MODEL_REPO")
        state = load_checkpoint_for_resume(self, save_dir, self.device, hf_repo_id=hf_repo)

        self.log_file.write(f"Starting training for {max_steps} steps (resume step={self.step})\n")
        self.log_file.flush()

        pbar = tqdm(total=max_steps, initial=self.step, desc="Training")

        while self.step < max_steps:
            self.epoch += 1

            for batch in self.train_loader:
                if self.step >= max_steps:
                    break

                loss = self.train_step(batch)

                if (self.step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.step % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('train/loss', loss, self.step)
                    self.writer.add_scalar('train/learning_rate', lr, self.step)
                    self.writer.add_scalar('train/tokens_processed', self.tokens_processed, self.step)

                    self.log_file.write(
                        f"Step {self.step} | Loss: {loss:.4f} | LR: {lr:.6f} | Tokens: {self.tokens_processed:,}\n"
                    )
                    self.log_file.flush()

                if self.step > 0 and self.step % eval_every == 0:
                    val_loss = self.evaluate()
                    ppl = math.exp(val_loss) if val_loss < 20 else float('inf')

                    self.writer.add_scalar('val/loss', val_loss, self.step)
                    self.writer.add_scalar('val/perplexity', ppl, self.step)

                    self.log_file.write(
                        f"Step {self.step} | Val Loss: {val_loss:.4f} | Val PPL: {ppl:.2f}\n"
                    )
                    self.log_file.flush()

                    print(f"\nStep {self.step} | Val Loss: {val_loss:.4f} | Val PPL: {ppl:.2f}")

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss

                    self.save_checkpoint(
                        Path(save_dir) / f"step_{self.step}.pt",
                        is_best=is_best
                    )

                    self._cleanup_old_checkpoints()

                self.step += 1
                pbar.update(1)

                if self.step >= max_steps:
                    break

        pbar.close()
        self.writer.close()
        self.log_file.close()

        final_path = Path(save_dir) / "final_model.pt"
        self.save_checkpoint(final_path, is_best=self.best_val_loss < float('inf'))

        print("Training completed!")

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        keep_n = self.config['checkpoint']['keep_last_n']
        save_dir = Path(self.config['checkpoint']['save_dir'])

        checkpoints = sorted(save_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
        for ckpt in checkpoints[:-keep_n]:
            ckpt.unlink()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_and_tokenizer(config: Dict[str, Any]):
    """Create model and load tokenizer from config."""
    # Create model
    model = create_model_from_config(config)

    # Load tokenizer - try the path from config, fall back to common locations
    tokenizer_path = config['data']['tokenizer_path']
    if os.path.exists(tokenizer_path):
        import pickle
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        # Handle both dict and customBPE object formats
        if isinstance(tokenizer_data, dict) and "merges" in tokenizer_data:
            # It's a saved model dict
            class _Tokenizer:
                def __init__(self, data):
                    self.num_merges = data.get("num_merges", 0)
                    self.min_pair_frequency = data.get("min_pair_frequency", 2)
                    self.merges = data["merges"]
                    self.vocab = data.get("vocab", {})
                    self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges.keys())}
                    self._token_cache = {}
                def tokenize(self, text):
                    # Simple tokenization using merge ranks
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
        elif hasattr(tokenizer_data, 'tokenize'):
            tokenizer = tokenizer_data
        else:
            tokenizer = None
    else:
        # Fallback: try to load via the tokenizer CLI's load mechanism
        from tokenizer.cli import load_data as _load_data
        # Just use a simple whitespace tokenizer as fallback
        from tokenizer.tokenizers import WhitespaceTokenizer
        tokenizer = WhitespaceTokenizer()

    print(f"Model parameters: {count_trainable_parameters(model):,}")
    print(f"Tokenizer loaded from: {tokenizer_path}")
    return model, tokenizer


def main():
    """Main entry point."""
    import sys

    # Determine config path: first arg, or default
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        config_path = sys.argv[1]
    else:
        config_path = 'phase2/configs/model_h.yaml'

    config = load_config(config_path)

    # Handle resume from checkpoint (config may have resume_from_checkpoint key)
    resume_from = config.get('resume_from_checkpoint', None)

    # Create model and load tokenizer
    model, tokenizer = create_model_and_tokenizer(config)

    # Create dataloaders
    train_loader, val_loader = create_text_dataloaders(
        config['data']['train_file'],
        config['data']['val_file'],
        tokenizer,
        context_length=config['model']['context_length'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        text_field=config['data']['text_field'],
    )

    trainer = Trainer(config)
    trainer.setup(model, tokenizer, train_loader, val_loader)

    # Print resume info
    ckpt = find_latest_checkpoint(config['checkpoint']['save_dir'])
    if resume_from:
        print(f"Resuming training from: {resume_from}")
    elif ckpt:
        step_num = ckpt.stem.split('_')[-1]
        print(f"Latest checkpoint: {ckpt} (will resume from step {step_num})")
    else:
        print("Starting training from scratch.")

    print("Trainer setup complete!")

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()