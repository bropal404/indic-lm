#!/usr/bin/env python3
"""Nepali Transformer Language Model Trainer.

Entry point: python3 nepali/train/training.py or python3 -m nepali.train

Phase 2 training for Model L (Nepali) - CPU optimized, no mixed precision.
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
import math


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
        
        if is_best:
            best_path = Path(path).parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.log_file.write(f"Saved best model at step {self.step}\n")
            self.log_file.flush()
    
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
        
        eval_steps = num_batches or self.config['eval']['eval_every']
        
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
        
        self.log_file.write(f"Starting training for {max_steps} steps\n")
        self.log_file.flush()
        
        pbar = tqdm(total=max_steps, initial=self.step, desc="Training", miniters=100)
        
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
                        Path(self.config['checkpoint']['save_dir']) / f"step_{self.step}.pt",
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
        
        final_path = Path(self.config['checkpoint']['save_dir']) / "final_model.pt"
        self.save_checkpoint(final_path)
        
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


def create_model_from_config(config: Dict[str, Any]):
    """Create model from configuration."""
    from nepali.train.model import create_model_from_config as _create_model
    
    model = _create_model(config)
    return model


if __name__ == "__main__":
    import sys
    
    from tokenizer_utils import load_tokenizer
    from dataset import create_dataloaders
    from model import TransformerLM
    
    config = load_config('nepali/configs/model_l.yaml')
    
    tokenizer = load_tokenizer(config['data']['tokenizer_path'])
    
    model = create_model_from_config(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader, val_loader = create_dataloaders(
        config['data']['train_file'],
        config['data']['val_file'],
        tokenizer,
        context_length=config['model']['context_length'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        text_field=config['data']['text_field']
    )
    
    trainer = Trainer(config)
    trainer.setup(model, tokenizer, train_loader, val_loader)
    
    print("Trainer setup complete!")