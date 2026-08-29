#!/usr/bin/env python
"""
Training script for Phase 2 models.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yaml
import argparse
from tokenizer_utils import load_tokenizer
from dataset import create_dataloaders
from training import Trainer, create_model_from_config


def main():
    parser = argparse.ArgumentParser(description='Train Phase 2 Transformer LM')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))
    
    print(f"Loading tokenizer from {config['data']['tokenizer_path']}...")
    tokenizer = load_tokenizer(config['data']['tokenizer_path'])
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    print("Creating model...")
    model = create_model_from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Creating dataloaders (streaming mode)...")
    train_loader, val_loader = create_dataloaders(
        config['data']['train_file'],
        config['data']['val_file'],
        tokenizer,
        context_length=config['model']['context_length'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        text_field=config['data']['text_field'],
        streaming=True
    )
    
    print("Setting up trainer...")
    trainer = Trainer(config)
    trainer.setup(model, tokenizer, train_loader, val_loader)
    
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()