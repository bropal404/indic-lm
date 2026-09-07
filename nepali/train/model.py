"""Model creation from configuration for the Nepali Trainer."""

import torch
import torch.nn as nn
from pathlib import Path

from model import GPTLanguageModel


def create_model_from_config(config: dict) -> GPTLanguageModel:
    """Create a GPTLanguageModel instance from the given configuration dict.

    Expected config structure:
    {
        "model": {
            "vocab_size": int,
            "d_model": int,
            "n_layers": int,
            "n_heads": int,
            "d_ff": int,
            "context_length": int,
            "dropout": float,
            "weight_tying": bool,
        },
        "data": {
            "text_field": str,
        }
    }
    """
    model_cfg = config["model"]
    model = GPTLanguageModel(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        context_len=model_cfg["context_length"],
        dropout=model_cfg.get("dropout", 0.1),
    )
    return model


def count_trainable_parameters(model: GPTLanguageModel) -> int:
    """Count the number of trainable parameters in the model (excluding embeddings when weight tying is on)."""
    n_params = sum(p.numel() for p in model.parameters())
    if model.weight_tying:
        n_params -= model.token_embedding.weight.numel()
    return n_params