"""Data loader creation for the Nepali Trainer."""

import json
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch


class TextDataset(Dataset):
    """Simple dataset that yields (input_ids, target_ids) pairs from a JSONL file."""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        context_length: int,
        text_field: str = "text",
        max_examples: int | None = None,
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.text_field = text_field

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_field) or obj.get("text") or ""
                if not text.strip():
                    continue
                # Tokenize and add to dataset
                tokens = tokenizer.tokenize(str(text))
                if len(tokens) < 2:
                    continue
                # Truncate or pad to context_length
                if len(tokens) > context_length:
                    tokens = tokens[:context_length]
                # Create input/output pairs
                # input_ids: tokens[:-1], target_ids: tokens[1:]
                if len(tokens) < 2:
                    continue
                self.examples.append(torch.tensor(tokens, dtype=torch.long))

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # input is all but last token, target is all but first token (shifted)
        if len(tokens) <= 1:
            # pad
            input_ids = tokens
            target_ids = tokens.new_zeros((1,), dtype=torch.long)
        else:
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
        # Pad to context_length if needed
        if input_ids.numel() < self.context_length:
            pad_len = self.context_length - input_ids.numel()
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.zeros(pad_len, dtype=torch.long)])
        return input_ids, target_ids


def create_dataloaders(
    train_file: str,
    val_file: str,
    tokenizer,
    context_length: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    text_field: str = "text",
    max_examples: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from JSONL files.

    Returns (train_loader, val_loader)
    """
    random.seed(seed)

    train_dataset = TextDataset(
        train_file, tokenizer, context_length, text_field, max_examples
    )
    val_dataset = TextDataset(
        val_file, tokenizer, context_length, text_field, max_examples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader