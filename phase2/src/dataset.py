import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Optional, Iterator
import random
import mmap
import os


class LanguageModelingDataset(Dataset):
    """
    Language Modeling Dataset for causal LM.
    
    Each example is a sequence of tokens where:
        input = tokens[t : t + T]
        target = tokens[t + 1 : t + T + 1]
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        context_length: int,
        text_field: str = "text",
        stride: Optional[int] = None
    ):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.text_field = text_field
        self.stride = stride or context_length // 2
        
        self.tokenized_data = self._load_and_tokenize()
        self.examples = self._create_examples()
    
    def _load_and_tokenize(self) -> List[int]:
        """Load and tokenize the entire dataset."""
        all_tokens = []
        
        print(f"Loading and tokenizing {self.data_file}...")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(self.text_field, '')
                    if text:
                        tokens = self.tokenizer.tokenize(text)
                        token_ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
                        all_tokens.extend(token_ids)
                except json.JSONDecodeError:
                    continue
        
        print(f"Total tokens: {len(all_tokens):,}")
        return all_tokens
    
    def _create_examples(self) -> List[torch.Tensor]:
        """Create training examples with sliding window."""
        examples = []
        for i in range(0, len(self.tokenized_data) - self.context_length, self.stride):
            example = self.tokenized_data[i:i + self.context_length + 1]
            if len(example) == self.context_length + 1:
                examples.append(torch.tensor(example, dtype=torch.long))
        
        print(f"Created {len(examples):,} examples")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example[:-1]
        target_ids = example[1:]
        return input_ids, target_ids


class StreamingLanguageModelingDataset(IterableDataset):
    """
    Streaming dataset for memory-efficient training on large corpora.
    Uses memory-mapped file reading and on-the-fly tokenization.
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        context_length: int,
        text_field: str = "text",
        buffer_size: int = 10000
    ):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.text_field = text_field
        self.buffer_size = buffer_size
        
        self.file_size = os.path.getsize(data_file)
    
    def __iter__(self):
        return self._stream_examples()
    
    def _stream_examples(self) -> Iterator[tuple]:
        """Stream examples from the dataset, looping infinitely."""
        while True:  # Loop infinitely for training
            buffer = []
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get(self.text_field, '')
                        if text:
                            tokens = self.tokenizer.tokenize(text)
                            token_ids = [self.tokenizer.vocab.get(t, 0) for t in tokens]
                            buffer.extend(token_ids)
                            
                            while len(buffer) >= self.context_length + 1:
                                example = buffer[:self.context_length + 1]
                                input_ids = torch.tensor(example[:-1], dtype=torch.long)
                                target_ids = torch.tensor(example[1:], dtype=torch.long)
                                yield input_ids, target_ids
                                buffer = buffer[self.context_length // 2:]
                                
                                if len(buffer) > self.buffer_size:
                                    buffer = buffer[-self.buffer_size:]
                    except json.JSONDecodeError:
                        continue
            
            # Yield remaining buffer
            while len(buffer) >= self.context_length + 1:
                example = buffer[:self.context_length + 1]
                input_ids = torch.tensor(example[:-1], dtype=torch.long)
                target_ids = torch.tensor(example[1:], dtype=torch.long)
                yield input_ids, target_ids
                buffer = buffer[self.context_length // 2:]


def create_dataloaders(
    train_file: str,
    val_file: str,
    tokenizer,
    context_length: int,
    batch_size: int,
    num_workers: int = 4,
    text_field: str = "text",
    streaming: bool = False,
    stride: Optional[int] = None
):
    """Create train and validation dataloaders."""
    
    if streaming:
        train_dataset = StreamingLanguageModelingDataset(
            train_file, tokenizer, context_length, text_field
        )
        val_dataset = StreamingLanguageModelingDataset(
            val_file, tokenizer, context_length, text_field
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_dataset = LanguageModelingDataset(
            train_file, tokenizer, context_length, text_field, stride
        )
        val_dataset = LanguageModelingDataset(
            val_file, tokenizer, context_length, text_field, stride
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    from tokenizer_utils import load_tokenizer
    
    tokenizer = load_tokenizer("hindi/tokenizer/artifacts/bpe_hi_32000.pkl")
    
    dataset = LanguageModelingDataset(
        "hindi/data/train.jsonl",
        tokenizer,
        context_length=512,
        text_field="text",
        stride=256
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    input_ids, target_ids = dataset[0]
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    
    train_loader, val_loader = create_dataloaders(
        "hindi/data/train.jsonl",
        "hindi/data/val.jsonl",
        tokenizer,
        context_length=512,
        batch_size=4,
        num_workers=0,
        text_field="text"
    )
    
    for batch in train_loader:
        x, y = batch
        print(f"Batch input shape: {x.shape}")
        print(f"Batch target shape: {y.shape}")
        break