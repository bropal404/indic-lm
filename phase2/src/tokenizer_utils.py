import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pickle
import json
from tokenizer.tokenizers import customBPE


def load_tokenizer(tokenizer_path: str):
    """Load a trained BPE tokenizer from a .pkl file."""
    tokenizer = customBPE()
    tokenizer.load(tokenizer_path)
    return tokenizer


def get_vocab_size(tokenizer_path: str) -> int:
    """Get vocabulary size from tokenizer config."""
    config_path = Path(tokenizer_path).with_suffix('.json').with_name(
        Path(tokenizer_path).stem.replace('.pkl', '') + '_config.json'
    )
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get('vocab_size', 0)
    
    tokenizer = load_tokenizer(tokenizer_path)
    return len(tokenizer.vocab)


def tokenize_text(text: str, tokenizer: customBPE):
    """Tokenize text using the custom BPE tokenizer."""
    return tokenizer.tokenize(text)


def encode_text(text: str, tokenizer: customBPE):
    """Encode text to token IDs."""
    tokens = tokenizer.tokenize(text)
    return [tokenizer.vocab.get(token, 0) for token in tokens]


def decode_ids(ids: list, tokenizer: customBPE):
    """Decode token IDs back to text."""
    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokens = [inv_vocab.get(id, '<unk>') for id in ids]
    return ''.join(tokens)


if __name__ == "__main__":
    # Test loading tokenizers
    hi_tokenizer = load_tokenizer("hindi/tokenizer/artifacts/bpe_hi_32000.pkl")
    ne_tokenizer = load_tokenizer("nepali/tokenizer/artifacts/bpe_ne_32000.pkl")
    
    print(f"Hindi vocab size: {len(hi_tokenizer.vocab)}")
    print(f"Nepali vocab size: {len(ne_tokenizer.vocab)}")
    
    # Test tokenization
    hi_text = "नमस्ते दुनिया"
    ne_text = "नमस्ते संसार"
    
    hi_tokens = hi_tokenizer.tokenize(hi_text)
    ne_tokens = ne_tokenizer.tokenize(ne_text)
    
    print(f"Hindi tokens: {hi_tokens}")
    print(f"Nepali tokens: {ne_tokens}")
    
    hi_ids = [hi_tokenizer.vocab.get(t, 0) for t in hi_tokens]
    ne_ids = [ne_tokenizer.vocab.get(t, 0) for t in ne_tokens]
    
    print(f"Hindi IDs: {hi_ids}")
    print(f"Nepali IDs: {ne_ids}")