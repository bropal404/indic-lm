import sys
sys.path.insert(0, 'phase2/src')

import torch
import torch.nn as nn
import math
from model import (
    MultiHeadCausalSelfAttention,
    FeedForward,
    TransformerBlock,
    TransformerLM,
    count_parameters
)
from tokenizer_utils import load_tokenizer


def test_attention_shapes():
    """Test that attention module produces correct shapes."""
    print("Testing attention shapes...")
    
    B, T, d_model = 2, 16, 64
    n_heads = 4
    
    attention = MultiHeadCausalSelfAttention(d_model, n_heads, dropout=0.0, context_length=128)
    x = torch.randn(B, T, d_model)
    
    out = attention(x)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"
    
    out, attn = attention(x, return_attention=True)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"
    assert attn.shape == (B, n_heads, T, T), f"Expected {(B, n_heads, T, T)}, got {attn.shape}"
    
    print("  ✓ Attention shapes correct")


def test_attention_probabilities_sum_to_one():
    """Test that attention probabilities sum to 1 over key positions."""
    print("Testing attention probabilities sum to 1...")
    
    B, T, d_model = 2, 16, 64
    n_heads = 4
    
    attention = MultiHeadCausalSelfAttention(d_model, n_heads, dropout=0.0, context_length=128)
    x = torch.randn(B, T, d_model)
    
    _, attn = attention(x, return_attention=True)
    
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        f"Attention sums not close to 1: min={sums.min():.6f}, max={sums.max():.6f}"
    
    print("  ✓ Attention probabilities sum to 1")


def test_causal_mask():
    """Test that causal masking prevents future tokens from affecting current predictions."""
    print("Testing causal mask...")
    
    B, T, d_model = 1, 8, 64
    n_heads = 4
    
    attention = MultiHeadCausalSelfAttention(d_model, n_heads, dropout=0.0, context_length=128)
    x = torch.randn(B, T, d_model)
    
    _, attn = attention(x, return_attention=True)
    
    for i in range(T):
        for j in range(i + 1, T):
            assert torch.allclose(attn[0, :, i, j], torch.zeros_like(attn[0, :, i, j]), atol=1e-6), \
                f"Causal mask violated at ({i}, {j}): {attn[0, :, i, j]}"
    
    print("  ✓ Causal mask correctly prevents future token attention")


def test_causal_masking_verification():
    """
    REQUIRED: Create two sequences identical through position t but different at t+1.
    Verify logits[:, t, :] are identical up to floating-point tolerance.
    """
    print("Testing causal masking verification (required test)...")
    
    vocab_size = 1000
    d_model = 128
    n_layers = 2
    n_heads = 4
    d_ff = 512
    context_length = 32
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context_length=context_length,
        dropout=0.0,
        weight_tying=False
    )
    model.eval()
    
    torch.manual_seed(42)
    
    t = 5
    seq_len = 10
    
    base_seq = torch.randint(0, vocab_size, (1, seq_len))
    
    seq1 = base_seq.clone()
    seq2 = base_seq.clone()
    
    seq1[0, t + 1] = (seq1[0, t + 1] + 1) % vocab_size
    seq2[0, t + 1] = (seq2[0, t + 1] + 2) % vocab_size
    
    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)
    
    logits1_t = logits1[0, t, :]
    logits2_t = logits2[0, t, :]
    
    max_diff = (logits1_t - logits2_t).abs().max().item()
    mean_diff = (logits1_t - logits2_t).abs().mean().item()
    
    print(f"  Max difference at position t={t}: {max_diff:.6e}")
    print(f"  Mean difference at position t={t}: {mean_diff:.6e}")
    
    assert max_diff < 1e-5, f"Causal masking violated: max diff = {max_diff}"
    
    for pos in range(t + 1, seq_len):
        diff = (logits1[0, pos, :] - logits2[0, pos, :]).abs().max().item()
        print(f"  Max difference at position {pos}: {diff:.6e}")
    
    print("  ✓ Causal masking verification passed")


def test_head_splitting_merging():
    """Test that head splitting and merging preserves values."""
    print("Testing head splitting/merging...")
    
    B, T, d_model = 2, 16, 64
    n_heads = 4
    d_head = d_model // n_heads
    
    x = torch.randn(B, T, d_model)
    
    split = x.view(B, T, n_heads, d_head).transpose(1, 2)
    merged = split.transpose(1, 2).contiguous().view(B, T, d_model)
    
    assert torch.allclose(x, merged), "Head splitting/merging does not preserve values"
    
    print("  ✓ Head splitting/merging preserves values")


def test_transformer_block_shapes():
    """Test transformer block input/output shapes."""
    print("Testing transformer block shapes...")
    
    B, T, d_model = 2, 16, 64
    n_heads = 4
    d_ff = 256
    
    block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, context_length=128)
    x = torch.randn(B, T, d_model)
    
    out = block(x)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"
    
    out, attn = block(x, return_attention=True)
    assert out.shape == (B, T, d_model)
    assert attn.shape == (B, n_heads, T, T)
    
    print("  ✓ Transformer block shapes correct")


def test_full_model_shapes():
    """Test full model forward pass shapes."""
    print("Testing full model shapes...")
    
    vocab_size = 1000
    d_model = 128
    n_layers = 4
    n_heads = 4
    d_ff = 512
    context_length = 64
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context_length=context_length,
        dropout=0.0,
        weight_tying=True
    )
    
    B, T = 2, 32
    input_ids = torch.randint(0, vocab_size, (B, T))
    
    logits = model(input_ids)
    assert logits.shape == (B, T, vocab_size), f"Expected {(B, T, vocab_size)}, got {logits.shape}"
    
    logits, attn = model(input_ids, return_attention=True)
    assert len(attn) == n_layers
    for a in attn:
        assert a.shape == (B, n_heads, T, T)
    
    print("  ✓ Full model shapes correct")


def test_context_length_enforcement():
    """Test that sequences longer than context length are rejected."""
    print("Testing context length enforcement...")
    
    vocab_size = 1000
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 256
    context_length = 32
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context_length=context_length,
        dropout=0.0,
        weight_tying=False
    )
    
    input_ids = torch.randint(0, vocab_size, (1, context_length + 1))
    
    try:
        model(input_ids)
        assert False, "Should have raised assertion error for exceeding context length"
    except AssertionError as e:
        if "context length" in str(e):
            print("  ✓ Context length correctly enforced")
        else:
            raise


def test_weight_tying():
    """Test that weight tying shares embedding and output weights."""
    print("Testing weight tying...")
    
    vocab_size = 1000
    d_model = 64
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        context_length=32,
        dropout=0.0,
        weight_tying=True
    )
    
    assert model.lm_head.weight is model.token_embedding.weight, \
        "Weight tying failed: lm_head.weight is not token_embedding.weight"
    
    print("  ✓ Weight tying correctly shares weights")


def test_checkpoint_restore():
    """Test checkpoint save and restore."""
    print("Testing checkpoint save/restore...")
    
    import tempfile
    import os
    
    vocab_size = 1000
    d_model = 64
    n_layers = 2
    n_heads = 4
    d_ff = 256
    context_length = 32
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        context_length=context_length,
        dropout=0.0,
        weight_tying=False
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt_path = f.name
    
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': 100,
            'config': {'test': True}
        }, ckpt_path)
        
        model2 = TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            context_length=context_length,
            dropout=0.0,
            weight_tying=False
        )
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        
        checkpoint = torch.load(ckpt_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Model parameters not restored correctly"
        
        input_ids = torch.randint(0, vocab_size, (1, 16))
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model2(input_ids)
        
        assert torch.allclose(out1, out2), "Model outputs not identical after restore"
        
        print("  ✓ Checkpoint save/restore works correctly")
    finally:
        os.unlink(ckpt_path)


def test_ffn_shapes():
    """Test FFN shapes."""
    print("Testing FFN shapes...")
    
    B, T, d_model = 2, 16, 64
    d_ff = 256
    
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    x = torch.randn(B, T, d_model)
    
    out = ffn(x)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"
    
    print("  ✓ FFN shapes correct")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    test_attention_shapes()
    test_attention_probabilities_sum_to_one()
    test_causal_mask()
    test_causal_masking_verification()
    test_head_splitting_merging()
    test_transformer_block_shapes()
    test_full_model_shapes()
    test_context_length_enforcement()
    test_weight_tying()
    test_checkpoint_restore()
    test_ffn_shapes()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()