# Tensor Shape Documentation

This document explains the tensor shapes through the forward pass of the TransformerLM model.

## Model Configuration (Example)
- **Vocabulary Size (V)**: 30,980 (Hindi) / 27,998 (Nepali)
- **Model Dimension (d_model)**: 384
- **Number of Layers (N)**: 6
- **Number of Heads (h)**: 6
- **Feed-Forward Dimension (d_ff)**: 1,536
- **Context Length (T_max)**: 256
- **Head Dimension (d_head)**: d_model / h = 384 / 6 = 64

---

## Forward Pass Tensor Shapes

### 1. Input
```
input_ids: (B, T)
```
- **B**: Batch size (e.g., 8)
- **T**: Sequence length (≤ context_length, e.g., 256)

### 2. Token Embedding
```
token_embedding = nn.Embedding(V, d_model)
token_emb = token_embedding(input_ids)
# token_emb: (B, T, d_model) = (8, 256, 384)
```

### 3. Positional Embedding
```
position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
# position_ids: (B, T) = (8, 256)

positional_embedding = nn.Embedding(context_length, d_model)
pos_emb = positional_embedding(position_ids)
# pos_emb: (B, T, d_model) = (8, 256, 384)
```

### 4. Embedding Sum + Dropout
```
x = token_emb + pos_emb
# x: (B, T, d_model) = (8, 256, 384)

x = embedding_dropout(x)
# x: (B, T, d_model) = (8, 256, 384)
```

---

### 5. Transformer Blocks (× N layers)

For each of the 6 layers:

#### 5.1 First LayerNorm (Pre-norm)
```
x = ln1(x)
# x: (B, T, d_model) = (8, 256, 384)
```

#### 5.2 Multi-Head Causal Self-Attention

**Q, K, V Projections:**
```
Q = W_Q(x)  # W_Q: (d_model, d_model) = (384, 384)
K = W_K(x)  # W_K: (384, 384)
V = W_V(x)  # W_V: (384, 384)

# Q, K, V: (B, T, d_model) = (8, 256, 384)
```

**Split Heads:**
```
Q = Q.view(B, T, h, d_head).transpose(1, 2)
K = K.view(B, T, h, d_head).transpose(1, 2)
V = V.view(B, T, h, d_head).transpose(1, 2)

# Q, K, V: (B, h, T, d_head) = (8, 6, 256, 64)
```

**Attention Scores:**
```
scores = Q @ K.transpose(-2, -1) / sqrt(d_head)
# scores: (B, h, T, T) = (8, 6, 256, 256)
```

**Causal Mask:**
```
mask = torch.triu(torch.ones(T, T) * -inf, diagonal=1)
# mask: (T, T) = (256, 256)
# Broadcast to (B, h, T, T)

scores = scores + mask
# scores: (B, h, T, T) = (8, 6, 256, 256)
```

**Attention Weights (Softmax):**
```
attn_weights = softmax(scores, dim=-1)
attn_weights = attn_dropout(attn_weights)

# attn_weights: (B, h, T, T) = (8, 6, 256, 256)
# Each row sums to 1.0 over key positions
```

**Attention Output:**
```
attn_out = attn_weights @ V
# attn_out: (B, h, T, d_head) = (8, 6, 256, 64)
```

**Merge Heads:**
```
attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_model)
# attn_out: (B, T, d_model) = (8, 256, 384)
```

**Output Projection:**
```
attn_out = W_O(attn_out)  # W_O: (d_model, d_model) = (384, 384)
attn_out = dropout(attn_out)

# attn_out: (B, T, d_model) = (8, 256, 384)
```

#### 5.3 First Residual Connection
```
x = x + attn_out
# x: (B, T, d_model) = (8, 256, 384)
```

#### 5.4 Second LayerNorm (Pre-norm)
```
x = ln2(x)
# x: (B, T, d_model) = (8, 256, 384)
```

#### 5.5 Feed-Forward Network
```
x = fc1(x)        # fc1: (d_model, d_ff) = (384, 1536)
# x: (B, T, d_ff) = (8, 256, 1536)

x = GELU(x)
# x: (B, T, d_ff) = (8, 256, 1536)

x = dropout(x)
# x: (B, T, d_ff) = (8, 256, 1536)

x = fc2(x)        # fc2: (d_ff, d_model) = (1536, 384)
# x: (B, T, d_model) = (8, 256, 384)

x = dropout(x)
# x: (B, T, d_model) = (8, 256, 384)
```

#### 5.6 Second Residual Connection
```
x = x + ffn_out
# x: (B, T, d_model) = (8, 256, 384)
```

---

### 6. Final LayerNorm
```
x = final_ln(x)
# x: (B, T, d_model) = (8, 256, 384)
```

---

### 7. LM Head (Output Projection)
```
logits = lm_head(x)  # lm_head: (d_model, V) = (384, 30980)
# With weight tying: lm_head.weight = token_embedding.weight

# logits: (B, T, V) = (8, 256, 30980)
```

---

## Causal Language Modeling Objective

For training, we shift targets:
```
input:  x_1, x_2, ..., x_T
target: x_2, x_3, ..., x_{T+1}

logits[:, :-1, :]  # (B, T-1, V)
targets[:, 1:]     # (B, T-1)

loss = cross_entropy(logits.view(-1, V), targets.view(-1))
```

---

## What Would Break If Each Component Were Removed

| Component | What Breaks |
|-----------|-------------|
| Token Embedding | No way to convert discrete tokens to continuous vectors |
| Positional Embedding | Model loses all positional information → permutation invariant |
| Embedding Dropout | Overfitting, no regularization at input |
| LayerNorm (Pre-norm) | Training instability, gradient explosion in deep networks |
| Multi-Head Attention | No token-to-token communication, no context mixing |
| Causal Mask | Future token leakage → invalid LM training |
| Q/K/V Projections | No learned attention patterns |
| Head Splitting | Single attention pattern, no multi-head diversity |
| Scaling (1/√d_head) | Softmax saturation, vanishing gradients |
| Attention Dropout | Overfitting in attention |
| Output Projection (W_O) | Cannot mix head outputs |
| Residual Connections | Vanishing gradients, cannot train deep networks |
| Feed-Forward Network | No position-wise non-linear transformation |
| GELU Activation | No non-linearity in FFN |
| FFN Dropout | Overfitting in FFN |
| Final LayerNorm | Unstable output distributions |
| LM Head | Cannot predict vocabulary logits |
| Weight Tying | ~12M extra parameters, worse generalization |

---

## Attention Weight Storage (Evaluation Only)

During evaluation with `return_attention=True`:
```
all_attention = []  # List of length N (6)
for each layer:
    attn_weights: (B, h, T, T) = (1, 6, 256, 256)
    all_attention.append(attn_weights)
```

**Memory Note**: Not stored during training to save memory (would be ~6 × 8 × 6 × 256 × 256 × 4 bytes ≈ 75 MB per batch).

---

## Generation Shapes

During autoregressive generation:
```
Step 0: input_ids = (B, T_prompt)
Step 1: logits = (B, T_prompt, V) → next_token = (B, 1)
Step 2: input_ids = cat([input_ids, next_token]) = (B, T_prompt + 1)
...
Step k: input_ids = (B, T_prompt + k)
```

At each step, only the last position's logits are used for sampling.