import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention implemented from scratch.
    
    Computes:
        Q = X W_Q, K = X W_K, V = X W_V
        scores = Q @ K^T / sqrt(d_head)
        scores = scores + causal_mask
        attn_weights = softmax(scores)
        output = attn_weights @ V
        output = output W_O
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, context_length: int = 512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.context_length = context_length
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length) * float('-inf'), diagonal=1)
        )
        
        self.store_attention = False
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, T, d_model)
            return_attention: Whether to return attention weights
            
        Returns:
            output: (B, T, d_model)
            attention_weights: (B, n_heads, T, T) if return_attention else None
        """
        B, T, _ = x.shape
        
        assert T <= self.context_length, f"Sequence length {T} exceeds context length {self.context_length}"
        
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        mask = self.causal_mask[:T, :T]
        scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        if return_attention or self.store_attention:
            self.attention_weights = attn_weights.detach()
        
        out = torch.matmul(attn_weights, V)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_O(out)
        out = self.dropout(out)
        
        if return_attention:
            return out, self.attention_weights
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Structure:
        Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> Dropout
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Block.
    
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, context_length: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadCausalSelfAttention(d_model, n_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, T, d_model)
            return_attention: Whether to return attention weights
            
        Returns:
            output: (B, T, d_model)
            attention_weights: (B, n_heads, T, T) if return_attention else None
        """
        residual = x
        x = self.ln1(x)
        if return_attention:
            x, attn_weights = self.attention(x, return_attention=True)
        else:
            x = self.attention(x)
        x = residual + x
        
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        if return_attention:
            return x, attn_weights
        return x


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model.
    
    Architecture:
        TokenEmbedding + PositionalEmbedding -> Dropout
        -> TransformerBlock x N
        -> Final LayerNorm
        -> LM Head
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        context_length: int,
        dropout: float = 0.1,
        weight_tying: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.dropout = dropout
        self.weight_tying = weight_tying
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(context_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, context_length)
            for _ in range(n_layers)
        ])
        
        self.final_ln = nn.LayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        if weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Args:
            input_ids: (B, T)
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            logits: (B, T, vocab_size)
            all_attention: List of (B, n_heads, T, T) if return_attention else None
        """
        B, T = input_ids.shape
        
        assert T <= self.context_length, f"Sequence length {T} exceeds context length {self.context_length}"
        
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.positional_embedding(position_ids)
        x = token_emb + pos_emb
        x = self.embedding_dropout(x)
        
        all_attention = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                all_attention.append(attn)
            else:
                x = block(x)
        
        x = self.final_ln(x)
        logits = self.lm_head(x)
        
        if return_attention:
            return logits, all_attention
        return logits
    
    def get_num_params(self, non_embedding: bool = True):
        """Return number of trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.weight_tying:
            n_params -= self.token_embedding.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        return_attention: bool = False
    ):
        """
        Generate text using the model.
        
        Args:
            input_ids: (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeated tokens (> 1.0 penalizes repetitions)
            return_attention: Whether to return attention weights
            
        Returns:
            generated_ids: (B, T + max_new_tokens)
            all_attention: List of attention weights per layer per step if return_attention
        """
        self.eval()
        B, T = input_ids.shape
        generated = input_ids
        all_attention = [] if return_attention else None
        
        for _ in range(max_new_tokens):
            if generated.shape[1] > self.context_length:
                generated = generated[:, -self.context_length:]
            
            if return_attention:
                logits, attn = self.forward(generated, return_attention=True)
                all_attention.append(attn)
            else:
                logits = self.forward(generated)
            
            logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for token_id in set(generated[b].tolist()):
                        logits[b, token_id] /= repetition_penalty
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        if return_attention:
            return generated, all_attention
        return generated


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerLM(
        vocab_size=30980,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        context_length=512,
        dropout=0.1,
        weight_tying=True
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    B, T = 2, 128
    input_ids = torch.randint(0, 30980, (B, T))
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Test generation
    generated = model.generate(input_ids[:1, :10], max_new_tokens=20, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
    
    # Test with attention
    logits, attn = model(input_ids[:1], return_attention=True)
    print(f"Number of attention layers: {len(attn)}")
    print(f"Attention shape per layer: {attn[0].shape}")