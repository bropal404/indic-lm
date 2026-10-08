import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
        mask = torch.triu(torch.ones(context_len, context_len) * float('-inf'), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, d_model = x.size()
        
        q = self.W_q(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        
        scores = scores + self.mask[:T, :T]
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(B, T, d_model)
        
        out = self.resid_drop(self.W_o(context))
        return out, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, context_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, context_len, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.context_len = context_len
        self.pad_token_id = pad_token_id
        
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, context_len, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.size()
        assert T <= self.context_len, f"Sequence length {T} exceeds context length {self.context_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            if return_attention:
                attention_maps.append(attn_weights)
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_token_id)
            
        if return_attention:
            return logits, loss, attention_maps
        return logits, loss

    @torch.no_grad()
    def greedy_generate(self, idx, max_new_tokens):
        """
        Greedy decoding: at each step, pick the single most probable next token (argmax).
        Deterministic — temperature has no effect.
        Required by Section 2.3 of the spec alongside temperature sampling.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.context_len else idx[:, -self.context_len:]
            logits, _ = self(idx_cond)
            idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Temperature sampling: higher temperature -> more diverse / random output.
        temperature=0.5  -> sharper distribution (more conservative)
        temperature=1.0  -> unmodified distribution
        temperature=1.5  -> flatter distribution (more creative / random)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.context_len else idx[:, -self.context_len:]
            
            logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
