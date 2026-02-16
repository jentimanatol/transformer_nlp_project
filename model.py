
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_padding_mask(token_ids, pad_id=0):
    return (token_ids == pad_id).unsqueeze(1).unsqueeze(2)

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, training=True):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    out = attn @ v
    return out, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        out, attn = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, training=self.training
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.proj(out)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.drop1(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop2(ffn_out)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_id=0,
                 d_model=128, n_heads=4, num_layers=2,
                 d_ff=512, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids):
        pad_mask = make_padding_mask(token_ids, pad_id=self.pad_id)
        x = self.embedding(token_ids)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=pad_mask)
        cls = self.drop(x[:, 0, :])
        return self.classifier(cls)
