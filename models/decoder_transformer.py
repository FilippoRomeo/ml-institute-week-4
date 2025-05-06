# models/decoder_transformer.py
import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention (masked)
        _tgt = self.norm1(tgt + self._self_attention(tgt, tgt_mask))
        # Cross-attention (encoder memory)
        _tgt = self.norm2(_tgt + self._cross_attention(_tgt, memory, memory_mask))
        # Feedforward
        _tgt = self.norm3(_tgt + self._feed_forward(_tgt))
        return _tgt

    def _self_attention(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        return self.dropout(attn_output)

    def _cross_attention(self, x, memory, mask):
        attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=mask)
        return self.dropout(attn_output)

    def _feed_forward(self, x):
        return self.dropout(self.linear2(torch.relu(self.linear1(x))))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_input, memory, tgt_mask=None, memory_mask=None):
        B, T = tgt_input.shape
        positions = torch.arange(0, T).unsqueeze(0).expand(B, T).to(tgt_input.device)

        x = self.token_embedding(tgt_input) + self.pos_embedding(positions)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [T, B, D] for nn.MultiheadAttention

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = x.transpose(0, 1)  # [B, T, D]
        logits = self.fc_out(x)
        return logits

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if __name__ == "__main__":
    vocab_size = 8000
    max_len = 30
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6
    batch_size = 2

    # Dummy caption input (token IDs) and dummy image memory
    tgt_input = torch.randint(0, vocab_size, (batch_size, max_len))          # [B, T]
    memory = torch.randn(batch_size, 196, embed_dim)                         # [B, S, D]
    memory = memory.transpose(0, 1)                                          # [S, B, D] for attention

    # Create model and mask
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers
    )

    tgt_mask = decoder.generate_square_subsequent_mask(max_len).to(tgt_input.device)

    # Forward pass
    logits = decoder(tgt_input, memory, tgt_mask=tgt_mask)

    print("Logits shape:", logits.shape)  # [B, T, vocab_size]
