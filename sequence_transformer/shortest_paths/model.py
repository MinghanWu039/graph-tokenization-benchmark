"""Transformer model tailored for the shortest-path prediction task."""

import math

import torch
import torch.nn as nn


class ShortestPathTransformer(nn.Module):
    """
    Transformer encoder that predicts discrete shortest-path distance classes.

    Positional encoding can be either sinusoidal (default) or a learned linear projection
    of normalized index positions ("simple-index").
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        max_len: int,
        dropout: float = 0.1,
        pos_emb_type: str = "sinusoidal",
        num_classes: int = 16,
    ) -> None:
        super().__init__()
        pos_emb_type = pos_emb_type.lower()
        if pos_emb_type not in {"sinusoidal", "simple-index"}:
            raise ValueError("pos_emb_type must be 'sinusoidal' or 'simple-index'")

        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb_type = pos_emb_type
        self.max_len = max_len

        if pos_emb_type == "sinusoidal":
            self.register_buffer("pos_buffer", self._build_sinusoidal(max_len, hidden_dim), persistent=False)
            self.pos_linear = None
        else:
            self.pos_buffer = None
            self.pos_linear = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _build_sinusoidal(max_len: int, hidden_dim: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_len {self.max_len}")

        token_embeddings = self.token_emb(input_ids)

        if self.pos_emb_type == "sinusoidal":
            positional = self.pos_buffer[:seq_len].unsqueeze(0)
        else:
            positions = torch.arange(seq_len, device=input_ids.device).float()
            max_range = max(1, self.max_len - 1)
            normalized = (positions / max_range).unsqueeze(0).unsqueeze(-1)
            positional = self.pos_linear(normalized)

        h = token_embeddings + positional

        padding_mask = attention_mask == 0
        encoded = self.encoder(h, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)

        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(encoded * mask, dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom

        return self.classifier(self.dropout(pooled))
