import math

import numpy as np
import torch
import torch.nn as nn

class CycleCheckTransformer(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		hidden_dim: int,
		num_heads: int,
		num_layers: int,
		mlp_dim: int,
		max_len: int,
		num_classes: int = 2,
		dropout: float = 0.1,
		pos_emb_type: str = "sinusoidal",
	):
		super().__init__()
		pos_emb_type = pos_emb_type.lower()
		if pos_emb_type not in {"sinusoidal", "simple-index", "learned"}:
			raise ValueError("pos_emb_type must be 'sinusoidal', 'simple-index', or 'learned'")
		self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
		self.pos_emb_type = pos_emb_type
		if pos_emb_type == "learned":
			self.pos_emb = nn.Embedding(max_len, hidden_dim)
		elif pos_emb_type == "simple-index":
			self.pos_linear = nn.Linear(1, hidden_dim)
		else:
			self.register_buffer("pos_buffer", self._build_sinusoidal(max_len, hidden_dim), persistent=False)
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
		self.max_len = max_len
		self.hidden_dim = hidden_dim

	@staticmethod
	def _build_sinusoidal(max_len: int, hidden_dim: int) -> torch.Tensor:
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
		pe = torch.zeros(max_len, hidden_dim)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		return pe

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		seq_len = input_ids.size(1)
		if seq_len > self.max_len:
			raise ValueError(f"Sequence length {seq_len} exceeds configured max_len {self.max_len}")
		positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
		if self.pos_emb_type == "learned":
			positional = self.pos_emb(positions)
		elif self.pos_emb_type == "simple-index":
			pos_values = positions.float() / max(1, self.max_len - 1)
			positional = self.pos_linear(pos_values.unsqueeze(-1))
		else:
			positional = self.pos_buffer[:seq_len].unsqueeze(0)
		h = self.token_emb(input_ids) + positional
		padding_mask = attention_mask == 0
		encoded = self.encoder(h, src_key_padding_mask=padding_mask)
		encoded = self.norm(encoded)
		# mask-aware average pooling
		mask = attention_mask.unsqueeze(-1)
		summed = torch.sum(encoded * mask, dim=1)
		denom = mask.sum(dim=1).clamp(min=1)
		pooled = summed / denom
		return self.classifier(self.dropout(pooled))

