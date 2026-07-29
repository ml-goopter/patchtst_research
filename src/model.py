"""PatchTST (PLAN.md section 12).

Channel-independent patched transformer: each input channel is patched and encoded
by a shared encoder, then all channel representations are flattened into one head.
Optional cross-channel attention and RevIN, both used as ablation switches
(PLAN.md section 21).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalization over the time axis, per channel."""

    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, C]
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x


class PatchTST(nn.Module):
    def __init__(self, n_channels: int, context_len: int, patch_len: int, stride: int,
                 n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 revin: bool = True, channel_attention: bool = False, n_outputs: int = 1,
                 head: str = "flatten"):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (context_len - patch_len) // stride + 1
        self.head_type = head

        self.revin = RevIN(n_channels) if revin else None

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.channel_attention = None
        if channel_attention:
            self.channel_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
            self.ca_norm = nn.LayerNorm(d_model)

        feat_dim = n_channels * (self.n_patches * d_model if head == "flatten" else d_model)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(feat_dim, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, C]
        B = x.shape[0]
        if self.revin is not None:
            x = self.revin(x)

        x = x.permute(0, 2, 1)                                    # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, P]
        x = x.reshape(B * self.n_channels, self.n_patches, self.patch_len)

        z = self.drop(self.patch_embed(x) + self.pos)
        z = self.norm(self.encoder(z))                            # [B*C, N, d]

        if self.channel_attention is not None:
            # let channels exchange information at each patch position
            d = z.shape[-1]
            zc = z.view(B, self.n_channels, self.n_patches, d).permute(0, 2, 1, 3)
            zc = zc.reshape(B * self.n_patches, self.n_channels, d)
            att, _ = self.channel_attention(zc, zc, zc, need_weights=False)
            zc = self.ca_norm(zc + att)
            z = zc.view(B, self.n_patches, self.n_channels, d).permute(0, 2, 1, 3)
            z = z.reshape(B * self.n_channels, self.n_patches, d)

        if self.head_type == "mean":
            z = z.mean(dim=1)                                     # [B*C, d]
        z = z.reshape(B, -1)
        return self.head(z)


def build(n_channels: int, cfg: dict, **overrides) -> PatchTST:
    kw = dict(
        n_channels=n_channels,
        context_len=cfg["context_len"], patch_len=cfg["patch_len"], stride=cfg["stride"],
        n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"], revin=cfg["revin"],
    )
    kw.update(overrides)
    return PatchTST(**kw)
