import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sin(nn.Module):
    def __init__(self, dim, w=10.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalMLPFilter(nn.Module):
    """
    Generates an implicit long convolution filter from positions.
    A lightweight Hyena-style filter generator.
    """
    def __init__(self, seq_len, d_model, order=2, hidden=64, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            Sin(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            Sin(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model)
        )

    def forward(self, L, device=None):
        device = device or next(self.parameters()).device
        t = torch.linspace(0, 1, L, device=device).unsqueeze(-1)  # [L, 1]
        h = self.net(t)  # [L, D]
        return h


class FFTLongConv(nn.Module):
    """
    Depthwise long convolution using FFT.
    Input:  [B, L, D]
    Output: [B, L, D]
    """
    def __init__(self, d_model, seq_len, filter_hidden=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.filter_fn = PositionalMLPFilter(
            seq_len=seq_len,
            d_model=d_model,
            hidden=filter_hidden,
            dropout=dropout,
        )

    def forward(self, x):
        B, L, D = x.shape
        assert D == self.d_model, f"Expected D={self.d_model}, got {D}"

        filt = self.filter_fn(L, device=x.device)  # [L, D]

        # FFT conv along length dimension independently per channel
        fft_size = 2 * L
        x_f = torch.fft.rfft(x.transpose(1, 2), n=fft_size)         # [B, D, F]
        h_f = torch.fft.rfft(filt.transpose(0, 1), n=fft_size)      # [D, F]

        y_f = x_f * h_f.unsqueeze(0)                                # [B, D, F]
        y = torch.fft.irfft(y_f, n=fft_size)[..., :L]               # [B, D, L]

        return y.transpose(1, 2)                                    # [B, L, D]


class HyenaBlock(nn.Module):
    """
    Small Hyena-inspired block:
      - input projection
      - gated multiplicative pathway
      - long convolution
      - output projection
      - residual connection
    """
    def __init__(
        self,
        d_model,
        seq_len,
        expansion=2,
        filter_hidden=64,
        dropout=0.1,
    ):
        super().__init__()
        inner_dim = d_model * expansion

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.long_conv = FFTLongConv(
            d_model=inner_dim,
            seq_len=seq_len,
            filter_hidden=filter_hidden,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        u, v = self.in_proj(x).chunk(2, dim=-1)   # [B, L, inner_dim] each
        u = F.gelu(u)
        v = torch.sigmoid(v)

        h = self.long_conv(u)
        x = h * v
        x = self.out_proj(x)
        x = self.dropout(x)

        return x + residual


class HyenaClassificationHead(nn.Module):
    """
    Hyena-based classification head for protein LLMs.

    Expected input:
        x: [B, L, D_llm]

    Optional:
        mask: [B, L] with 1 for valid residues, 0 for padding
    """
    def __init__(
        self,
        llm_dim,
        num_classes,
        seq_len,
        d_model=256,
        n_layers=2,
        expansion=2,
        filter_hidden=64,
        dropout=0.1,
        pool="mean",
        use_cls=False,
    ):
        super().__init__()

        assert pool in {"mean", "max", "first", "attn"}

        self.pool = pool
        self.use_cls = use_cls

        self.input_proj = nn.Linear(llm_dim, d_model)

        self.blocks = nn.ModuleList([
            HyenaBlock(
                d_model=d_model,
                seq_len=seq_len,
                expansion=expansion,
                filter_hidden=filter_hidden,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        if pool == "attn":
            self.attn_pool = nn.Linear(d_model, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def masked_mean_pool(self, x, mask):
        mask = mask.unsqueeze(-1).float()                  # [B, L, 1]
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

    def masked_max_pool(self, x, mask):
        mask = mask.unsqueeze(-1).bool()
        x = x.masked_fill(~mask, float("-inf"))
        return x.max(dim=1).values

    def attention_pool(self, x, mask=None):
        scores = self.attn_pool(x).squeeze(-1)             # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bl,bld->bd", attn, x)

    def forward(self, x, mask=None):
        """
        x:    [B, L, D_llm]
        mask: [B, L] optional
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        if self.use_cls or self.pool == "first":
            pooled = x[:, 0]
        elif self.pool == "mean":
            pooled = self.masked_mean_pool(x, mask) if mask is not None else x.mean(dim=1)
        elif self.pool == "max":
            pooled = self.masked_max_pool(x, mask) if mask is not None else x.max(dim=1).values
        elif self.pool == "attn":
            pooled = self.attention_pool(x, mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")

        logits = self.classifier(pooled)
        return logits