"""Small GPT in plain PyTorch — modernised, Llama-style. Yours to modify;
evaluate.py rebuilds the exact model from the checkpoint's saved config, and
the parameter cap (2,000,000) still holds.

Techniques (all pure-PyTorch, no pretrained anything). Cited in RUNLOG.md:
  * RoPE rotary positions   (Su et al., RoFormer, 2021) — no learned params,
    replaces the absolute position table.
  * RMSNorm                 (Zhang & Sennrich, 2019) — used in Llama.
  * SwiGLU MLP              (Shazeer, GLU Variants, 2020) — used in Llama/PaLM.
  * weight tying + GPT-2 scaled init (baseline had neither).
Every architectural choice is a Config flag, so old and new variants both
reconstruct from a saved checkpoint.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True
    init_std = 0.02
    pos_type = "rope"      # "rope" | "learned"
    norm_type = "rms"      # "rms"  | "ln"
    mlp_type = "swiglu"    # "swiglu" | "gelu"
    mlp_hidden = 384       # SwiGLU hidden (≈ 8/3 * n_embd, rounded)
    rope_base = 10000.0
    qk_norm = False        # RMS-normalise q,k per head (modded-nanoGPT)
    logit_softcap = 0.0    # 0 = off; else c*tanh(logits/c)  (Gemma-2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm_type == "rms" else nn.LayerNorm(cfg.n_embd)


def build_rope(block, hd, base, device=None):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    t = torch.arange(block, dtype=torch.float32)
    freqs = torch.outer(t, inv)                 # (block, hd/2)
    return torch.cos(freqs), torch.sin(freqs)   # each (block, hd/2)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd) ; rotate pairs (even, odd)
    B, nh, T, hd = x.shape
    x = x.view(B, nh, T, hd // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[:T].view(1, 1, T, hd // 2)
    sin = sin[:T].view(1, 1, T, hd // 2)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).view(B, nh, T, hd)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.use_rope = cfg.pos_type == "rope"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.qk_norm = getattr(cfg, "qk_norm", False)
        if self.qk_norm:
            hd = cfg.n_embd // cfg.n_head
            self.q_scale = nn.Parameter(torch.ones(hd))
            self.k_scale = nn.Parameter(torch.ones(hd))
        if self.use_rope:
            cos, sin = build_rope(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        if self.qk_norm:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * self.q_scale
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * self.k_scale
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.mlp_hidden
        self.w_gate = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_down = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GeluMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.mlp_type == "swiglu" else GeluMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_learned_pos = cfg.pos_type == "learned"
        if self.use_learned_pos:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self._init_std = getattr(cfg, "init_std", 0.02)
        self.apply(self._init)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight") \
               or name.endswith("mlp.2.weight"):
                with torch.no_grad():
                    p.mul_(scale)

        if getattr(cfg, "tie_weights", False):
            self.head.weight = self.tok_emb.weight

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self._init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.use_learned_pos:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        cap = getattr(self.cfg, "logit_softcap", 0.0)
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
