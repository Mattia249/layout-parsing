"""La rete: nodi -> vettori -> transformer con bias geometrico -> classe dei nodi e link tra coppie."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASSES, REL_BUCKETS, REL_FEATS, VOCAB
from .features import NODE_FEATS


class Block(nn.Module):
    """Un layer del transformer (pre-LN): attenzione tra i nodi, poi rielaborazione di ogni nodo."""

    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):                              # x: [B, N, d]   bias: [B, H, N, N]
        b, n, d = x.shape
        d_head = d // self.heads
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # ciascuno [B, H, N, d_head]
        att = q @ k.transpose(-2, -1) / math.sqrt(d_head) + bias
        att = self.drop(att.softmax(-1))                     # quanto ogni nodo guarda ciascun altro
        y = (att @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.norm2(x)))


class LayoutModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.heads = cfg.n_heads
        self.feat_mlp = nn.Sequential(nn.Linear(NODE_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.tok_emb = nn.Embedding(VOCAB, cfg.d_tok, padding_idx=0)
        self.tok_proj = nn.Linear(cfg.d_tok, d)
        self.in_norm = nn.LayerNorm(d)
        self.rel_bias = nn.Embedding(REL_FEATS * REL_BUCKETS, cfg.n_heads)
        self.layers = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.cls_head = nn.Linear(d, len(CLASSES))
        self.pair_emb = nn.Embedding(REL_FEATS * REL_BUCKETS, cfg.d_pair)
        self.link_head = nn.Sequential(nn.Linear(4 * d + cfg.d_pair, d), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        # Ogni feature di coppia ha i suoi 25 bucket: sommando questi scostamenti stanno tutte in una sola tabella.
        self.register_buffer("offsets", torch.arange(REL_FEATS) * REL_BUCKETS, persistent=False)
        nn.init.zeros_(self.rel_bias.weight)                 # all'inizio la geometria non influenza l'attenzione
        nn.init.normal_(self.pair_emb.weight, std=0.1)

    def forward(self, feats, tok, rel, mask, pairs, pair_rel):
        # 1) da nodo a vettore
        n_tok = (tok != 0).sum(-1, keepdim=True).clamp(min=1)
        text = self.tok_emb(tok).sum(2) / n_tok              # media degli embedding degli id non vuoti
        x = self.in_norm(self.feat_mlp(feats) + self.tok_proj(text))            # [B, N, d]

        # 2) bias geometrico, uguale per tutti i layer
        bias = torch.zeros(rel.shape[0], rel.shape[1], rel.shape[2], self.heads, device=feats.device)
        for f in range(rel.shape[-1]):
            bias = bias + self.rel_bias(rel[..., f].long() + self.offsets[f])
        bias = bias.permute(0, 3, 1, 2)                                          # [B, H, N, N]
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))          # nessuno guarda il padding

        # 3) sei layer
        for layer in self.layers:
            x = layer(x, bias)
        h = self.out_norm(x)

        # 4) teste
        cls_logits = self.cls_head(h)                                            # [B, N, 12]
        d = h.shape[-1]
        hi = h.gather(1, pairs[..., 0:1].expand(-1, -1, d))                      # [B, P, d]
        hj = h.gather(1, pairs[..., 1:2].expand(-1, -1, d))
        pe = self.pair_emb(pair_rel + self.offsets).sum(2)                       # [B, P, d_pair]
        link_logits = self.link_head(torch.cat([hi, hj, (hi - hj).abs(), hi * hj, pe], -1)).squeeze(-1)
        return cls_logits, link_logits                                           # [B, N, 12], [B, P]


def run(model, batch):
    return model(batch["feats"], batch["tok"], batch["rel"], batch["mask"], batch["pairs"], batch["pair_rel"])


def compute_losses(model, batch, cfg):
    cls_logits, link_logits = run(model, batch)
    loss_cls = F.cross_entropy(cls_logits.flatten(0, 1), batch["label"].flatten(),
                               ignore_index=-100, label_smoothing=cfg.label_smoothing)
    keep = batch["pair_mask"]
    if keep.any():
        loss_link = F.binary_cross_entropy_with_logits(link_logits[keep], batch["link"][keep])
    else:
        loss_link = link_logits.sum() * 0.0
    return loss_cls + cfg.link_weight * loss_link, loss_cls.item(), loss_link.item()
