"""Caricamento delle pagine per PyTorch."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import load_raw, make_example


class PageDataset(Dataset):
    def __init__(self, folder, cfg, train):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"Nessun file .npz in {folder}")
        self.cfg, self.train = cfg, train
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        return make_example(load_raw(self.files[k]), self.cfg, self.train, self.rng)


def collate(batch):
    """Unisce più pagine in un batch, riempiendo le più corte con padding."""
    b_size = len(batch)
    n = max(len(e["feats"]) for e in batch)
    p = max(max(len(e["pairs"]) for e in batch), 1)
    n_feats, n_tok, n_rel = batch[0]["feats"].shape[1], batch[0]["tok"].shape[1], batch[0]["rel"].shape[-1]
    out = {
        "feats": torch.zeros(b_size, n, n_feats),
        "tok": torch.zeros(b_size, n, n_tok, dtype=torch.long),
        "rel": torch.zeros(b_size, n, n, n_rel, dtype=torch.uint8),
        "mask": torch.zeros(b_size, n, dtype=torch.bool),
        "pairs": torch.zeros(b_size, p, 2, dtype=torch.long),
        "pair_rel": torch.zeros(b_size, p, n_rel, dtype=torch.long),
        "pair_mask": torch.zeros(b_size, p, dtype=torch.bool),
        "label": torch.full((b_size, n), -100, dtype=torch.long),    # -100 = "ignora nella loss"
        "link": torch.zeros(b_size, p),
    }
    for b, e in enumerate(batch):
        k, q = len(e["feats"]), len(e["pairs"])
        out["feats"][b, :k] = torch.from_numpy(e["feats"])
        out["tok"][b, :k] = torch.from_numpy(e["tok"])
        out["rel"][b, :k, :k] = torch.from_numpy(np.ascontiguousarray(e["rel"]))
        out["mask"][b, :k] = True
        out["pairs"][b, :q] = torch.from_numpy(e["pairs"])
        out["pair_rel"][b, :q] = torch.from_numpy(e["pair_rel"].astype(np.int64))
        out["pair_mask"][b, :q] = True
        if "label" in e:
            out["label"][b, :k] = torch.from_numpy(e["label"])
            out["link"][b, :q] = torch.from_numpy(e["link"])
    out["block"] = [e.get("block") for e in batch]           # lista di array: serve solo alle metriche
    return out


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
