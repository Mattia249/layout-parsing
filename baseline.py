"""Baseline senza rete neurale: classifica ogni nodo dalle sue feature e da quelle dei due nodi adiacenti.
Uso: python baseline.py D:/AI/cache"""
import glob
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from layout.config import CLASSES, Config
from layout.decode import f1_per_class
from layout.features import BASE_FEATS, load_raw, make_example


def page_xy(path, cfg):
    ex = make_example(load_raw(path), cfg)
    n, r = len(ex["feats"]), ex["rel"].shape[-1]
    prev = np.full((n, r), 12, np.uint8)                     # 12 = "nessuna differenza"
    nxt = prev.copy()
    if n > 1:
        step = ex["rel"][np.arange(n - 1), np.arange(1, n)]  # relazione tra ogni nodo e il successivo
        nxt[:-1] = step
        prev[1:] = step
    return np.concatenate([ex["feats"][:, :BASE_FEATS], prev, nxt], 1), ex["label"]


def load(folder, n_pages, cfg):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    order = np.random.default_rng(0).permutation(len(files))[:n_pages]
    xs, ys = zip(*(page_xy(files[i], cfg) for i in order))
    return np.concatenate(xs), np.concatenate(ys)


def main(cache_dir, n_train=3000, n_val=500):
    cfg = Config()
    x, y = load(os.path.join(cache_dir, "train"), n_train, cfg)
    xv, yv = load(os.path.join(cache_dir, "validation"), n_val, cfg)
    print(f"Nodi di training: {len(y)}, di validation: {len(yv)}")
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1)
    clf.fit(x, y)
    f1, support = f1_per_class(yv, clf.predict(xv), len(CLASSES))
    for name, value, count in zip(CLASSES, f1, support):
        print(f"{name:16s} F1 {value:.3f}   nodi {count}")
    print(f"Macro-F1 (sulle classi presenti): {f1[support > 0].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
