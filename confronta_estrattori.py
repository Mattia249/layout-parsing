"""Stesso modello, stesse pagine di validation, due estrattori: PyMuPDF (quello del training) e LiteParse.
Misura le metriche di train.py contro le annotazioni di DocLayNet.
Uso: uv run confronta_estrattori.py runs/v1/best.pt [numero_pagine]"""
import glob
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import pymupdf
import torch

from infer import load_model
from layout import lite
from layout.config import CLASSES, KIND_TEXT
from layout.data import collate
from layout.extract import extract_page
from layout.features import add_text_feats, make_example
from layout.targets import assign, page_annotations
from train import evaluate

DATA = "datasets/DocLayNet-v1.2/data"


def pages(n):
    """Le prime n pagine di validation con testo, prese a passo regolare da tutti i file (tutte le categorie)."""
    files = sorted(glob.glob(f"{DATA}/validation-*.parquet"))
    per_file = -(-n // len(files))
    for path in files:
        taken = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["bboxes", "category_id", "metadata", "pdf"]):
            for row in batch.to_pylist():
                if taken < per_file:
                    taken += 1
                    yield row
            if taken >= per_file:
                break


def main(ckpt, n=500):
    pymupdf.TOOLS.mupdf_display_errors(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt, device)
    parser = lite.make_parser()
    examples = {"pymupdf": [], "liteparse": []}
    seconds = {"pymupdf": 0.0, "liteparse": 0.0}
    skipped = 0
    for row in pages(n):
        ann_cls = np.asarray(row["category_id"], np.int64) - 1
        raws = {}
        t = time.perf_counter()
        doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
        raws["pymupdf"] = extract_page(doc[0], with_graphics=False)
        seconds["pymupdf"] += time.perf_counter() - t
        t = time.perf_counter()
        lp_page = parser.parse(row["pdf"]).pages[0]
        doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
        raws["liteparse"] = lite.extract_page_liteparse(doc[0], lp_page, with_graphics=False)
        seconds["liteparse"] += time.perf_counter() - t
        if not all((r["kind"] == KIND_TEXT).any() for r in raws.values()):
            skipped += 1                                     # confronto solo sulle pagine con testo per entrambi
            continue
        for name, raw in raws.items():
            raw = add_text_feats(raw)
            ann = page_annotations(row, raw["page_w"], raw["page_h"])
            raw["label"], raw["block"] = assign(raw["box"], ann, ann_cls)
            examples[name].append((make_example(raw, cfg), (raw["block"] >= 0).mean()))

    print(f"Pagine confrontate: {len(examples['pymupdf'])} (escluse {skipped} senza testo per almeno un estrattore)")
    print(f"{'':10s} {'F1 nodi':>8s} {'F1 link':>8s} {'F1 blocchi':>10s} {'nodi/pag':>9s} {'copertura':>9s} {'ms/pag':>7s}")
    per_class = {}
    for name, items in examples.items():
        batches = [collate([ex for ex, _ in items[i:i + 8]]) for i in range(0, len(items), 8)]
        m = evaluate(model, batches, cfg, device)
        per_class[name] = m["per_class"]
        nodes = np.mean([len(ex["feats"]) for ex, _ in items])
        cover = np.mean([c for _, c in items])
        ms = 1000 * seconds[name] / max(len(items) + skipped, 1)
        print(f"{name:10s} {m['node_f1']:8.3f} {m['link_f1']:8.3f} {m['block_f1']:10.3f} {nodes:9.1f} {cover:9.3f} {ms:7.1f}")
    support = np.bincount(np.concatenate([ex["label"] for ex, _ in examples["pymupdf"]]), minlength=len(CLASSES))
    print(f"\n{'classe':15s} {'pymupdf':>8s} {'liteparse':>9s} {'nodi':>6s}")
    for c, name in enumerate(CLASSES):
        print(f"{name:15s} {per_class['pymupdf'][c]:8.3f} {per_class['liteparse'][c]:9.3f} {support[c]:6d}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500)
