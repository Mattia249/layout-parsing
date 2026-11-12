"""Applica il modello a un PDF e scrive i blocchi trovati in un file JSON.
Uso: python infer.py runs/v1/best.pt documento.pdf"""
import json
import sys
import time

import numpy as np
import pymupdf
import torch

from layout.config import CLASSES, Config
from layout.data import collate, to_device
from layout.decode import block_classes, find_blocks
from layout.extract import extract_page
from layout.features import add_text_feats, make_example
from layout.model import LayoutModel, run


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    cfg = Config(**ck["cfg"])
    model = LayoutModel(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg


@torch.no_grad()
def predict_page(page, model, cfg, device, extract=None):
    """Ritorna i blocchi della pagina e i tempi (estrazione, feature, modello) in secondi.
    extract: funzione pagina -> nodi; di default il livello di testo del PDF (annotate.py la sostituisce con l'OCR)."""
    t0 = time.perf_counter()
    if extract is None:
        extract = lambda p: extract_page(p, with_graphics=cfg.use_graphics)
    raw = add_text_feats(extract(page))
    t1 = time.perf_counter()
    ex = make_example(raw, cfg, cap=10 ** 9)                 # in uso reale nessun limite al numero di nodi
    n = len(ex["feats"])
    if n == 0:
        return [], (t1 - t0, 0.0, 0.0)
    batch = to_device(collate([ex]), device)
    t2 = time.perf_counter()
    cls_logits, link_logits = run(model, batch)
    logits = cls_logits[0].cpu().numpy()
    prob = torch.sigmoid(link_logits[0]).cpu().numpy()[:len(ex["pairs"])]
    t3 = time.perf_counter()

    block = find_blocks(n, ex["pairs"], prob, logits.argmax(1), cfg.link_threshold)
    cls = block_classes(block, logits)
    out = []
    for b in range(int(block.max()) + 1):
        idx = np.flatnonzero(block == b)                     # i nodi sono già in ordine di lettura
        box = ex["box"][idx]
        out.append({
            "class": CLASSES[int(cls[b])],
            "bbox": [float(box[:, 0].min()), float(box[:, 1].min()), float(box[:, 2].max()), float(box[:, 3].max())],
            "text": " ".join(ex["texts"][i] for i in idx),
            "nodes": len(idx),
        })
    out.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))     # dall'alto in basso: non è un vero ordine di lettura
    return out, (t1 - t0, t2 - t1, t3 - t2)


def main(ckpt, pdf_path):
    sys.stdout.reconfigure(errors="replace")                 # la console di Windows non stampa ogni carattere (es. α)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt, device)
    doc = pymupdf.open(pdf_path)
    pages, times = [], np.zeros(3)
    for number, page in enumerate(doc, start=1):
        blocks, t = predict_page(page, model, cfg, device)
        times += t
        pages.append({"page": number, "blocks": blocks})
        print(f"Pagina {number}: {sum(b['nodes'] for b in blocks)} nodi, {len(blocks)} blocchi")
        for b in blocks[:8]:
            print(f"   {b['class']:15s} {b['text'][:70]}")
    with open(pdf_path + ".layout.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    per_page = 1000 * times / max(len(pages), 1)
    print(f"Tempo medio per pagina: estrazione {per_page[0]:.1f} ms, feature {per_page[1]:.1f} ms, "
          f"modello {per_page[2]:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
