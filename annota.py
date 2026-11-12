"""OCR -> modello di layout -> PDF annotato con i blocchi trovati.

Uso:
  uv run annota.py runs/v1/best.pt documento.pdf
  uv run annota.py runs/v1/best.pt documento.pdf --ocr auto --dpi 300 --out risultato.pdf

--ocr sempre  OCR su ogni pagina, ignorando il testo già presente nel PDF (default)
--ocr auto    OCR solo sulle pagine senza livello di testo; le altre usano il testo del PDF
--ocr mai     solo il testo del PDF, come infer.py

Scrive il PDF annotato (default: documento.annotato.pdf) e, accanto, un .json con i blocchi.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pymupdf
import torch

from infer import load_model, predict_page
from layout.config import CLASSES
from layout.extract import extract_page
from layout.ocr import extract_page_ocr, make_engine

COLORS = {                                                   # un colore per classe (RGB da 0 a 1)
    "caption": (0.90, 0.49, 0.13), "footnote": (0.55, 0.34, 0.29), "formula": (0.58, 0.40, 0.74),
    "list_item": (0.09, 0.75, 0.81), "page_footer": (0.50, 0.50, 0.50), "page_header": (0.50, 0.50, 0.50),
    "picture": (0.84, 0.15, 0.16), "section_header": (0.12, 0.47, 0.71), "table": (0.17, 0.63, 0.17),
    "text": (0.74, 0.74, 0.13), "title": (0.89, 0.12, 0.56), "other": (0.70, 0.70, 0.70),
}
assert set(COLORS) == set(CLASSES)


def draw_blocks(page, blocks):
    """Un rettangolo colorato per blocco, con numero e classe in alto a sinistra."""
    shape = page.new_shape()
    labels = []
    for k, b in enumerate(blocks, start=1):
        rect = pymupdf.Rect(b["bbox"]) + (-1, -1, 1, 1)
        color = COLORS[b["class"]]
        shape.draw_rect(rect)
        shape.finish(color=color, fill=color, width=0.8, fill_opacity=0.12, stroke_opacity=0.9)
        label = f"{k} {b['class']}"
        width = pymupdf.get_text_length(label, fontname="helv", fontsize=6) + 3
        top = rect.y0 - 7 if rect.y0 >= 7 else rect.y0     # sopra il box se c'è spazio, altrimenti dentro
        tag = pymupdf.Rect(rect.x0, top, rect.x0 + width, top + 7)
        shape.draw_rect(tag)
        shape.finish(color=None, fill=color, fill_opacity=0.85)
        labels.append((tag, label))
    shape.commit(overlay=True)
    for tag, label in labels:                                # il testo dopo i riquadri, così resta leggibile
        page.insert_text((tag.x0 + 1.5, tag.y1 - 1.6), label, fontname="helv", fontsize=6, color=(1, 1, 1))


def main():
    sys.stdout.reconfigure(errors="replace")                 # la console di Windows non stampa ogni carattere (es. α)
    parser =argparse.ArgumentParser(description="OCR, modello di layout e PDF annotato.")
    parser.add_argument("checkpoint", help="per esempio runs/v1/best.pt")
    parser.add_argument("pdf")
    parser.add_argument("--ocr", choices=["sempre", "auto", "mai"], default="sempre")
    parser.add_argument("--dpi", type=int, default=200, help="risoluzione a cui la pagina viene letta dall'OCR")
    parser.add_argument("--out", help="PDF annotato (default: <nome>.annotato.pdf)")
    args = parser.parse_args()

    src = pathlib.Path(args.pdf)
    out = pathlib.Path(args.out) if args.out else src.with_name(src.stem + ".annotato.pdf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.checkpoint, device)
    engine = make_engine(use_cuda=device == "cuda") if args.ocr != "mai" else None

    def native(page):
        return extract_page(page, with_graphics=cfg.use_graphics)

    def ocr(page):
        return extract_page_ocr(page, engine, dpi=args.dpi, with_graphics=cfg.use_graphics)

    doc = pymupdf.open(src)
    pages, times = [], np.zeros(3)
    for number, page in enumerate(doc, start=1):
        use_ocr = args.ocr == "sempre" or (args.ocr == "auto" and not page.get_text("text").strip())
        blocks, t = predict_page(page, model, cfg, device, ocr if use_ocr else native)
        times += t
        draw_blocks(page, blocks)
        pages.append({"page": number, "source": "ocr" if use_ocr else "pdf", "blocks": blocks})
        source = "OCR" if use_ocr else "testo del PDF"
        print(f"Pagina {number} ({source}): {sum(b['nodes'] for b in blocks)} nodi, {len(blocks)} blocchi, "
              f"{t[0] * 1000:.0f} ms di estrazione, {(t[1] + t[2]) * 1000:.0f} ms di feature e modello")
        for k, b in enumerate(blocks[:8], start=1):
            print(f"   {k:2d} {b['class']:15s} {b['text'][:70]}")

    doc.save(out, garbage=3, deflate=True)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    per_page = 1000 * times / max(len(pages), 1)
    print(f"Scritto {out} (e {out.with_suffix('.json').name}).")
    print(f"Tempo medio per pagina: estrazione/OCR {per_page[0]:.0f} ms, feature {per_page[1]:.0f} ms, "
          f"modello {per_page[2]:.0f} ms")


if __name__ == "__main__":
    main()
