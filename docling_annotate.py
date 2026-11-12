"""Docling (il "teacher" della guida, sezione 17.1) -> blocchi con classe e bbox -> PDF annotato.

Formato che la guida richiede per usare i blocchi come annotazioni (lo stesso che si passa a layout.targets.assign):
  - box [x0, y0, x1, y1] in punti tipografici (1/72 di pollice), origine in alto a sinistra, y verso il basso
    (Docling restituisce l'origine in basso a sinistra: si converte con la altezza della pagina);
  - classe = indice in layout.config.CLASSES (0..10), da cui `assign` ricava classe e blocco di ogni nodo.

Uso:
  uv run python docling_annotate.py document.pdf
  uv run python docling_annotate.py document.pdf --out result.pdf --pages 1-5
  uv run python docling_annotate.py document.pdf --cache cache_docling/mio     (scrive anche le pseudo-etichette, .npz)
  uv run python docling_annotate.py document.pdf --ocr                          (PDF senza livello di testo)

Scrive il PDF annotato (default: document.docling.pdf) e, accanto, un .json con i blocchi, nello stesso formato di
annotate.py ("nodes" = nodi di testo di PyMuPDF che cadono nel blocco). Per una cartella intera: docling_batch.py.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pymupdf

from layout.config import CLASSES, KIND_TEXT
from layout.extract import extract_page
from layout.features import add_text_feats, save_raw
from layout.targets import assign

# Etichette di Docling (DocItemLabel) -> classi di DocLayNet. Quelle senza corrispondenza diventano "other".
LABEL_TO_CLASS = {
    "caption": "caption", "footnote": "footnote", "formula": "formula", "list_item": "list_item",
    "page_footer": "page_footer", "page_header": "page_header", "picture": "picture", "chart": "picture",
    "section_header": "section_header", "table": "table", "document_index": "table",
    "text": "text", "paragraph": "text", "code": "text", "reference": "text", "title": "title",
}


def parse_pages(spec, n_pages):
    """"1-5,8" -> [1, 2, 3, 4, 5, 8]; None -> tutte."""
    if not spec:
        return list(range(1, n_pages + 1))
    pages = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        pages += range(int(a), int(b or a) + 1)
    return [p for p in pages if 1 <= p <= n_pages]


def make_converter(ocr=False):
    """DocumentConverter di Docling (carica i modelli: va creato una volta sola e riusato)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def docling_blocks(dl_doc):
    """DoclingDocument -> {pagina: [blocchi]}, in ordine di lettura, con box in punti e origine in alto a sinistra."""
    from docling_core.types.doc import ContentLayer

    pages = {}
    layers = set(ContentLayer)                               # di default Docling salta intestazioni e piè di pagina
    for item, _level in dl_doc.iterate_items(included_content_layers=layers):
        label = getattr(item, "label", None)
        if label is None or not getattr(item, "prov", None):
            continue                                         # gruppi (liste, sezioni): non hanno una regione propria
        cls = LABEL_TO_CLASS.get(getattr(label, "value", str(label)), "other")
        if cls == "table":
            text = item.export_to_markdown(dl_doc)
        else:
            text = getattr(item, "text", "") or ""
        for prov in item.prov:
            height = dl_doc.pages[prov.page_no].size.height
            b = prov.bbox.to_top_left_origin(height)
            pages.setdefault(prov.page_no, []).append({
                "class": cls, "bbox": [float(b.l), float(b.t), float(b.r), float(b.b)], "text": " ".join(text.split()),
            })
    return pages


def annotate_pdf(src, converter, out, pages_spec=None, cache=None, verbose=True):
    """Annota un PDF con Docling: scrive out (.pdf annotato), out .json (blocchi) e, se cache, un .npz per pagina.
    Il .json è scritto per ultimo e in modo atomico: se esiste, il documento è completo.
    Ritorna un dizionario con le statistiche."""
    from annotate import draw_blocks                         # stessi colori e stesso disegno di annotate.py

    src, out = pathlib.Path(src), pathlib.Path(out)
    doc = pymupdf.open(src)
    numbers = parse_pages(pages_spec, doc.page_count)

    t0 = time.perf_counter()
    dl_doc = converter.convert(src, page_range=(min(numbers), max(numbers))).document
    t_docling = time.perf_counter() - t0
    by_page = docling_blocks(dl_doc)
    if verbose:
        print(f"Docling: {len(numbers)} pagine in {t_docling:.1f} s ({t_docling / len(numbers):.2f} s per pagina)")

    if cache:
        pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
    keep = pymupdf.open()                                    # il PDF annotato contiene solo le pagine richieste
    pages, covered, total = [], 0, 0
    for number in numbers:
        page = doc[number - 1]
        blocks = by_page.get(number, [])
        raw = add_text_feats(extract_page(page))             # nodi di testo PyMuPDF: serve a contare e alle etichette
        is_text = raw["kind"] == KIND_TEXT
        if blocks and is_text.any():
            ann = np.array([b["bbox"] for b in blocks], np.float32)
            cls = np.array([CLASSES.index(b["class"]) for b in blocks], np.int64)
            label, block = assign(raw["box"], ann, cls)
            covered += int((block[is_text] >= 0).sum())
            for k, b in enumerate(blocks):
                b["nodes"] = int(((block == k) & is_text).sum())
            if cache:
                raw["label"], raw["block"] = label, block
                save_raw(raw, str(pathlib.Path(cache) / f"{src.stem}_p{number:04d}.npz"))
        else:
            for b in blocks:
                b["nodes"] = 0
        total += int(is_text.sum())
        draw_blocks(page, blocks)
        keep.insert_pdf(doc, from_page=number - 1, to_page=number - 1)
        pages.append({"page": number, "source": "docling", "blocks": blocks})
        if verbose:
            print(f"Pagina {number}: {len(blocks)} blocchi, {int(is_text.sum())} nodi di testo")
            for k, b in enumerate(blocks[:6], start=1):
                print(f"   {k:2d} {b['class']:15s} {b['text'][:60]}")

    tmp_pdf = out.with_name(out.name + ".tmp")
    keep.save(tmp_pdf, garbage=3, deflate=True)
    os.replace(tmp_pdf, out)
    out_json = out.with_suffix(".json")
    tmp_json = out_json.with_name(out_json.name + ".tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    os.replace(tmp_json, out_json)

    counts = {}
    for p in pages:
        for b in p["blocks"]:
            counts[b["class"]] = counts.get(b["class"], 0) + 1
    return {"pages": len(numbers), "blocks": sum(counts.values()), "classes": counts,
            "covered": covered, "total": total, "t_docling": t_docling}


def main():
    sys.stdout.reconfigure(errors="replace")                 # la console di Windows non stampa ogni carattere
    parser = argparse.ArgumentParser(description="Annota un PDF con Docling.")
    parser.add_argument("pdf")
    parser.add_argument("--out", help="PDF annotato (default: <name>.docling.pdf)")
    parser.add_argument("--pages", help="per esempio 1-5,8 (default: tutte)")
    parser.add_argument("--ocr", action="store_true", help="attiva l'OCR di Docling (default: spento, il PDF ha il testo)")
    parser.add_argument("--cache", help="cartella in cui scrivere le pseudo-etichette (.npz, come build_cache.py)")
    args = parser.parse_args()

    src = pathlib.Path(args.pdf)
    out = pathlib.Path(args.out) if args.out else src.with_name(src.stem + ".docling.pdf")
    s = annotate_pdf(src, make_converter(args.ocr), out, args.pages, args.cache)

    print(f"\nScritto {out} (e {out.with_suffix('.json').name}): {s['blocks']} blocchi")
    print("Classi:", ", ".join(f"{c} {n}" for c, n in sorted(s["classes"].items(), key=lambda t: -t[1])))
    print(f"Nodi di testo dentro un blocco di Docling: {s['covered']}/{s['total']} "
          f"({s['covered'] / max(s['total'], 1) * 100:.1f}%) (controllo sulle coordinate: deve essere alto)")
    if args.cache:
        print(f"Pseudo-etichette in {args.cache}")


if __name__ == "__main__":
    main()
