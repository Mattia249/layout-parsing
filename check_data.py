"""Primo contatto con il dataset: colonne, convenzione dei box, statistiche, immagini di controllo.
Uso: python check_data.py D:/AI/datasets/DocLayNet-v1.2/data"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pymupdf

from layout.config import KIND_TEXT
from layout.extract import extract_page
from layout.targets import assign, page_annotations


def main(data_dir, n_pages=200, n_images=5):
    path = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))[0]
    f = pq.ParquetFile(path)
    print("Colonne:", f.schema_arrow.names)
    # le quattro ipotesi su come sono scritti i box: formato x spazio
    cover = {(fmt, space): [] for fmt in ("xywh", "xyxy") for space in ("1025", "punti")}
    n_nodes, ids, samples, no_text, done = [], [], [], 0, 0
    for batch in f.iter_batches(batch_size=16, columns=["bboxes", "category_id", "metadata", "pdf"]):
        for row in batch.to_pylist():
            if done == n_pages:
                break
            done += 1
            ids += list(row["category_id"])
            try:
                doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
                raw = extract_page(doc[0])
                doc.close()
            except Exception as error:
                print(f"Pagina {done} illeggibile: {error}")
                continue
            text_box = raw["box"][raw["kind"] == KIND_TEXT]
            if len(text_box) == 0:
                no_text += 1
                continue
            n_nodes.append(len(text_box))
            for fmt, space in cover:
                ann = page_annotations(row, raw["page_w"], raw["page_h"], fmt, space)
                _, block = assign(text_box, ann, np.zeros(len(ann), np.int64))
                cover[fmt, space].append((block >= 0).mean())
            if len(samples) < n_images:
                samples.append((done, row, raw, text_box))
        if done == n_pages:
            break

    print(f"Pagine esaminate: {done}, senza testo: {no_text}")
    for (fmt, space), values in cover.items():
        print(f"Copertura con box {fmt} nello spazio {space}: {np.mean(values):.3f}")
    fmt, space = max(cover, key=lambda key: np.mean(cover[key]))
    print(f"Ipotesi migliore: {fmt} {space} (attesa: xywh 1025, con copertura sopra 0.9).")
    print(f"Comando: python build_cache.py CARTELLA_DATI CARTELLA_CACHE {fmt} {space}")
    print(f"Id delle classi: da {min(ids)} a {max(ids)} (atteso: da 1 a 11)")
    print(f"Nodi di testo per pagina: mediana {np.median(n_nodes):.0f}, "
          f"95° percentile {np.percentile(n_nodes, 95):.0f}, massimo {max(n_nodes)}")

    for number, row, raw, text_box in samples:               # immagini di controllo con l'ipotesi migliore
        doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
        page = doc[0]
        for b in page_annotations(row, raw["page_w"], raw["page_h"], fmt, space):
            page.draw_rect(pymupdf.Rect([float(v) for v in b]), color=(1, 0, 0), width=1)
        for b in text_box:
            page.draw_rect(pymupdf.Rect([float(v) for v in b]), color=(0, 0, 1), width=0.3)
        page.get_pixmap(dpi=110).save(f"check_{number:03d}.png")
        doc.close()


if __name__ == "__main__":
    main(sys.argv[1])
