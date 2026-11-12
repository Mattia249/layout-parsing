"""Copertura dei box (xywh, 1025) e nodi per pagina, per categoria di documento, su 12 file sparsi del train.
Uso: uv run check_categories.py"""
import collections, glob, os, sys
import numpy as np, pyarrow.parquet as pq, pymupdf
from layout.config import KIND_TEXT
from layout.extract import extract_page
from layout.targets import assign, page_annotations

pymupdf.TOOLS.mupdf_display_errors(False)
files = sorted(glob.glob("datasets/DocLayNet-v1.2/data/train-*.parquet"))
stats = collections.defaultdict(lambda: {"cov": [], "n": [], "vuote": 0})
for path in files[::6]:                                   # 12 file sparsi sui 72
    done = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["bboxes", "category_id", "metadata", "pdf"]):
        for row in batch.to_pylist():
            s = stats[row["metadata"]["doc_category"]]
            doc = pymupdf.open(stream=row["pdf"], filetype="pdf"); raw = extract_page(doc[0]); doc.close()
            tb = raw["box"][raw["kind"] == KIND_TEXT]
            done += 1
            if len(tb) == 0: s["vuote"] += 1; continue
            ann = page_annotations(row, raw["page_w"], raw["page_h"])
            s["cov"].append((assign(tb, ann, np.zeros(len(ann), np.int64))[1] >= 0).mean()); s["n"].append(len(tb))
        if done >= 60: break
for cat, s in sorted(stats.items()):
    c, n = np.array(s["cov"]), np.array(s["n"])
    print(f"{cat:22s} pagine {len(c)+s['vuote']:4d}  senza testo {s['vuote']:3d}  copertura media {c.mean():.3f}  "
          f"pagine <0.8: {(c<0.8).mean():.1%}  nodi mediana {np.median(n):.0f} p95 {np.percentile(n,95):.0f} max {n.max()}")
