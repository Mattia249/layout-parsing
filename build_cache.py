"""Estrae tutte le pagine del dataset e le salva con le etichette. Si può interrompere e rilanciare.
Uso: python build_cache.py D:/AI/datasets/DocLayNet-v1.2/data D:/AI/cache [xywh|xyxy] [1025|punti]"""
import glob
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import pymupdf

from layout.config import KIND_TEXT
from layout.extract import extract_page
from layout.features import add_text_feats, save_raw
from layout.targets import assign, page_annotations

COLUMNS = ["bboxes", "category_id", "metadata", "pdf"]


def build_split(data_dir, out_dir, split, fmt, space):
    files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not files:
        print(f"Nessun file per lo split '{split}' in {data_dir}: lo salto.")
        return
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    written = skipped = failed = 0
    start = time.time()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=COLUMNS):
            for row in batch.to_pylist():
                meta = row["metadata"]
                dst = os.path.join(out_dir, split, meta["page_hash"] + ".npz")
                if os.path.exists(dst):
                    continue
                try:                                         # un PDF illeggibile non deve fermare ore di lavoro
                    doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
                    raw = add_text_feats(extract_page(doc[0]))
                    doc.close()
                except Exception as error:
                    failed += 1
                    print(f"Pagina {meta['page_hash']} saltata: {error}")
                    continue
                if not (raw["kind"] == KIND_TEXT).any():
                    skipped += 1
                    continue
                ann = page_annotations(row, raw["page_w"], raw["page_h"], fmt, space)
                cls = np.asarray(row["category_id"], np.int64) - 1          # da 1..11 a 0..10
                if len(cls) and (cls.min() < 0 or cls.max() > 10):           # meglio fermarsi qui che a metà training
                    raise ValueError(f"Pagina {meta['page_hash']}: id di classe fuori da 1..11: {sorted(set(cls + 1))}")
                raw["label"], raw["block"] = assign(raw["box"], ann, cls)
                save_raw(raw, dst)
                written += 1
                if written % 500 == 0:
                    print(f"{split}: {written} pagine scritte in {time.time() - start:.0f} s")
    print(f"{split}: finito. Scritte {written}, senza testo {skipped}, illeggibili {failed}.")


if __name__ == "__main__":
    fmt = sys.argv[3] if len(sys.argv) > 3 else "xywh"
    space = sys.argv[4] if len(sys.argv) > 4 else "1025"
    for split in ("train", "validation", "test"):
        build_split(sys.argv[1], sys.argv[2], split, fmt, space)
