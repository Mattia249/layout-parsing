"""Confronta dataset di PDF: pagine per documento e quanta grafica c'è (immagini e tracciati vettoriali).
Per ogni documento si campionano al massimo SAMPLE pagine distribuite lungo tutto il documento.
Una pagina è "grafica" se ha almeno 1 immagine incorporata o almeno MIN_DRAWINGS tracciati vettoriali.
Uso: uv run python compare_datasets.py [--max-docs N] nome=cartella [nome=cartella ...]
(le cartelle vengono cercate ricorsivamente per *.pdf)"""
import argparse
import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pymupdf

SAMPLE = 8
MIN_DRAWINGS = 20


def measure(path):
    """-> (pagine, pagine campionate, pagine grafiche, immagini, tracciati) oppure None se il PDF non si apre"""
    try:
        with pymupdf.open(path) as doc:
            n = doc.page_count
            idx = sorted({round(i * (n - 1) / max(SAMPLE - 1, 1)) for i in range(min(SAMPLE, n))})
            graphic = imgs = draws = 0
            for i in idx:
                page = doc[i]
                ni, nd = len(page.get_images()), len(page.get_drawings())
                imgs += ni
                draws += nd
                graphic += ni >= 1 or nd >= MIN_DRAWINGS
            return n, len(idx), graphic, imgs, draws
    except Exception:
        return None


def main(sets, max_docs):
    print(f"{'dataset':<18}{'doc':>6}{'pag.media':>10}{'mediana':>8}{'p90':>5}{'min':>5}{'max':>6}"
          f"{'%pag.grafiche':>14}{'img/pag':>9}{'vett/pag':>10}{'%doc>=20pag':>12}")
    for name, folder in sets:
        paths = sorted(glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True))
        if max_docs and len(paths) > max_docs:
            paths = random.Random(0).sample(paths, max_docs)
        with ProcessPoolExecutor() as ex:
            res = [r for r in ex.map(measure, paths, chunksize=8) if r]
        n, k, g, im, dr = (np.array(c) for c in zip(*res))
        print(f"{name:<18}{len(n):>6}{n.mean():>10.1f}{np.median(n):>8.0f}{np.percentile(n, 90):>5.0f}{n.min():>5}{n.max():>6}"
              f"{g.sum() / k.sum() * 100:>14.1f}{im.sum() / k.sum():>9.2f}{dr.sum() / k.sum():>10.1f}{(n >= 20).mean() * 100:>12.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=0, help="campiona al massimo N documenti per dataset (0 = tutti)")
    ap.add_argument("sets", nargs="+", help="nome=cartella")
    a = ap.parse_args()
    main([s.split("=", 1) for s in a.sets], a.max_docs)
