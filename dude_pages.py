"""Quante pagine hanno i documenti di DUDE: media, mediana e istogramma.
Uso: uv run python dude_pages.py [cartella PDF di DUDE] [file PNG di uscita]"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

DEFAULT_PDF_DIR = "C:/Users/Matti/Desktop/Layout/datasets/DUDE/data/DUDE_train-val-test_binaries/DUDE_train-val-test_binaries/PDF"
SPLITS = ("train", "val", "test")


def count_pages(path):
    try:
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception as e:  # PDF rotto: lo segnalo e lo salto
        print(f"! {os.path.basename(path)}: {e}", file=sys.stderr)
        return None


def main(pdf_dir, out_png):
    pages = {}  # split -> lista di pagine per documento
    for split in SPLITS:
        paths = sorted(glob.glob(os.path.join(pdf_dir, split, "*.pdf")))
        with ProcessPoolExecutor() as ex:
            counts = list(ex.map(count_pages, paths, chunksize=32))
        pages[split] = np.array([c for c in counts if c is not None])
        print(f"{split}: {len(paths)} PDF, {len(pages[split])} letti")

    everything = np.concatenate(list(pages.values()))
    print(f"\n{'':6}{'documenti':>10}{'pagine':>9}{'media':>8}{'mediana':>9}{'min':>5}{'max':>6}")
    for name, a in [*pages.items(), ("totale", everything)]:
        print(f"{name:6}{len(a):>10}{a.sum():>9}{a.mean():>8.2f}{np.median(a):>9.0f}{a.min():>5}{a.max():>6}")
    p90, p99 = np.percentile(everything, [90, 99])
    print(f"\n90° percentile: {p90:.0f} pagine, 99°: {p99:.0f} pagine")

    # istogramma: una barra per ogni numero di pagine fino a CAP, il resto in un'unica barra "CAP+"
    cap = 30
    clipped = np.minimum(everything, cap)
    values = np.arange(1, cap + 1)
    freq = np.bincount(clipped, minlength=cap + 1)[1:]
    mean, median = everything.mean(), np.median(everything)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    bars = ax.bar(values, freq, width=0.8, color="#2a6fb0")
    bars[-1].set_color("#8a97a6")  # barra cumulativa: colore diverso
    ax.axvline(mean, color="#c0392b", lw=1.5, ls="--", label=f"media = {mean:.1f}")
    ax.axvline(median, color="#1e8449", lw=1.5, ls=":", label=f"mediana = {median:.0f}")
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values[:-1]] + [f"{cap}+"], fontsize=8)
    ax.set_xlabel("Pagine per documento")
    ax.set_ylabel("Numero di documenti")
    ax.set_title(f"DUDE: pagine per documento ({len(everything)} PDF, {everything.sum()} pagine)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for v, f in zip(values, freq):  # etichetta solo le barre più alte
        if f >= 0.03 * freq.max():
            ax.text(v, f, str(f), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Grafico salvato in {out_png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_DIR,
         sys.argv[2] if len(sys.argv) > 2 else "results/dude_pages.png")
