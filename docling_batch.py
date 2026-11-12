"""Annota con Docling tutti i PDF di una cartella, caricando i modelli una volta sola. Si può interrompere e riprendere.

Uso:
  uv run python docling_batch.py CARTELLA_PDF [--out-dir DIR] [--cache DIR] [--ocr] [--limit N]

Per ogni <nome>.pdf scrive in --out-dir <nome>.docling.pdf e <nome>.docling.json e, con --cache, le pseudo-etichette
<nome>_pNNNN.npz (formato della guida, sezione 17.1).

Ripresa: un documento è fatto se esiste il suo .docling.json, che viene scritto per ultimo e in modo atomico (file .tmp
e poi rename). Se lo script si ferma a metà di un documento (Ctrl+C, crash, spegnimento) quel documento non ha il .json
e al rilancio viene rifatto da capo; i .npz parziali vengono sovrascritti. Basta rilanciare lo stesso comando.
I documenti che danno errore finiscono in errors.log e si ritentano al rilancio successivo.
"""
import argparse
import logging
import pathlib
import sys
import time
import traceback

from docling_annotate import annotate_pdf, make_converter


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Annota con Docling tutti i PDF di una cartella (riprendibile).")
    ap.add_argument("folder", help="cartella con i PDF (ricerca ricorsiva di *.pdf)")
    ap.add_argument("--out-dir", help="dove scrivere .docling.pdf/.json e i log (default: <cartella>_docling)")
    ap.add_argument("--cache", help="cartella per le pseudo-etichette .npz (default: non le scrive)")
    ap.add_argument("--ocr", action="store_true", help="attiva l'OCR di Docling")
    ap.add_argument("--limit", type=int, default=0, help="processa al massimo N documenti nuovi (per prova)")
    args = ap.parse_args()

    folder = pathlib.Path(args.folder)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else folder.with_name(folder.name + "_docling")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tmp in out_dir.glob("*.tmp"):                         # resti di un'interruzione durante il salvataggio
        tmp.unlink()

    pdfs = sorted(p for p in folder.rglob("*.pdf") if not p.name.endswith(".docling.pdf"))
    def out_of(p):
        return out_dir / (p.stem + ".docling.pdf")
    todo = [p for p in pdfs if not out_of(p).with_suffix(".json").exists()]
    print(f"{len(pdfs)} PDF, {len(pdfs) - len(todo)} già fatti, {len(todo)} da fare. Output in {out_dir}", flush=True)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        return

    logging.getLogger("docling").setLevel(logging.WARNING)   # niente log per pagina di Docling
    t_load = time.perf_counter()
    converter = make_converter(args.ocr)
    print(f"Converter pronto in {time.perf_counter() - t_load:.1f} s", flush=True)

    progress = open(out_dir / "progress.log", "a", encoding="utf-8")
    t0, pages_done, failed = time.perf_counter(), 0, 0
    try:
        for i, pdf in enumerate(todo, start=1):
            t_doc = time.perf_counter()
            try:
                s = annotate_pdf(pdf, converter, out_of(pdf), cache=args.cache, verbose=False)
            except Exception:
                failed += 1
                with open(out_dir / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {pdf}\n{traceback.format_exc()}\n")
                print(f"[{i}/{len(todo)}] ERRORE {pdf.name} (dettagli in errors.log)", flush=True)
                continue
            pages_done += s["pages"]
            elapsed = time.perf_counter() - t0
            per_doc = elapsed / i
            eta_h = per_doc * (len(todo) - i) / 3600
            cov = s["covered"] / max(s["total"], 1) * 100
            line = (f"[{i}/{len(todo)}] {pdf.name}: {s['pages']} pag, {s['blocks']} blocchi, copertura {cov:.0f}%, "
                    f"{time.perf_counter() - t_doc:.0f} s ({pages_done / elapsed:.2f} pag/s, ETA {eta_h:.1f} h)")
            print(line, flush=True)
            progress.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
            progress.flush()
    except KeyboardInterrupt:
        print("\nInterrotto: il documento in corso verrà rifatto. Rilancia lo stesso comando per riprendere.")
    finally:
        progress.close()
    print(f"Fine: {pages_done} pagine in {(time.perf_counter() - t0) / 3600:.2f} h, {failed} errori")


if __name__ == "__main__":
    main()
