"""Prova a secco su pagine sintetiche, senza dataset.
Uso: python smoke_test.py [cartella_cache_finta]"""
import os
import sys

import numpy as np

from layout.config import CLASSES, OTHER, REL_BUCKETS, TOKENS, Config
from layout.decode import block_match, find_blocks
from layout.extract import nodes_from_textdict, pack_nodes
from layout.features import NODE_FEATS, add_text_feats, make_example, save_raw, token_ids
from layout.targets import ann_to_points, assign

WORDS = ("il la di che un per con non una su le si da come ma anno valore totale ricavi costi "
         "margine periodo societa risultato netto gruppo mercato crescita rispetto precedente").split()
W, H = 595.0, 842.0


def fake_page(rng):
    """Una pagina nel formato di page.get_text("dict"), con le annotazioni nello stile di DocLayNet."""
    lines, anns = [], []

    def span(text, x, y, size, bold=False):
        return {"text": text, "bbox": (x, y, x + 0.5 * size * len(text), y + 1.2 * size), "size": size,
                "flags": 16 if bold else 0, "font": "Fake-Bold" if bold else "Fake", "color": 0}

    def words(n):
        return " ".join(rng.choice(WORDS, n))

    def add(cls, block_lines):
        boxes = [s["bbox"] for ln in block_lines for s in ln]
        anns.append((CLASSES.index(cls) + 1, min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)))
        lines.extend({"dir": (1.0, 0.0), "spans": ln} for ln in block_lines)

    add("page_header", [[span("Relazione annuale 2026", 72, 30, 8)]])
    add("title", [[span("Titolo del documento", 150, 70, 20, True)]])
    y = 120.0
    for sec in range(int(rng.integers(2, 4))):
        # il numero è uno span lontano dal nome: l'estrazione deve farne due nodi, l'etichetta un solo blocco
        add("section_header", [[span(f"{sec + 1}.", 72, y, 13, True), span("Nome della sezione", 110, y, 13, True)]])
        y += 26
        for _ in range(int(rng.integers(1, 3))):
            para = []
            for k in range(int(rng.integers(2, 6))):
                first = span(words(10), 72, y, 10)
                # una parola in grassetto attaccata al resto: deve restare nello stesso nodo
                para.append([first, span(" " + words(1), first["bbox"][2], y, 10, True)])
                y += 12
            add("text", para)
            y += 8
        if rng.random() < 0.5:
            for _ in range(int(rng.integers(2, 4))):
                add("list_item", [[span("\u2022 " + words(6), 90, y, 10)]])
                y += 14
            y += 6
    add("caption", [[span("Tabella 1: " + words(4), 72, y, 9)]])
    y += 16
    rows = [[span(str(int(rng.integers(1, 9999))), x, y + 13 * r, 9) for x in (72, 250, 420)]
            for r in range(int(rng.integers(3, 6)))]
    add("table", rows)                                       # ogni riga è una sola riga del PDF con tre celle lontane
    add("page_footer", [[span("12", 290, 800, 8)]])
    lines.append({"dir": (1.0, 0.0), "spans": [span("BOZZA", 500, 815, 7)]})   # testo senza annotazione
    n_nodes = sum(1 for ln in lines for s in ln["spans"] if not s["text"].startswith(" "))
    bboxes = [[x0 * 1025 / W, y0 * 1025 / H, (x1 - x0) * 1025 / W, (y1 - y0) * 1025 / H]
              for _, x0, y0, x1, y1 in anns]
    return {"blocks": [{"type": 0, "lines": lines}]}, bboxes, [a[0] for a in anns], n_nodes


def fake_raw(rng):
    data, bboxes, cats, n_nodes = fake_page(rng)
    raw = add_text_feats(pack_nodes(nodes_from_textdict(data), W, H))
    ann = ann_to_points(bboxes, 1025, 1025, W, H, "xywh")
    raw["label"], raw["block"] = assign(raw["box"], ann, np.array(cats) - 1)
    return raw, bboxes, n_nodes


def check_pipeline(rng, n_pages=50):
    cfg = Config()
    ok_blocks = np.zeros(3)
    right_cover, wrong_cover = [], []
    for _ in range(n_pages):
        raw, bboxes, n_nodes = fake_raw(rng)
        assert len(raw["box"]) == n_nodes, "l'estrazione non ha spezzato le righe come previsto"
        assert (raw["label"] == OTHER).sum() == 1 and (raw["block"] < 0).sum() == 1, "etichette sbagliate"
        right_cover.append((raw["block"] >= 0).mean())
        wrong = ann_to_points(bboxes, 1025, 1025, W, H, "xyxy")
        wrong_cover.append((assign(raw["box"], wrong, np.zeros(len(wrong), np.int64))[1] >= 0).mean())
        ex = make_example(raw, cfg)
        n = len(ex["feats"])
        assert ex["feats"].shape == (n, NODE_FEATS) and np.isfinite(ex["feats"]).all()
        assert ex["tok"].shape == (n, TOKENS) and ex["rel"].shape == (n, n, 9)
        assert ex["rel"][..., :8].max() < REL_BUCKETS and ex["rel"][..., 8].max() <= 2
        assert (ex["pairs"][:, 0] < ex["pairs"][:, 1]).all()
        # con i link veri, i candidati devono bastare a ricostruire esattamente i blocchi veri
        blocks = find_blocks(n, ex["pairs"], ex["link"])
        ok_blocks += block_match(blocks, ex["label"], ex["block"], ex["label"])
    assert (token_ids("Tabella 1") == token_ids("tabella 2")).all(), "l'hashing del testo non è stabile"
    assert ok_blocks[0] == ok_blocks[1] == ok_blocks[2], f"i candidati non bastano: {ok_blocks}"
    print(f"Catena dei dati: OK su {n_pages} pagine ({int(ok_blocks[2])} blocchi ricostruiti dai link veri).")
    print(f"Copertura dei nodi: {np.mean(right_cover):.2f} con la convenzione giusta dei box, "
          f"{np.mean(wrong_cover):.2f} con quella sbagliata.")


def check_model(rng):
    try:
        import torch
    except ImportError:
        print("PyTorch non è installato: salto la prova del modello.")
        return
    from layout.data import collate
    from layout.model import LayoutModel, compute_losses, run

    n_params = sum(p.numel() for p in LayoutModel(Config()).parameters())
    print(f"Parametri del modello: {n_params:,} (attesi 9,334,837)")

    cfg = Config(d_model=128, n_layers=2, d_ff=256, lr=1e-3)      # modello ridotto: la prova dura poco
    torch.manual_seed(0)
    model = LayoutModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pages = [make_example(fake_raw(rng)[0], cfg) for _ in range(16)]
    history = []
    model.train()
    for step in range(150):
        batch = collate([pages[i] for i in rng.choice(16, 8, replace=False)])
        loss, _, _ = compute_losses(model, batch, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    model.eval()
    with torch.no_grad():
        batch = collate(pages)
        cls_logits, _ = run(model, batch)
        keep = batch["label"] >= 0
        acc = (cls_logits.argmax(-1)[keep] == batch["label"][keep]).float().mean().item()
    first, last = np.mean(history[:5]), np.mean(history[-5:])
    verdict = "OK" if last < 0.7 * first else "ATTENZIONE: la loss non scende"
    print(f"Modello: loss da {first:.2f} a {last:.2f}, accuratezza sulle pagine di prova {acc:.2f}. {verdict}")


def write_cache(folder, rng, n_train=200, n_val=40):
    for split, count in (("train", n_train), ("validation", n_val)):
        os.makedirs(os.path.join(folder, split), exist_ok=True)
        for k in range(count):
            save_raw(fake_raw(rng)[0], os.path.join(folder, split, f"finta_{k:04d}.npz"))
    print(f"Cache finta scritta in {folder}: {n_train} pagine di training, {n_val} di validation.")


if __name__ == "__main__":
    generator = np.random.default_rng(0)
    check_pipeline(generator)
    check_model(generator)
    if len(sys.argv) > 1:
        write_cache(sys.argv[1], generator)
