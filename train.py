"""Addestramento. Uso: python train.py CARTELLA_CACHE CARTELLA_USCITA [epoche]"""
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from layout.config import CLASSES, Config
from layout.data import PageDataset, collate, to_device, worker_init
from layout.decode import block_classes, block_match, f1_per_class, find_blocks
from layout.model import LayoutModel, compute_losses, run


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    true, pred = [], []
    link = np.zeros(3)       # veri positivi, falsi positivi, falsi negativi
    blocks = np.zeros(3)     # blocchi corretti, blocchi predetti, blocchi veri
    for batch in loader:
        gt_blocks = batch["block"]
        batch = to_device(batch, device)
        cls_logits, link_logits = run(model, batch)
        cls_logits = cls_logits.cpu().numpy()
        prob = torch.sigmoid(link_logits).cpu().numpy()
        labels = batch["label"].cpu().numpy()
        pairs = batch["pairs"].cpu().numpy()
        link_true = batch["link"].cpu().numpy() > 0.5
        n_nodes = batch["mask"].sum(1).tolist()
        n_pairs = batch["pair_mask"].sum(1).tolist()
        for b, (n, p) in enumerate(zip(n_nodes, n_pairs)):
            logits = cls_logits[b, :n]
            node_cls = logits.argmax(1)
            true.append(labels[b, :n])
            pred.append(node_cls)
            hit, y = prob[b, :p] > cfg.link_threshold, link_true[b, :p]
            link += [(hit & y).sum(), (hit & ~y).sum(), (~hit & y).sum()]
            blk = find_blocks(n, pairs[b, :p], prob[b, :p], node_cls, cfg.link_threshold)
            blocks += block_match(blk, block_classes(blk, logits)[blk], gt_blocks[b], labels[b, :n])
    f1, support = f1_per_class(np.concatenate(true), np.concatenate(pred), len(CLASSES))
    return {
        "node_f1": float(f1[support > 0].mean()),
        "per_class": [float(v) for v in f1],
        "link_f1": float(2 * link[0] / max(2 * link[0] + link[1] + link[2], 1)),
        "block_f1": float(2 * blocks[0] / max(blocks[1] + blocks[2], 1)),
    }


def main(cache_dir, out_dir, epochs=None):
    cfg = Config()
    if epochs is not None:
        cfg.epochs = epochs
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = PageDataset(os.path.join(cache_dir, "train"), cfg, train=True)
    val_ds = PageDataset(os.path.join(cache_dir, "validation"), cfg, train=False)
    torch.backends.cuda.matmul.allow_tf32 = True             # sulle GPU Ampere (RTX 30xx): matmul più veloci
    torch.backends.cudnn.allow_tf32 = True
    loader = dict(collate_fn=collate, num_workers=cfg.num_workers, worker_init_fn=worker_init,
                  persistent_workers=cfg.num_workers > 0, pin_memory=device == "cuda",
                  prefetch_factor=4 if cfg.num_workers > 0 else None)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader)
    val_dl = DataLoader(val_ds, batch_size=max(cfg.batch_size // 2, 1), shuffle=False, **loader)
    print(f"Dispositivo: {device}. Pagine: {len(train_ds)} di training, {len(val_ds)} di validation. "
          f"Processi per i dati: {cfg.num_workers}.")

    model = LayoutModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(train_dl)
    warmup = min(cfg.warmup_steps, max(total // 10, 1))

    def lr_factor(step):                                     # sale nel warmup, poi scende a coseno fino a zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    def eta(step, epochs_done, train_time, eval_time):
        """Tempo mancante: step restanti al ritmo medio finora, più una validation per ogni epoca restante."""
        left = (total - step) * train_time / max(step, 1) + (cfg.epochs - epochs_done) * eval_time
        end = time.strftime("%H:%M", time.localtime(time.time() + left))
        note = "" if eval_time or epochs_done == cfg.epochs else " + validation"   # finché non ne ho misurata una
        return f"ETA {int(left) // 3600}h{int(left) % 3600 // 60:02d}m{note} (fine verso le {end})"

    best, step = -1.0, 0
    train_time = eval_time = 0.0     # secondi spesi finora nel training; durata dell'ultima validation
    for epoch in range(cfg.epochs):
        model.train()
        start = time.time()
        for batch in train_dl:
            batch = to_device(batch, device)
            loss, loss_cls, loss_link = compute_losses(model, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total}  loss classi {loss_cls:.3f}  loss link {loss_link:.3f}  "
                      f"{time.time() - start:.0f} s da inizio epoca  "
                      f"{eta(step, epoch, train_time + time.time() - start, eval_time)}")
        train_time += time.time() - start
        t_eval = time.time()
        m = evaluate(model, val_dl, cfg, device)
        eval_time = time.time() - t_eval
        print(f"Epoca {epoch + 1}: F1 nodi {m['node_f1']:.3f}  F1 link {m['link_f1']:.3f}  "
              f"F1 blocchi {m['block_f1']:.3f}   validation {eval_time:.0f} s  "
              f"{eta(step, epoch + 1, train_time, eval_time)}")
        print("   per classe: " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CLASSES, m["per_class"])))
        if m["block_f1"] > best:
            best = m["block_f1"]
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "metrics": m},
                       os.path.join(out_dir, "best.pt"))
            print("   salvato come modello migliore")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
