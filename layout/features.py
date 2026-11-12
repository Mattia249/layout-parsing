"""Tutto ciò che il modello vede: feature di nodo, id di testo, feature di coppia, coppie candidate."""
import os
import re
import zlib

import numpy as np

from .config import KIND_TEXT, N_FOURIER, N_SHAPE, TOKENS, VOCAB

BASE_FEATS = 42                                  # feature di nodo senza le coordinate di Fourier
NODE_FEATS = BASE_FEATS + 6 * 2 * N_FOURIER      # 138
PER_NODE = ("box", "kind", "size", "style", "font", "vert", "shape", "tok", "label", "block")

_WORD = re.compile(r"\w+|[^\w\s]")
_NUM = re.compile(r"^\(?\d{1,3}(\.\d{1,3})*[.)]?(\s|$)")                           # "2." "2.1 Titolo" "(3)"
_ENUM = re.compile(r"^\(?([a-z]|[ivxlcdm]{1,5})[.)](\s|$)", re.I)                  # "a)" "iv." "(b)"
_BULLET = re.compile(r"^[\u2022\u25cf\u25aa\u25e6\u2023\u00b7\-\u2013\u2014*\uf0b7\uf0a7]\s")
_NUM_ONLY = re.compile(r"^\(?(\d{1,3}(\.\d{1,3})*|[a-z]|[ivxlcdm]{1,5})[.)]?$", re.I)   # il nodo è solo "2."
_NUMERIC = re.compile(r"^[\d\s.,%\u20ac$\u00a3+\-()]+$")                           # cella numerica


# ---------- testo ----------

def _hash(s):
    return zlib.crc32(s.encode("utf-8")) % (VOCAB - 1) + 1       # da 1 a VOCAB-1; lo 0 è il padding


def token_ids(text):
    """Testo -> TOKENS id (prime 12 e ultime 4 parole, due id per parola)."""
    words = [re.sub(r"\d", "0", w) for w in _WORD.findall(text.lower())]
    if len(words) > 16:
        words = words[:12] + words[-4:]
    ids = []
    for w in words:
        ids += [_hash("w:" + w), _hash("p:" + w[:4])]
    out = np.zeros(TOKENS, np.uint16)
    out[:len(ids)] = ids
    return out


def shape_feats(text):
    """Forma del testo: 14 numeri che non dipendono dalla lingua."""
    f = np.zeros(N_SHAPE, np.float32)
    n = len(text)
    if n == 0:
        return f
    letters = sum(c.isalpha() for c in text)
    f[0] = min(np.log1p(n) / 5, 1.5)
    f[1] = min(np.log1p(len(text.split())) / 3, 1.5)
    f[2] = sum(c.isdigit() for c in text) / n
    f[3] = sum(c.isupper() for c in text) / max(letters, 1)
    f[4] = letters / n
    f[5] = sum(not c.isalnum() and not c.isspace() for c in text) / n
    f[6] = bool(_NUM.match(text))
    f[7] = bool(_ENUM.match(text))
    f[8] = bool(_BULLET.match(text))
    f[9] = bool(_NUM_ONLY.match(text))
    f[10] = bool(_NUMERIC.match(text))
    f[11] = text[-1] in ".!?"
    f[12] = text[-1] == ":"
    f[13] = text[0].isupper()
    return f


def add_text_feats(raw):
    """Aggiunge a una pagina estratta le feature che dipendono dal testo."""
    raw["shape"] = np.array([shape_feats(t) for t in raw["texts"]], np.float32).reshape(-1, N_SHAPE)
    raw["tok"] = np.array([token_ids(t) for t in raw["texts"]], np.uint16).reshape(-1, TOKENS)
    return raw


# ---------- nodi ----------

def page_units(raw):
    """Altezza mediana delle righe e corpo mediano del font: le unità di misura della pagina."""
    is_text = raw["kind"] == KIND_TEXT
    if not is_text.any():
        return 12.0, 10.0
    height = raw["box"][:, 3] - raw["box"][:, 1]
    return max(float(np.median(height[is_text])), 1.0), max(float(np.median(raw["size"][is_text])), 1.0)


def log_size(raw, s_med):
    """Logaritmo del rapporto tra il corpo del nodo e il corpo mediano (0 per i nodi non di testo)."""
    rel = np.clip(np.log(np.maximum(raw["size"], 0.1) / s_med), -1.5, 1.5)
    return np.where(raw["kind"] == KIND_TEXT, rel, 0.0).astype(np.float32)


def node_feats(raw):
    """Feature di nodo: [N, 138]."""
    box, w_page, h_page = raw["box"], raw["page_w"], raw["page_h"]
    n = len(box)
    h_med, s_med = page_units(raw)
    x0, y0, x1, y1 = box.T
    w, h = x1 - x0, y1 - y0
    is_text = (raw["kind"] == KIND_TEXT).astype(np.float32)

    pos = np.stack([x0 / w_page, y0 / h_page, x1 / w_page, y1 / h_page,
                    (x0 + x1) / 2 / w_page, (y0 + y1) / 2 / h_page], 1)
    dim = np.stack([w / w_page, h / h_page, np.clip(np.log(np.maximum(h, 0.1) / h_med), -2, 2),
                    np.log1p(w / h_med) / 4, np.full(n, w_page / h_page)], 1)
    page = np.stack([np.full(n, np.log1p(n) / 6), np.full(n, np.log(h_med / 12.0))], 1)
    rank = np.zeros((n, 4), np.float32)
    rank[np.arange(n), np.clip(raw["font"], 0, 3)] = is_text
    kind = np.eye(4, dtype=np.float32)[raw["kind"]]
    angle = pos[:, :, None] * (2.0 ** np.arange(N_FOURIER)) * np.pi          # [N, 6, 8]
    fourier = np.concatenate([np.sin(angle), np.cos(angle)], 2).reshape(n, 6 * 2 * N_FOURIER)

    return np.concatenate([pos, dim, page, log_size(raw, s_med)[:, None], raw["style"], rank,
                           raw["shape"], kind, raw["vert"][:, None], fourier], 1).astype(np.float32)


# ---------- coppie ----------

def _bucket(v):
    """Valore con segno -> bucket da 0 a 24. 12 = quasi zero; poi gli intervalli raddoppiano a ogni passo."""
    a = np.abs(v)
    step = np.where(a < 0.05, 0, 1 + np.floor(np.log2(np.maximum(a, 0.05) / 0.05)))
    return (12 + np.sign(v) * np.clip(step, 0, 12)).astype(np.uint8)


def rel_feats(raw, i, j):
    """Feature di coppia tra i nodi i e j (array di indici). Ritorna uint8 [..., 9]."""
    h_med, s_med = page_units(raw)
    u = raw["box"] / h_med                                   # box in righe
    size = log_size(raw, s_med)
    a, b = u[i], u[j]
    gap_x = np.maximum(0, np.maximum(a[..., 0], b[..., 0]) - np.minimum(a[..., 2], b[..., 2]))
    gap_y = np.maximum(0, np.maximum(a[..., 1], b[..., 1]) - np.minimum(a[..., 3], b[..., 3]))
    values = [b[..., 0] - a[..., 0], b[..., 2] - a[..., 2],
              (b[..., 0] + b[..., 2] - a[..., 0] - a[..., 2]) / 2,
              b[..., 1] - a[..., 1], b[..., 3] - a[..., 3], gap_x, gap_y, size[j] - size[i]]
    both_text = (raw["kind"][i] == KIND_TEXT) & (raw["kind"][j] == KIND_TEXT)
    same_font = np.where(both_text, raw["font"][i] == raw["font"][j], 2).astype(np.uint8)
    shape = np.broadcast(a[..., 0], b[..., 0]).shape
    return np.stack([np.broadcast_to(_bucket(v), shape) for v in values]
                    + [np.broadcast_to(same_font, shape)], -1)


def candidate_pairs(raw, k=12):
    """Coppie (i, j) con i < j su cui la testa di link decide "stesso blocco": [P, 2]."""
    n = len(raw["box"])
    if n < 2:
        return np.zeros((0, 2), np.int64)
    h_med, _ = page_units(raw)
    x0, y0, x1, y1 = (raw["box"] / h_med).T
    gap_x = np.maximum(0, np.maximum(x0[:, None], x0[None]) - np.minimum(x1[:, None], x1[None]))
    gap_y = np.maximum(0, np.maximum(y0[:, None], y0[None]) - np.minimum(y1[:, None], y1[None]))
    dist = np.hypot(gap_x, gap_y)
    np.fill_diagonal(dist, np.inf)
    k = min(k, n - 1)
    near = np.argpartition(dist, k - 1, axis=1)[:, :k]       # i k nodi più vicini a ciascuno
    pairs = [np.stack([np.repeat(np.arange(n), k), near.reshape(-1)], 1)]

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for facing, gap, pos in ((gap_x == 0, gap_y, cy), (gap_y == 0, gap_x, cx)):
        for sign in (1, -1):                                 # sotto e sopra, poi destra e sinistra
            ok = facing & (sign * (pos[None] - pos[:, None]) > 0)
            g = np.where(ok, gap, np.inf)
            j = g.argmin(1)
            found = np.isfinite(g[np.arange(n), j])
            pairs.append(np.stack([np.arange(n)[found], j[found]], 1))

    p = np.sort(np.concatenate(pairs), axis=1)               # ogni coppia nella forma (minore, maggiore)
    return np.unique(p, axis=0).astype(np.int64)


# ---------- esempio completo ----------

def _take(raw, keep):
    out = dict(raw)
    for key in PER_NODE:
        if key in raw:
            out[key] = raw[key][keep]
    if "texts" in raw:
        out["texts"] = [raw["texts"][i] for i in np.arange(len(raw["box"]))[keep]]
    return out


def make_example(raw, cfg, train=False, rng=None, cap=None):
    """Pagina -> tutto ciò che serve al modello (e, se ci sono le etichette, alla loss)."""
    if not cfg.use_graphics:
        raw = _take(raw, raw["kind"] == KIND_TEXT)
    if cap is None:
        cap = cfg.max_nodes_train if train else cfg.max_nodes_eval
    n = len(raw["box"])
    if n > cap:                                              # pagina troppo lunga: una finestra di nodi consecutivi
        start = int(rng.integers(0, n - cap + 1)) if train else 0
        raw = _take(raw, slice(start, start + cap))
    tok = raw["tok"].astype(np.int64)
    if train:
        raw = dict(raw)
        box = raw["box"] + rng.normal(0, 0.3, raw["box"].shape).astype(np.float32)   # rumore di 0,3 punti
        box[:, 2] = np.maximum(box[:, 2], box[:, 0] + 0.1)
        box[:, 3] = np.maximum(box[:, 3], box[:, 1] + 0.1)
        raw["box"] = box
        if rng.random() < cfg.text_dropout:                  # ogni tanto la pagina viene vista senza testo
            tok = np.zeros_like(tok)

    idx = np.arange(len(raw["box"]))
    pairs = candidate_pairs(raw, cfg.knn)
    ex = {
        "feats": node_feats(raw),
        "tok": tok,
        "rel": rel_feats(raw, idx[:, None], idx[None, :]),           # [N, N, 9]
        "pairs": pairs,
        "pair_rel": rel_feats(raw, pairs[:, 0], pairs[:, 1]),        # [P, 9]
        "box": raw["box"],
    }
    if "texts" in raw:
        ex["texts"] = raw["texts"]
    if "label" in raw:
        blk = raw["block"]
        ex["label"] = raw["label"].astype(np.int64)
        ex["block"] = blk
        ex["link"] = ((blk[pairs[:, 0]] == blk[pairs[:, 1]]) & (blk[pairs[:, 0]] >= 0)).astype(np.float32)
    return ex


# ---------- cache su disco ----------

def save_raw(raw, path):
    """Scrive una pagina in un file .npz. Scrive prima un file temporaneo: un'interruzione non lascia file rotti."""
    raw = dict(raw)
    text = "\n".join(raw.pop("texts"))
    with open(path + ".part", "wb") as f:
        np.savez_compressed(f, text=text, **raw)
    os.replace(path + ".part", path)


def load_raw(path, with_text=False):
    with np.load(path) as z:
        raw = {key: z[key] for key in z.files if key != "text"}
        if with_text:
            raw["texts"] = str(z["text"]).split("\n")
    raw["page_w"], raw["page_h"] = float(raw["page_w"]), float(raw["page_h"])
    return raw
