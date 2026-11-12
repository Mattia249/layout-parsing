"""Dal PDF ai nodi. Coordinate in punti (1/72 di pollice), origine in alto a sinistra."""
import re

import numpy as np

from .config import KIND_IMAGE, KIND_RULE_H, KIND_RULE_V, KIND_TEXT

GAP_EM = 1.0                          # spazio (in multipli del corpo del font) oltre il quale la riga si spezza
_SUBSET = re.compile(r"^[A-Z]{6}\+")  # prefisso dei font incorporati solo in parte, es. "ABCDEF+Arial"


def _text_node(spans, horizontal):
    solid = [s for s in spans if s["text"].strip()]          # gli span di soli spazi contano solo per il testo
    n_chars = [len(s["text"]) for s in solid]
    total = max(sum(n_chars), 1)
    main = solid[int(np.argmax(n_chars))]                    # span dominante: quello con più caratteri
    name = _SUBSET.sub("", main["font"]).lower()

    def frac(test):                                          # quota di caratteri che soddisfa una condizione
        return sum(n for s, n in zip(solid, n_chars) if test(s)) / total

    def is_bold(s):
        return bool(s["flags"] & 16) or "bold" in s["font"].lower()

    def is_italic(s):
        return bool(s["flags"] & 2) or "italic" in s["font"].lower() or "oblique" in s["font"].lower()

    return {
        "box": (min(s["bbox"][0] for s in solid), min(s["bbox"][1] for s in solid),
                max(s["bbox"][2] for s in solid), max(s["bbox"][3] for s in solid)),
        "kind": KIND_TEXT,
        "text": " ".join("".join(s["text"] for s in spans).split()),
        "size": float(main["size"]),
        "font": f"{name}|{round(main['size'] * 2) / 2}",
        "style": (frac(is_bold), frac(is_italic),
                  float(bool(main["flags"] & 8)),            # monospaziato
                  float(main.get("color", 0) != 0),          # colore diverso dal nero
                  float(bool(main["flags"] & 1))),           # apice
        "vert": 0 if horizontal else 1,
    }


def nodes_from_textdict(data):
    """Trasforma l'uscita di page.get_text("dict") in una lista di nodi di testo."""
    nodes = []
    for block in data["blocks"]:
        if block.get("type", 0) != 0:                        # 0 = testo, 1 = immagine
            continue
        for line in block["lines"]:
            horizontal = abs(line.get("dir", (1.0, 0.0))[0]) > 0.95
            group, prev = [], None
            for s in line["spans"]:
                if not s["text"].strip():
                    if group:
                        group.append(s)
                    continue
                if prev is not None and horizontal:
                    gap = s["bbox"][0] - prev["bbox"][2]
                    if abs(gap) > GAP_EM * max(prev["size"], s["size"]):
                        nodes.append(_text_node(group, horizontal))
                        group = []
                group.append(s)
                prev = s
            if group:
                nodes.append(_text_node(group, horizontal))
    return nodes


def _blank_node(box, kind):
    return {"box": box, "kind": kind, "text": "", "size": 0.0, "font": None, "style": (0.0,) * 5, "vert": 0}


def graphic_nodes(page, max_rules=300, max_images=50):
    """Filetti (linee sottili) e immagini raster della pagina."""
    rules = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.height <= 2 and r.width >= 8:
            rules.append((r.width, (r.x0, r.y0, r.x1, r.y1), KIND_RULE_H))
        elif r.width <= 2 and r.height >= 8:
            rules.append((r.height, (r.x0, r.y0, r.x1, r.y1), KIND_RULE_V))
    rules.sort(reverse=True)                                 # se sono troppi, tengo i più lunghi
    nodes = [_blank_node(box, kind) for _, box, kind in rules[:max_rules]]
    images = [tuple(i["bbox"]) for i in page.get_image_info()]
    images = [b for b in images if b[2] - b[0] >= 8 and b[3] - b[1] >= 8][:max_images]
    return nodes + [_blank_node(b, KIND_IMAGE) for b in images]


def reading_order(box, kind):
    """Ordine per righe visive: dall'alto in basso, e da sinistra a destra sulla stessa riga."""
    if len(box) == 0:
        return np.zeros(0, np.int64)
    is_text = kind == KIND_TEXT
    height = box[:, 3] - box[:, 1]
    h_med = float(np.median(height[is_text])) if is_text.any() else 12.0
    cy = (box[:, 1] + box[:, 3]) / 2
    row = np.zeros(len(box), np.int64)
    current, start = 0, None
    for k in np.argsort(cy, kind="stable"):
        if start is None or cy[k] - start > 0.5 * h_med:     # più di mezza riga sotto: comincia una riga nuova
            current += 1
            start = cy[k]
        row[k] = current
    return np.lexsort((box[:, 0], row))


def pack_nodes(nodes, page_w, page_h):
    """Lista di nodi -> array numpy, in ordine di lettura."""
    box = np.array([n["box"] for n in nodes], np.float32).reshape(-1, 4)
    box[:, [0, 2]] = box[:, [0, 2]].clip(0, page_w)
    box[:, [1, 3]] = box[:, [1, 3]].clip(0, page_h)
    box[:, 2] = np.maximum(box[:, 2], box[:, 0] + 0.1)       # mai larghezza o altezza nulla
    box[:, 3] = np.maximum(box[:, 3], box[:, 1] + 0.1)
    kind = np.array([n["kind"] for n in nodes], np.uint8)

    weight = {}                                              # caratteri scritti con ciascun font
    for n in nodes:
        if n["kind"] == KIND_TEXT:
            weight[n["font"]] = weight.get(n["font"], 0) + len(n["text"])
    rank = {f: r for r, f in enumerate(sorted(weight, key=weight.get, reverse=True))}

    raw = {
        "page_w": float(page_w), "page_h": float(page_h), "box": box, "kind": kind,
        "size": np.array([n["size"] for n in nodes], np.float32),
        "style": np.array([n["style"] for n in nodes], np.float32).reshape(-1, 5),
        "font": np.array([rank.get(n["font"], -1) for n in nodes], np.int32),   # 0 = font più usato
        "vert": np.array([n["vert"] for n in nodes], np.uint8),
        "texts": [n["text"] for n in nodes],
    }
    order = reading_order(box, kind)
    for key in ("box", "kind", "size", "style", "font", "vert"):
        raw[key] = raw[key][order]
    raw["texts"] = [raw["texts"][i] for i in order]
    return raw


def extract_page(page, with_graphics=True):
    """Pagina di PyMuPDF -> dizionario di array."""
    import pymupdf   # importato qui: il resto del file si può provare anche senza PyMuPDF

    if page.rotation:
        page.remove_rotation()
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES   # non caricare i pixel delle immagini
    nodes = nodes_from_textdict(page.get_text("dict", flags=flags))
    if with_graphics:
        nodes += graphic_nodes(page)
    return pack_nodes(nodes, page.rect.width, page.rect.height)
