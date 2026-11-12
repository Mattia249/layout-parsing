"""Dai box annotati di DocLayNet alle etichette dei nodi."""
import numpy as np

from .config import OTHER


def ann_to_points(bboxes, coco_w, coco_h, page_w, page_h, fmt="xywh"):
    """Box annotati -> [x0, y0, x1, y1] in punti. fmt: "xywh" oppure "xyxy" (vedi check_data.py)."""
    a = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy()
    if fmt == "xywh":
        a[:, 2] += a[:, 0]
        a[:, 3] += a[:, 1]
    a[:, [0, 2]] *= page_w / coco_w
    a[:, [1, 3]] *= page_h / coco_h
    return a


def page_annotations(row, page_w, page_h, fmt="xywh", space="1025"):
    """Box annotati di una riga del dataset, in punti.
    space: "1025" se i box sono nello spazio dell'immagine, "punti" se sono già in punti della pagina."""
    meta = row["metadata"]
    src_w, src_h = (page_w, page_h) if space == "punti" else (meta["coco_width"], meta["coco_height"])
    return ann_to_points(row["bboxes"], src_w, src_h, page_w, page_h, fmt)


def assign(box, ann_box, ann_cls, min_cover=0.5):
    """Per ogni nodo: classe (0..11) e blocco (indice del box annotato che lo contiene, -1 se nessuno)."""
    n = len(box)
    label = np.full(n, OTHER, np.int64)
    block = np.full(n, -1, np.int64)
    if n == 0 or len(ann_box) == 0:
        return label, block
    b = box.astype(np.float32).copy()
    b[:, :2] -= 0.5                                          # gonfio di mezzo punto: i filetti hanno area quasi nulla
    b[:, 2:] += 0.5
    ix = np.minimum(b[:, None, 2], ann_box[None, :, 2]) - np.maximum(b[:, None, 0], ann_box[None, :, 0])
    iy = np.minimum(b[:, None, 3], ann_box[None, :, 3]) - np.maximum(b[:, None, 1], ann_box[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    cover = inter / ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[:, None]     # quota del nodo coperta da ogni box
    ann_area = np.abs((ann_box[:, 2] - ann_box[:, 0]) * (ann_box[:, 3] - ann_box[:, 1]))
    score = cover - 1e-6 * ann_area / max(float(ann_area.max()), 1e-6)       # a parità vince il box più piccolo
    best = score.argmax(1)
    ok = cover[np.arange(n), best] >= min_cover
    label[ok] = ann_cls[best[ok]]
    block[ok] = best[ok]
    return label, block
