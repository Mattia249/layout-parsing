"""Estrazione con LiteParse (run-llama/liteparse): testo con font e colore, filetti e forme vettoriali, immagini.

Produce gli stessi nodi di extract.py, così il modello non cambia, e in più descrive per l'uscita il font dei
blocchi e i vettori della pagina. Le coordinate di LiteParse sono già quelle di PyMuPDF: punti, origine in alto
a sinistra.

Tre cose verificate sui dati, che spiegano le scelte qui sotto:
- il corpo del font giusto è font_height; font_size a volte vale 1.0 (font scalati dalla matrice del testo);
- font_weight non è affidabile (CMR10, un tondo, ha peso 744; CMBX10, il neretto, 836): grassetto e corsivo
  si ricavano dal nome del font, e il peso si usa solo se il nome non dice niente;
- un TextItem può essere una parola sola: le righe vanno ricostruite raggruppando gli item per altezza.
"""
import re

import numpy as np

from .config import KIND_IMAGE, KIND_RULE_H, KIND_RULE_V
from .extract import _blank_node, nodes_from_textdict, pack_nodes

_BOLD = re.compile(r"bold|black|heavy|semibold|demibold|extrabold|ultrabold|cmbx|[-,]bd\b|[-,]b$", re.I)
_LIGHT = re.compile(r"light|thin|regular|roman|book|medium|\bcmr\d|cmmi", re.I)
_ITALIC = re.compile(r"italic|oblique|slant|[-,]it\b|[-,]i$|cmti|cmmi|cmsl", re.I)
_MONO = re.compile(r"mono|courier|consol|typewriter|code|cmtt|xtt\b|\btt\d", re.I)


def make_parser(images=True):
    from liteparse import LiteParse

    return LiteParse(quiet=True, ocr_enabled=False, extract_vector_graphics=True, extract_text_metadata=True,
                     extract_images=images, continue_on_page_error=True)


def parse_document(path, images=True):
    """Tutto il documento in una volta (LiteParse lavora per file): ritorna (pagine, immagini per pagina)."""
    result = make_parser(images).parse(str(path))
    per_page = {}
    for img in result.images or []:
        per_page.setdefault(img.page, []).append(img)
    return result.pages, per_page


# ---------- font ----------

def font_style(item):
    """(grassetto, corsivo, monospaziato) dal nome del font; il peso solo se il nome non decide."""
    name = item.font_name or ""
    bold = bool(_BOLD.search(name)) or (not _LIGHT.search(name) and (item.font_weight or 0) >= 700)
    return bold, bool(_ITALIC.search(name)), bool(_MONO.search(name))


def font_size(item):
    return float(item.font_height or item.font_size or item.height)


def rgb(color):
    """'ffRRGGBB' (ARGB esadecimale) -> intero 0xRRGGBB, come il campo color di PyMuPDF. None -> nero."""
    if not color:
        return 0
    return int(color[-6:], 16)


# ---------- testo ----------

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")       # caratteri di controllo che LiteParse a volte lascia


def _span(item, text):
    text = _CONTROL.sub(" ", text)
    bold, italic, mono = font_style(item)
    return {"text": text, "bbox": (item.x, item.y, item.x + item.width, item.y + item.height),
            "size": font_size(item), "font": item.font_name or "sconosciuto", "color": rgb(item.fill_color),
            "flags": 16 * bold + 2 * italic + 8 * mono}


def textdict(lp_page):
    """Item di testo di LiteParse -> dizionario nel formato di page.get_text("dict").

    Gli item orizzontali che si sovrappongono in altezza per più di metà formano una riga; dentro la riga, in
    ordine da sinistra a destra, diventano span. nodes_from_textdict spezzerà poi la riga sui grandi spazi.
    Due eccezioni, trovate confrontando con PyMuPDF sulle pagine di validation:
    - un item molto più alto della mediana (es. un "25" decorativo di 300 punti) è una riga a sé: altrimenti
      si sovrappone a decine di righe e si incolla a una di esse;
    - dentro una riga, due item vicini ma non consecutivi nell'ordine in cui il PDF li scrive appartengono a
      blocchi diversi (una colonna, un riquadro): li separo, come fa PyMuPDF.
    """
    items = [it for it in lp_page.text_items if it.text.strip()]
    stream = {id(it): k for k, it in enumerate(items)}       # ordine di scrittura nel PDF
    flat = [it for it in items if abs(((it.rotation or 0) + 45) % 180 - 45) < 10]       # quasi orizzontali
    rotated = [it for it in items if it not in flat]
    h_med = float(np.median([it.height for it in flat])) if flat else 0.0
    huge = [it for it in flat if it.height > 2.5 * h_med]
    flat = [it for it in flat if it.height <= 2.5 * h_med]
    lines, current = [[it] for it in huge], []
    for it in sorted(flat, key=lambda it: it.y + it.height / 2):
        if current:
            top = min(c.y for c in current)
            bottom = max(c.y + c.height for c in current)
            overlap = min(bottom, it.y + it.height) - max(top, it.y)
            if overlap < 0.5 * min(it.height, bottom - top):
                lines.append(current)
                current = []
        current.append(it)
    if current:
        lines.append(current)

    out = []
    for line in lines:
        line.sort(key=lambda it: it.x)
        spans = []
        for k, it in enumerate(line):
            text = it.text
            if k + 1 < len(line):
                nxt = line[k + 1]
                gap = nxt.x - (it.x + it.width)
                if stream[id(nxt)] != stream[id(it)] + 1 and gap > 0.25 * font_size(it):
                    spans.append(_span(it, text))            # blocchi diversi del PDF: riga separata
                    out.append({"dir": (1.0, 0.0), "spans": spans})
                    spans = []
                    continue
                if gap > 0.08 * font_size(it) and not text.endswith(" "):    # con 0,15 si perdevano spazi veri
                    text += " "
            spans.append(_span(it, text))
        out.append({"dir": (1.0, 0.0), "spans": spans})
    for it in rotated:                                       # testo ruotato: un nodo per item, senza spezzare
        out.append({"dir": (0.0, 1.0), "spans": [_span(it, it.text)]})
    return {"blocks": [{"type": 0, "lines": out}]}


# ---------- vettori e immagini ----------

def rules(lp_page, min_len=8.0, max_thick=2.0):
    """Filetti orizzontali e verticali: linee tracciate e rettangoli sottili. Ritorna [(lunghezza, box, tipo)]."""
    found = []
    vg = lp_page.vector_graphics
    if vg is None:
        return found
    for ln in vg.lines:
        if not (ln.stroke or ln.fill):
            continue
        x0, x1 = sorted((ln.x1, ln.x2))
        y0, y1 = sorted((ln.y1, ln.y2))
        half = max(ln.stroke_width or 0.5, 0.5) / 2
        if y1 - y0 <= max_thick and x1 - x0 >= min_len:
            found.append((x1 - x0, (x0, y0 - half, x1, y1 + half), KIND_RULE_H))
        elif x1 - x0 <= max_thick and y1 - y0 >= min_len:
            found.append((y1 - y0, (x0 - half, y0, x1 + half, y1), KIND_RULE_V))
    for sh in vg.shapes:
        x, y, w, h = sh.bbox
        if h <= max_thick and w >= min_len:
            found.append((w, (x, y, x + w, y + h), KIND_RULE_H))
        elif w <= max_thick and h >= min_len:
            found.append((h, (x, y, x + w, y + h), KIND_RULE_V))
    return found


def graphic_nodes(lp_page, lp_images, max_rules=300, max_images=50):
    """Come extract.graphic_nodes, ma dai vettori e dalle immagini di LiteParse."""
    found = sorted(rules(lp_page), key=lambda r: r[0], reverse=True)[:max_rules]
    nodes = [_blank_node(box, kind) for _, box, kind in found]
    boxes = [(i.bbox.x, i.bbox.y, i.bbox.x + i.bbox.width, i.bbox.y + i.bbox.height) for i in lp_images]
    boxes = [b for b in boxes if b[2] - b[0] >= 8 and b[3] - b[1] >= 8][:max_images]
    return nodes + [_blank_node(b, KIND_IMAGE) for b in boxes]


def extract_page_liteparse(page, lp_page, lp_images=(), with_graphics=True):
    """Come extract.extract_page, con LiteParse al posto di PyMuPDF. page serve solo per togliere la rotazione,
    così i riquadri disegnati poi sulla pagina cadono nello stesso sistema di coordinate."""
    if page.rotation:
        page.remove_rotation()
    nodes = nodes_from_textdict(textdict(lp_page))
    if with_graphics:
        nodes += graphic_nodes(lp_page, lp_images)
    return pack_nodes(nodes, lp_page.width, lp_page.height)


# ---------- descrizione per l'uscita ----------

def _inside(bbox, x0, y0, x1, y1):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def describe_block(bbox, lp_page):
    """Font del blocco (quello con più caratteri) e filetti che ci cadono dentro."""
    items = [it for it in lp_page.text_items if it.text.strip()
             and _inside(bbox, it.x, it.y, it.x + it.width, it.y + it.height)]
    info = {}
    if items:
        weight = {}
        for it in items:
            key = (it.font_name, round(font_size(it), 1), it.fill_color)
            weight[key] = weight.get(key, 0) + len(it.text)
        name, size, color = max(weight, key=weight.get)
        sample = next(it for it in items if (it.font_name, round(font_size(it), 1), it.fill_color) == (name, size, color))
        bold, italic, mono = font_style(sample)
        info["font"] = {"nome": name or "sconosciuto", "corpo": size, "grassetto": bold, "corsivo": italic, "monospaziato": mono,
                        "colore": f"#{rgb(color):06x}", "quota_caratteri": round(weight[name, size, color] / sum(weight.values()), 2)}
        info["font_usati"] = sorted({it.font_name or "sconosciuto" for it in items})
    inside = [r for r in rules(lp_page) if _inside(bbox, *r[1])]
    info["filetti"] = {"orizzontali": sum(r[2] == KIND_RULE_H for r in inside),
                       "verticali": sum(r[2] == KIND_RULE_V for r in inside)}
    return info


def page_vectors(lp_page):
    """Vettori della pagina per il JSON: filetti e forme con area visibile (niente segmenti di lunghezza zero)."""
    vg = lp_page.vector_graphics
    if vg is None:
        return {"filetti": [], "forme": []}
    return {
        "filetti": [{"tipo": "orizzontale" if kind == KIND_RULE_H else "verticale",
                     "bbox": [round(v, 2) for v in box]} for _, box, kind in rules(lp_page)],
        "forme": [{"bbox": [round(sh.bbox[0], 2), round(sh.bbox[1], 2), round(sh.bbox[0] + sh.bbox[2], 2),
                            round(sh.bbox[1] + sh.bbox[3], 2)],
                   "riempimento": f"#{rgb(sh.fill_color):06x}" if sh.fill else None,
                   "contorno": f"#{rgb(sh.stroke_color):06x}" if sh.stroke else None, "curve": sh.has_curve}
                  for sh in vg.shapes if sh.bbox[2] >= 2 and sh.bbox[3] >= 2],
    }
