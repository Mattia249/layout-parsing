"""Estrazione con LiteParse (run-llama/liteparse): solo il testo della pagina, come fa l'OCR.

LiteParse qui legge soltanto le parole e la loro posizione, con il font usato. Non estrae vettori né immagini
e non fa nessuna analisi di layout (blocchi, tabelle, liste): quella la fa il nostro modello. L'uscita è la
stessa di ocr.py, cioè i nodi di extract.py. Le coordinate di LiteParse sono già quelle di PyMuPDF: punti,
origine in alto a sinistra.

Tre cose verificate sui dati, che spiegano le scelte qui sotto:
- il corpo del font giusto è font_height; font_size a volte vale 1.0 (font scalati dalla matrice del testo);
- font_weight non è affidabile (CMR10, un tondo, ha peso 744; CMBX10, il neretto, 836): grassetto e corsivo
  si ricavano dal nome del font, e il peso si usa solo se il nome non dice niente;
- un TextItem può essere una parola sola: le righe vanno ricostruite raggruppando gli item per altezza.
"""
import re

import numpy as np

from .extract import graphic_nodes, nodes_from_textdict, pack_nodes

_BOLD = re.compile(r"bold|black|heavy|semibold|demibold|extrabold|ultrabold|cmbx|[-,]bd\b|[-,]b$", re.I)
_LIGHT = re.compile(r"light|thin|regular|roman|book|medium|\bcmr\d|cmmi", re.I)
_ITALIC = re.compile(r"italic|oblique|slant|[-,]it\b|[-,]i$|cmti|cmmi|cmsl", re.I)
_MONO = re.compile(r"mono|courier|consol|typewriter|code|cmtt|xtt\b|\btt\d", re.I)


def make_parser():
    """Solo testo: niente OCR (lo fa ocr.py), vettori, immagini o blocchi di layout di LiteParse."""
    from liteparse import LiteParse

    return LiteParse(quiet=True, ocr_enabled=False, extract_text_metadata=True, extract_vector_graphics=False,
                     extract_images=False, extract_blocks=False, continue_on_page_error=True)


def parse_document(path):
    """Tutto il documento in una volta (LiteParse lavora per file): ritorna le pagine."""
    return make_parser().parse(str(path)).pages


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


def extract_page_liteparse(page, lp_page, with_graphics=True):
    """Come ocr.extract_page_ocr, con il testo letto da LiteParse invece che dall'OCR. page serve per togliere la
    rotazione, così i riquadri disegnati poi cadono nello stesso sistema di coordinate, e, come per l'OCR, per i
    nodi grafici quando il modello li usa."""
    if page.rotation:
        page.remove_rotation()
    nodes = nodes_from_textdict(textdict(lp_page))
    if with_graphics:
        nodes += graphic_nodes(page)
    return pack_nodes(nodes, lp_page.width, lp_page.height)
