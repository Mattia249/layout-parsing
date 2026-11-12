"""OCR con RapidOCR: dalla pagina renderizzata agli stessi nodi dell'estrazione nativa (extract.py).

RapidOCR restituisce righe di testo (box a quattro vertici, in pixel) e, dentro ogni riga, le parole con il loro
box. Ogni parola diventa uno "span" nel formato di page.get_text("dict"): così nodes_from_textdict spezza le
righe sui grandi spazi orizzontali esattamente come fa con i PDF nativi ("2." lontano da "Introduzione" -> due nodi).

Differenze rispetto ai PDF nativi, di cui tenere conto leggendo i risultati:
- non c'è informazione di stile: grassetto, corsivo, monospaziato e colore valgono sempre 0;
- il corpo del font è stimato dall'altezza della riga, e il "font" di un nodo è solo il suo corpo arrotondato,
  quindi il rango del font distingue le righe per dimensione, non per carattere tipografico.
"""
import numpy as np

from .extract import graphic_nodes, nodes_from_textdict, pack_nodes

LINE_TO_SIZE = 1.25     # altezza del box di una riga OCR / corpo del font (stima)


def make_engine(use_cuda=False):
    """Motore OCR con il riconoscitore per alfabeto latino (italiano, inglese, ...).
    use_cuda: rilevamento delle righe sulla GPU (PyTorch). Il riconoscimento resta su CPU, perché il modello latino
    esiste solo per onnxruntime. Misura su 3 pagine a 200 dpi: 3,0 s per pagina tutto su CPU, 1,5 s così."""
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

    params = {
        "Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
        "Global.log_level": "error",
        "Global.max_side_len": 4000,       # di default le immagini più grandi di 2000 px vengono rimpicciolite
    }
    if use_cuda:
        for task in ("Det", "Cls"):
            params[f"{task}.engine_type"] = EngineType.TORCH
        params["EngineConfig.torch.use_cuda"] = True
    return RapidOCR(params=params)


def _box(quad, scale):
    q = np.asarray(quad, np.float32) * scale
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def _spans(text, words, line_box, size, scale):
    """Parole di una riga -> span. Lo spazio dopo una parola si mette solo se c'è anche nel testo della riga."""
    spans, pos = [], 0
    for k, (word, _score, quad) in enumerate(words):
        start = text.find(word, pos)
        pos = start + len(word) if start >= 0 else pos
        space = k < len(words) - 1 and (start < 0 or text[pos:pos + 1] == " ")
        x0, _, x1, _ = _box(quad, scale)
        spans.append({"text": word + (" " if space else ""),
                      "bbox": (x0, line_box[1], x1, line_box[3]),    # altezza della riga: le parole sono allineate
                      "size": size, "flags": 0, "font": "ocr", "color": 0})
    return spans


def ocr_textdict(page, engine, dpi=200):
    """Renderizza la pagina, fa l'OCR e restituisce un dizionario nel formato di page.get_text("dict")."""
    import pymupdf

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
    result = engine(img, return_word_box=True)
    if result.boxes is None or len(result.txts) == 0:
        return {"blocks": []}
    scale = 72.0 / dpi                                       # da pixel a punti
    words_per_line = result.word_results or [()] * len(result.txts)
    lines = []
    for quad, text, words in zip(result.boxes, result.txts, words_per_line):
        if not text.strip():
            continue
        line_box = _box(quad, scale)
        q = np.asarray(quad, np.float32)
        dx, dy = q[1] - q[0]                                 # primo lato del box: direzione della scrittura
        norm = max(float(np.hypot(dx, dy)), 1e-6)
        size = float(round((line_box[3] - line_box[1]) / LINE_TO_SIZE))   # arrotondato: righe simili, stesso "font"
        if not words:
            words = [(text, 1.0, quad)]
        lines.append({"dir": (float(dx) / norm, float(dy) / norm),
                      "spans": _spans(text, words, line_box, max(size, 1.0), scale)})
    return {"blocks": [{"type": 0, "lines": lines}]}


def extract_page_ocr(page, engine, dpi=200, with_graphics=True):
    """Come extract.extract_page, ma il testo viene dall'OCR invece che dal livello di testo del PDF."""
    if page.rotation:
        page.remove_rotation()
    nodes = nodes_from_textdict(ocr_textdict(page, engine, dpi))
    if with_graphics:
        nodes += graphic_nodes(page)
    return pack_nodes(nodes, page.rect.width, page.rect.height)
