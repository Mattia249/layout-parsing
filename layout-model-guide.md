# Modello di layout da testo e coordinate: guida all'implementazione (v1)

Versione del 12 novembre 2026. Tutto il codice necessario è in questo file, nei blocchi che iniziano con `# file: ...`.

## Stato di questo documento

Il codice è stato verificato a tre livelli diversi. Li tengo distinti perché non valgono uguale.

**Eseguito davvero, su pagine sintetiche.** Etichette, feature, coppie candidate, blocchi, metriche, lettura e scrittura della cache, le funzioni di estrazione che non toccano il PDF, la parte dati di `smoke_test.py` e `baseline.py`. Qui il codice gira senza errori; durante le prove è emerso ed è stato corretto un errore sulle pagine senza testo.

**Eseguito solo contro sostituti.** Nell'ambiente in cui ho lavorato PyTorch, PyMuPDF e PyArrow non sono installati e non ho potuto installarli. Per `layout/data.py`, `layout/model.py`, `train.py`, `infer.py`, `check_data.py`, `build_cache.py` e la funzione `extract_page` ho quindi scritto dei sostituti minimi di quelle librerie e ho eseguito il codice contro di essi. Questo verifica la logica: le forme dei tensori, gli indici, le maschere, il flusso degli script dall'inizio alla fine, il conteggio dei parametri, e due proprietà del modello (il padding non altera il risultato di una pagina; l'ordine dei nodi non altera il risultato). Non verifica che ogni chiamata alle librerie vere sia scritta correttamente, né che il modello impari: i sostituti non calcolano gradienti.

**Non verificato.** Niente è stato provato su DocLayNet 1.2 né su un PDF vero. Non ho quindi nessun numero di qualità, e gli unici tempi misurati sono quelli del calcolo delle feature (sezione 15).

Conseguenza pratica: i primi due comandi della sezione 4 servono a verificare sul tuo computer, in pochi minuti e senza dataset, proprio le parti che io non ho potuto eseguire. Se uno dei due dà errore, il messaggio completo basta per correggerlo.

## 1. Requisiti

Sono quelli che hai posto nella conversazione, numerati per poterli richiamare.

| # | Requisito | Dove viene soddisfatto |
|---|---|---|
| R1 | L'ingresso è solo ciò che si estrae dal PDF: testo, coordinate (bbox), font. Nessuna immagine nel percorso principale. | Sezioni 7 e 9 |
| R2 | Più veloce di Docling, con un modello piccolo. | Sezione 13 (circa 9,3 milioni di parametri), sezione 15 (misura dei tempi) |
| R3 | Preciso, cioè misurato e confrontato. | Sezioni 11, 12 e 14 |
| R4 | La bbox entra come vettore, il testo come token. | Sezione 9 |
| R5a | Dire "questo è un titolo". | Testa di classe, sezione 13 |
| R5b | Dire "questi paragrafi formano una tabella". | Testa di link e blocchi, sezioni 12 e 13 |
| R5c | Dire "questo '2.' va unito al nome della sezione". | Estrazione e testa di link, sezioni 7 e 13 |
| R6 | Addestrabile da te su DocLayNet 1.2. | Sezioni 6, 8, 10 e 14 |
| R7 | Visione solo dove si sospettano tabelle o immagini. | Sezione 17 (passo successivo, non nel codice della v1) |

**Cosa la v1 non fa.** Lo dico subito perché DocLayNet non contiene le etichette necessarie:

- non ricostruisce righe e colonne dentro una tabella: dice solo "questi nodi sono la stessa tabella";
- non assegna il livello ai titoli (capitolo, sezione, sottosezione);
- non calcola l'ordine di lettura su pagine a più colonne: i blocchi escono ordinati dall'alto in basso;
- non lavora su scansioni: senza livello di testo non ci sono nodi.

## 2. Assunzioni

Sono le cose che finora davo per scontate. Ognuna ha un modo per verificarla.

| # | Assunzione | Se è falsa | Come verificarla |
|---|---|---|---|
| A1 | I tuoi PDF sono nativi, con testo selezionabile. | Serve un OCR prima di tutto il resto. | `infer.py` stampa quanti nodi trova per pagina: zero nodi significa pagina senza testo. |
| A2 | I tuoi documenti sono soprattutto in italiano. DocLayNet è quasi tutto in inglese (dato che ricordo, non verificato). | Nessun problema se sono in inglese. Se sono in italiano, gli indizi lessicali ("Table", "Figure") non si trasferiscono: vedi sezione 9 (pagine senza testo) e sezione 17, punto 1. | Guarda le immagini prodotte da `check_data.py`. |
| A3 | Lavori su Windows con Python 3.11 o successivo. Una GPU NVIDIA accelera molto il training, ma non è obbligatoria. | Su sola CPU il training completo può richiedere giorni: riduci il dataset (sezione 14.2). | `python -c "import torch; print(torch.cuda.is_available())"` |
| A4 | In DocLayNet 1.2 i box sono `[x, y, larghezza, altezza]` in uno spazio 1025×1025 e le classi vanno da 1 a 11. | Le etichette risulterebbero spostate e il modello imparerebbe rumore. | `check_data.py` prova quattro convenzioni possibili e stampa quale combacia con il testo. |
| A5 | Ogni riga del dataset contiene il PDF di una sola pagina nella colonna `pdf`. | Lo script di cache si fermerebbe con un errore. | `check_data.py` stampa le colonne. |
| A6 | In DocLayNet il numero di sezione ("2.") sta nello stesso box del titolo della sezione. | Il requisito R5c non sarebbe supervisionato e servirebbero etichette tue. | Guarda le immagini di `check_data.py`: il box rosso dell'intestazione deve includere il numero. |
| A7 | Poche pagine superano 512 nodi. | In training quelle pagine vengono viste a finestre di 512 nodi. | `check_data.py` stampa mediana, 95° percentile e massimo. |
| A8 | PyMuPDF si può usare nel tuo contesto. È distribuito con licenza AGPL oppure commerciale (da verificare sul loro sito). | Per un prodotto chiuso serve la licenza commerciale o un altro estrattore. | Sito di PyMuPDF. |

## 3. Concetti minimi

Termini che ho usato senza spiegarli. Bastano queste definizioni per seguire il resto.

- **Nodo.** L'unità su cui il modello decide: un segmento di riga di testo con la sua bbox e il suo font.
- **Feature.** Un numero che descrive un nodo (per esempio "quota di cifre nel testo"). Un nodo è descritto da un vettore di feature.
- **Embedding.** Una tabella che associa a ogni identificatore (una parola, un intervallo di distanza) un vettore di numeri che il modello impara. È il modo per far entrare in una rete cose che non sono numeri.
- **Hashing del testo.** Invece di un vocabolario, ogni parola viene trasformata in un numero tra 1 e 65.535 con una funzione fissa. Parole diverse possono finire sullo stesso numero ("collisione"): è un compromesso accettato per non dover gestire un vocabolario.
- **Bucket.** Un intervallo. "Distanza tra 0,4 e 0,8 righe" è un bucket. Trasformare un numero continuo nel numero del suo bucket permette di usare un embedding.
- **Transformer e attenzione.** Una rete in cui ogni nodo aggiorna il proprio vettore guardando tutti gli altri nodi della pagina e pesandoli. I pesi si chiamano attenzione. Un "layer" è un giro di aggiornamento; qui ce ne sono 6.
- **Bias di attenzione.** Un numero sommato al peso tra due nodi prima che venga calcolato. Qui dipende dalla geometria della coppia: così "sono allineati a sinistra" influenza direttamente quanto due nodi si guardano.
- **Testa.** L'ultimo pezzo della rete, che trasforma i vettori in una risposta. Qui ce ne sono due: classe del nodo e link tra due nodi.
- **Logit.** Il punteggio grezzo che esce da una testa, prima di essere trasformato in probabilità.
- **Loss.** Il numero che misura quanto il modello sbaglia sugli esempi di training. Il training consiste nel farlo scendere.
- **Cross-entropy.** La loss usata quando la risposta è una classe tra tante.
- **BCE (binary cross-entropy).** La loss usata quando la risposta è sì o no, come per i link.
- **Label smoothing.** Si dice al modello che l'etichetta è giusta al 90% invece che al 100%. Serve quando le etichette contengono errori, per non insegnargli una sicurezza eccessiva.
- **Batch, step, epoca.** Un batch è un gruppo di pagine elaborate insieme. Uno step è un aggiornamento dei pesi dopo un batch. Un'epoca è un passaggio completo su tutto il training set.
- **Learning rate, warmup, cosine.** Il learning rate è l'ampiezza di ogni aggiornamento. Il warmup lo fa crescere gradualmente nei primi step, il cosine lo fa scendere dolcemente fino a zero.
- **Padding e maschera.** Le pagine di un batch hanno un numero diverso di nodi: si riempiono le più corte con nodi finti (padding) e una maschera dice quali sono veri.
- **Train, validation, test.** Tre insiemi separati di pagine. Sul train si impara, sulla validation si prendono le decisioni (quale modello tenere, quale soglia), il test si guarda una volta sola alla fine.
- **Overfitting.** Il modello impara a memoria il training set: la loss di training scende, i risultati sulla validation peggiorano.
- **Precisione, recall, F1.** Precisione: quante delle cose che il modello ha dichiarato sono giuste. Recall: quante delle cose vere ha trovato. F1: la loro media armonica. **Macro-F1**: la media degli F1 delle singole classi, in cui ogni classe pesa uguale anche se è rara.
- **Baseline.** Un metodo semplice che dà il numero da battere. Se la rete non lo batte, c'è un errore o la rete non serve.
- **Ablation.** Togliere un pezzo (il testo, il bias geometrico) e rimisurare, per capire quanto contava.
- **kNN.** I k vicini più prossimi. Qui: per ogni nodo, i 12 nodi più vicini sulla pagina.
- **Union-find.** Un algoritmo che, data una lista di coppie "questi due stanno insieme", restituisce i gruppi.
- **Fine-tuning.** Continuare il training di un modello già addestrato su dati nuovi, di solito pochi e più vicini al caso reale.
- **Teacher e pseudo-etichette.** Usare un modello lento ma buono (Docling) per etichettare automaticamente i tuoi documenti, e addestrare il modello veloce su quelle etichette.
- **Checkpoint.** Il file con i pesi del modello salvati.

## 4. Panoramica

```
PDF ──► estrazione ──► nodi (testo, bbox, font)
                         │
                         ├─► feature di nodo  (138 numeri + 32 id di testo)
                         ├─► feature di coppia (9 bucket per ogni coppia di nodi)
                         └─► coppie candidate  (i vicini di ogni nodo)
                                   │
                              transformer (6 layer)
                              ├─► testa di classe: una classe per nodo
                              └─► testa di link:   "stesso blocco?" per ogni coppia candidata
                                   │
                              union-find ──► blocchi con classe, bbox e testo
```

Per il training serve in più l'etichetta di ogni nodo, ricavata dai box annotati di DocLayNet (sezione 8).

Struttura della cartella di progetto:

```
progetto/
  layout/
    __init__.py
    config.py      costanti e iperparametri
    extract.py     dal PDF ai nodi
    targets.py     dai box di DocLayNet alle etichette dei nodi
    features.py    feature, coppie candidate, lettura e scrittura della cache
    decode.py      dai link ai blocchi, metriche
    data.py        caricamento per PyTorch
    model.py       la rete
  smoke_test.py    prova a secco su pagine sintetiche
  check_data.py    primo contatto con DocLayNet
  build_cache.py   estrazione di tutto il dataset, una volta sola
  baseline.py      il numero da battere
  train.py         addestramento
  infer.py         uso su un tuo PDF
```

Ordine dei comandi, dall'inizio alla fine:

```
python extract_code.py layout-model-guide.md        (crea i file, vedi appendice)
python smoke_test.py cache_fake                     (verifica l'installazione)
python train.py cache_fake runs/prova 3            (verifica il training, pochi minuti)
python check_data.py D:/AI/datasets/DocLayNet-v1.2/data
python build_cache.py D:/AI/datasets/DocLayNet-v1.2/data D:/AI/cache xywh 1025
python baseline.py D:/AI/cache
python train.py D:/AI/cache runs/v1
python infer.py runs/v1/best.pt documento.pdf
```

## 5. Ambiente

```
python -m venv .venv
.venv\Scripts\activate
pip install numpy pyarrow pymupdf scikit-learn
```

PyTorch va installato a parte, perché il comando dipende dalla tua scheda video: prendilo dal selettore su pytorch.org (scegli Windows, Pip, e la versione CUDA se hai una GPU NVIDIA).

Versioni: il codice usa `import pymupdf`, che richiede PyMuPDF 1.24.3 o successivo, e PyTorch 2.x.

Costanti e iperparametri stanno tutti in un file. Gli iperparametri sono le scelte che non vengono imparate dal modello ma fissate da te; i valori qui sono punti di partenza ragionevoli, non valori ottimi.

```python
# file: layout/__init__.py
# Vuoto di proposito: serve solo a far riconoscere "layout" come pacchetto Python.
```

```python
# file: layout/config.py
from dataclasses import dataclass

# Le 11 classi di DocLayNet (id 1..11 nel dataset, 0..10 qui) più "other" per i nodi senza annotazione.
CLASSES = ["caption", "footnote", "formula", "list_item", "page_footer", "page_header",
           "picture", "section_header", "table", "text", "title", "other"]
OTHER = 11

# Tipi di nodo.
KIND_TEXT, KIND_RULE_H, KIND_RULE_V, KIND_IMAGE = 0, 1, 2, 3

VOCAB = 65536        # bucket dell'hashing del testo; lo 0 è riservato al padding
TOKENS = 32          # id di testo per nodo
REL_FEATS = 9        # feature di coppia
REL_BUCKETS = 25     # bucket per ogni feature di coppia
N_SHAPE = 14         # feature sulla forma del testo
N_FOURIER = 8        # frequenze usate per codificare le coordinate


@dataclass
class Config:
    # modello
    d_model: int = 256          # lunghezza del vettore di ogni nodo
    n_heads: int = 8            # teste di attenzione per layer
    n_layers: int = 6           # layer del transformer
    d_ff: int = 1024            # larghezza dello strato interno di ogni layer
    d_tok: int = 64             # lunghezza dell'embedding di una parola
    d_pair: int = 32            # lunghezza dell'embedding di coppia nella testa di link
    dropout: float = 0.1        # quota di valori azzerati a caso in training (contro l'overfitting)
    # dati
    knn: int = 12               # vicini considerati per i link
    use_graphics: bool = False  # True: anche filetti e immagini diventano nodi (sezione 17)
    max_nodes_train: int = 512  # oltre, la pagina viene vista a finestre
    max_nodes_eval: int = 1024
    text_dropout: float = 0.1   # quota di pagine viste senza testo in training
    # training
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01  # spinge i pesi verso zero (contro l'overfitting)
    warmup_steps: int = 1000
    epochs: int = 30
    label_smoothing: float = 0.1
    link_weight: float = 1.0    # peso della loss dei link rispetto a quella delle classi
    link_threshold: float = 0.5 # probabilità oltre la quale due nodi vengono uniti
    seed: int = 0
```

## 6. Guardare i dati

Prima di scrivere il modello bisogna verificare le assunzioni A4, A5, A6 e A7. Questo script apre il primo file di training, estrae 200 pagine e stampa quattro cose.

1. **Le colonne.** Devono esserci `bboxes`, `category_id`, `metadata` e `pdf`.
2. **La copertura.** È la quota di nodi di testo che cadono per almeno metà dentro un box annotato. Viene calcolata con quattro ipotesi su come sono scritti i box: formato `[x, y, larghezza, altezza]` oppure `[x0, y0, x1, y1]`, nello spazio 1025×1025 oppure già in punti. Mi aspetto che quella giusta stia sopra il 90% e le altre molto più in basso. Lo script stampa il comando da usare nella sezione 10 con l'ipotesi migliore.
3. **Gli id delle classi.** Devono andare da 1 a 11.
4. **I nodi per pagina.** Servono a decidere `max_nodes_train`.

Salva anche cinque immagini `check_NNN.png`: in rosso i box annotati, in blu i nodi estratti. Aprile: i box blu devono stare dentro quelli rossi, e l'intestazione di sezione deve contenere anche il numero (assunzione A6). Se nessuna ipotesi supera il 90%, le immagini mostrano di quanto e in che direzione i box sono spostati.

Lo script usa `extract.py` e `targets.py`, definiti nelle due sezioni successive.

```python
# file: check_data.py
"""Primo contatto con il dataset: colonne, convenzione dei box, statistiche, immagini di controllo.
Uso: python check_data.py D:/AI/datasets/DocLayNet-v1.2/data"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pymupdf

from layout.config import KIND_TEXT
from layout.extract import extract_page
from layout.targets import assign, page_annotations


def main(data_dir, n_pages=200, n_images=5):
    path = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))[0]
    f = pq.ParquetFile(path)
    print("Colonne:", f.schema_arrow.names)
    # le quattro ipotesi su come sono scritti i box: formato x spazio
    cover = {(fmt, space): [] for fmt in ("xywh", "xyxy") for space in ("1025", "points")}
    n_nodes, ids, samples, no_text, done = [], [], [], 0, 0
    for batch in f.iter_batches(batch_size=16, columns=["bboxes", "category_id", "metadata", "pdf"]):
        for row in batch.to_pylist():
            if done == n_pages:
                break
            done += 1
            ids += list(row["category_id"])
            try:
                doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
                raw = extract_page(doc[0])
                doc.close()
            except Exception as error:
                print(f"Pagina {done} illeggibile: {error}")
                continue
            text_box = raw["box"][raw["kind"] == KIND_TEXT]
            if len(text_box) == 0:
                no_text += 1
                continue
            n_nodes.append(len(text_box))
            for fmt, space in cover:
                ann = page_annotations(row, raw["page_w"], raw["page_h"], fmt, space)
                _, block = assign(text_box, ann, np.zeros(len(ann), np.int64))
                cover[fmt, space].append((block >= 0).mean())
            if len(samples) < n_images:
                samples.append((done, row, raw, text_box))
        if done == n_pages:
            break

    print(f"Pagine esaminate: {done}, senza testo: {no_text}")
    for (fmt, space), values in cover.items():
        print(f"Copertura con box {fmt} nello spazio {space}: {np.mean(values):.3f}")
    fmt, space = max(cover, key=lambda key: np.mean(cover[key]))
    print(f"Ipotesi migliore: {fmt} {space} (attesa: xywh 1025, con copertura sopra 0.9).")
    print(f"Comando: python build_cache.py CARTELLA_DATI CARTELLA_CACHE {fmt} {space}")
    print(f"Id delle classi: da {min(ids)} a {max(ids)} (atteso: da 1 a 11)")
    print(f"Nodi di testo per pagina: mediana {np.median(n_nodes):.0f}, "
          f"95° percentile {np.percentile(n_nodes, 95):.0f}, massimo {max(n_nodes)}")

    for number, row, raw, text_box in samples:               # immagini di controllo con l'ipotesi migliore
        doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
        page = doc[0]
        for b in page_annotations(row, raw["page_w"], raw["page_h"], fmt, space):
            page.draw_rect(pymupdf.Rect([float(v) for v in b]), color=(1, 0, 0), width=1)
        for b in text_box:
            page.draw_rect(pymupdf.Rect([float(v) for v in b]), color=(0, 0, 1), width=0.3)
        page.get_pixmap(dpi=110).save(f"check_{number:03d}.png")
        doc.close()


if __name__ == "__main__":
    main(sys.argv[1])
```

## 7. Estrazione dal PDF

**Che cosa restituisce PyMuPDF.** `page.get_text("dict")` restituisce la pagina come blocchi, ogni blocco come righe, ogni riga come *span*. Uno span è un tratto di testo con lo stesso font: una riga con una parola in grassetto in mezzo ha tre span. Ogni span ha testo, bbox, nome del font, corpo (`size`), colore e un numero `flags` i cui bit dicono se è apice (1), corsivo (2), monospaziato (8), grassetto (16).

**Come si passa dagli span ai nodi.** Gli span della stessa riga vengono uniti in un nodo, tranne quando tra due span consecutivi c'è uno spazio orizzontale più grande del corpo del font (`GAP_EM = 1.0`). In quel caso la riga viene spezzata. È questa regola che produce i nodi separati di cui hai bisogno:

- "2." e "Introduzione", distanziati da una tabulazione, diventano due nodi (che poi la testa di link riunisce: requisito R5c);
- le celle di una riga di tabella diventano un nodo ciascuna.

Un cambio di font dentro la riga non spezza: resta descritto dalle feature di stile (quota di caratteri in grassetto e in corsivo).

Un limite: la regola guarda lo spazio *tra* span. Se un PDF allinea le colonne inserendo caratteri di spazio dentro lo stesso span, le celle restano in un nodo solo. Si riconosce nelle immagini di controllo della sezione 6, dove ogni nodo è un rettangolo blu.

**Stile del nodo.** Corpo e nome del font sono quelli dello span con più caratteri. I nomi dei font non si confrontano tra documenti diversi, quindi a ogni nodo viene assegnato il *rango* del suo font nella pagina: 0 per il font con cui è scritta la maggior parte dei caratteri (di solito il corpo del testo), 1 per il secondo, e così via.

**Ordine dei nodi.** I nodi vengono ordinati per righe visive: dall'alto in basso e, sulla stessa riga, da sinistra a destra. Il modello non usa questo ordine (vedi sezione 13); serve a ricomporre il testo di un blocco nell'ordine giusto.

**Coordinate.** Sono in punti tipografici (1/72 di pollice), con origine in alto a sinistra e y che cresce verso il basso. Una pagina A4 misura 595×842 punti.

**Filetti e immagini.** Lo script estrae anche le linee sottili (tipiche dei bordi di tabella) e le bbox delle immagini raster. La v1 non li usa (`use_graphics = False`), ma averli già nella cache evita di rifare l'estrazione quando serviranno.

```python
# file: layout/extract.py
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
```

## 8. Etichette

DocLayNet non etichetta i nodi: etichetta regioni rettangolari. Bisogna quindi proiettare le regioni sui nodi.

**Conversione delle coordinate.** I box annotati sono nello spazio dell'immagine 1025×1025, che è la pagina deformata in un quadrato. Per riportarli in punti si moltiplica x per `larghezza_pagina / 1025` e y per `altezza_pagina / 1025`: due fattori diversi, perché la pagina non è quadrata.

**Classe del nodo.** Ogni nodo prende la classe del box che ne copre almeno metà dell'area. Se più box lo coprono allo stesso modo vince il più piccolo. Se nessuno lo copre, la classe è `other`.

**Blocco del nodo.** È l'indice del box che lo contiene, oppure -1. Due nodi con lo stesso blocco sono un esempio positivo per la testa di link. Da qui nasce tutta la supervisione dei requisiti R5b e R5c, senza annotare nulla a mano:

- le righe di un paragrafo stanno nello stesso box `text`;
- "2." e "Introduzione" stanno nello stesso box `section_header` (assunzione A6);
- tutte le celle di una tabella stanno nello stesso box `table`.

```python
# file: layout/targets.py
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
    space: "1025" se i box sono nello spazio dell'immagine, "points" se sono già in punti della pagina."""
    meta = row["metadata"]
    src_w, src_h = (page_w, page_h) if space == "points" else (meta["coco_width"], meta["coco_height"])
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
```

## 9. Feature

È il cuore del requisito R4. Tutto ciò che il modello vede viene calcolato qui, in numpy, fuori dalla rete: così la rete resta piccola e questa parte si può provare senza PyTorch.

### 9.1 Due unità di misura

Le distanze sono espresse in due modi, per due scopi diversi.

- **Frazione di pagina** (x diviso la larghezza, y diviso l'altezza). Dice *dove* sta un nodo: in alto, a margine, a piè di pagina.
- **Righe** (distanza divisa per l'altezza mediana delle righe di testo della pagina). Dice *quanto* due nodi sono distanti in un'unità che ha senso per il testo: "0,2 righe sotto" è la riga successiva dello stesso paragrafo, "1,5 righe sotto" è probabilmente un altro paragrafo. Così il modello non dipende dal corpo del font del documento.

### 9.2 Feature di nodo (138 numeri)

| Gruppo | Quante | Contenuto |
|---|---|---|
| Posizione | 6 | x0, y0, x1, y1 e centro, in frazione di pagina |
| Dimensione | 5 | larghezza e altezza in frazione di pagina e in righe, rapporto larghezza/altezza della pagina |
| Pagina | 2 | numero di nodi e altezza mediana delle righe |
| Corpo | 1 | logaritmo del rapporto tra il corpo del nodo e il corpo mediano della pagina |
| Stile | 5 | quota in grassetto, quota in corsivo, monospaziato, colore non nero, apice |
| Rango del font | 4 | il font è il 1°, 2°, 3° o oltre il 3° più usato nella pagina |
| Forma del testo | 14 | lunghezza, numero di parole, quota di cifre, maiuscole, lettere, punteggiatura; inizia con una numerazione, un'enumerazione, un punto elenco; è solo una numerazione; è solo un numero; finisce con un punto o con due punti; inizia con maiuscola |
| Tipo | 5 | testo, filetto orizzontale, filetto verticale, immagine; testo verticale |
| Coordinate di Fourier | 96 | seni e coseni delle 6 coordinate di posizione a 8 frequenze |

Tre scelte che meritano una spiegazione.

- **Tutti i valori sono dell'ordine di 1.** Una rete impara male se una feature vale 0,01 e un'altra 800. Per questo ci sono logaritmi, divisioni e limiti.
- **Il corpo è relativo.** "14 punti" non dice niente; "1,4 volte il corpo del testo" dice "probabilmente un titolo".
- **Le coordinate di Fourier.** Una rete fatica a distinguere 0,120 da 0,125 se riceve il numero così com'è. Passando ogni coordinata attraverso seni e coseni a frequenze crescenti, piccole differenze di posizione diventano differenze grandi in almeno una delle componenti.

### 9.3 Testo (32 id per nodo)

Il testo viene messo in minuscolo, le cifre sostituite con "0" e diviso in parole e segni. Si tengono le prime 12 e le ultime 4 parole: la classe di un nodo si capisce quasi sempre da come comincia ("Tabella 3:", "Art. 12", "•") e a volte da come finisce. Ogni parola produce due id tramite hashing: uno per la parola intera e uno per i suoi primi 4 caratteri, così "tabella" e "tabelle" condividono un id.

Rispetto al requisito R4: il testo è tokenizzato, ma i token di un nodo vengono fusi in un solo vettore (la media dei loro embedding). Il modello non legge la pagina parola per parola. È la scelta che lo rende piccolo e veloce; il costo è che non può fare ragionamenti sul contenuto delle frasi.

L'hashing usa `zlib.crc32` e non la funzione `hash()` di Python, che cambia a ogni avvio del programma e renderebbe inutilizzabile un modello salvato.

### 9.4 Feature di coppia (9 bucket)

Per ogni coppia di nodi (i, j) si calcolano, in righe:

| # | Feature | Che cosa segnala |
|---|---|---|
| 0 | differenza tra i bordi sinistri | allineamento a sinistra: stesso paragrafo, stessa colonna |
| 1 | differenza tra i bordi destri | allineamento a destra: colonne di numeri |
| 2 | differenza tra i centri orizzontali | testo centrato |
| 3 | differenza tra i bordi superiori | stessa riga di tabella |
| 4 | differenza tra i bordi inferiori | stessa linea di base |
| 5 | spazio orizzontale tra i due box | celle affiancate, colonne |
| 6 | spazio verticale tra i due box | interlinea, stacco tra paragrafi |
| 7 | logaritmo del rapporto tra i corpi | titolo e testo |
| 8 | stesso font (0 no, 1 sì, 2 se uno dei due non è testo) | |

Le prime otto vengono trasformate in uno di 25 bucket. Il bucket centrale (12) significa "differenza sotto 0,05 righe", cioè allineati. Poi gli intervalli raddoppiano a ogni passo (0,05–0,1, 0,1–0,2, 0,2–0,4, ...), con 12 bucket per i valori positivi e 12 per i negativi. La risoluzione è fine dove serve, vicino allo zero, e grossolana per le distanze grandi.

### 9.5 Coppie candidate

La testa di link non può valutare tutte le coppie: sarebbero decine di migliaia per pagina e quasi tutte ovviamente negative. Valuta, per ogni nodo:

- i 12 nodi più vicini (distanza tra i bordi dei box);
- il primo nodo che incontra in ciascuna delle quattro direzioni (sopra, sotto, sinistra, destra).

Le quattro direzioni servono per le tabelle larghe, dove la cella accanto può essere più lontana di 12 altre celle.

Una condizione da verificare: i candidati devono bastare a ricostruire ogni blocco vero. `smoke_test.py` lo controlla unendo i nodi con i link veri e confrontando i blocchi ottenuti con quelli veri.

```python
# file: layout/features.py
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
```

**Le due modifiche fatte ai dati in training** (`train=True`) si chiamano *augmentation* e servono a rendere il modello meno fragile:

- un rumore di 0,3 punti sulle coordinate, perché estrattori e PDF diversi non danno bbox identiche;
- una pagina su dieci viene vista senza testo. Costringe il modello a saper decidere da geometria e font, che è ciò che si trasferisce dall'inglese all'italiano (assunzione A2).

## 10. Cache

L'estrazione dai PDF è la parte lenta e non cambia mai: va fatta una volta e salvata. Lo script scrive un file `.npz` per pagina, con dentro gli array della pagina e le sue etichette.

Quattro comportamenti da conoscere.

- **Si può interrompere e rilanciare**: salta le pagine già scritte.
- **Le pagine senza nodi di testo vengono scartate**: il modello non avrebbe nulla su cui lavorare.
- **I PDF illeggibili vengono saltati e contati**: uno solo non ferma tutto il lavoro.
- **Non carica la colonna delle immagini**: legge solo le quattro colonne che servono, quindi usa poca memoria.

Il terzo e il quarto argomento sono il formato e lo spazio dei box: usa il comando che `check_data.py` stampa. Non ho misurato il tempo: dipende dal disco e dal processore. Lo script stampa l'avanzamento ogni 500 pagine, da cui puoi stimarlo.

```python
# file: build_cache.py
"""Estrae tutte le pagine del dataset e le salva con le etichette. Si può interrompere e rilanciare.
Uso: python build_cache.py D:/AI/datasets/DocLayNet-v1.2/data D:/AI/cache [xywh|xyxy] [1025|points]"""
import glob
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import pymupdf

from layout.config import KIND_TEXT
from layout.extract import extract_page
from layout.features import add_text_feats, save_raw
from layout.targets import assign, page_annotations

COLUMNS = ["bboxes", "category_id", "metadata", "pdf"]


def build_split(data_dir, out_dir, split, fmt, space):
    files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not files:
        print(f"Nessun file per lo split '{split}' in {data_dir}: lo salto.")
        return
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    written = skipped = failed = 0
    start = time.time()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=COLUMNS):
            for row in batch.to_pylist():
                meta = row["metadata"]
                dst = os.path.join(out_dir, split, meta["page_hash"] + ".npz")
                if os.path.exists(dst):
                    continue
                try:                                         # un PDF illeggibile non deve fermare ore di lavoro
                    doc = pymupdf.open(stream=row["pdf"], filetype="pdf")
                    raw = add_text_feats(extract_page(doc[0]))
                    doc.close()
                except Exception as error:
                    failed += 1
                    print(f"Pagina {meta['page_hash']} saltata: {error}")
                    continue
                if not (raw["kind"] == KIND_TEXT).any():
                    skipped += 1
                    continue
                ann = page_annotations(row, raw["page_w"], raw["page_h"], fmt, space)
                cls = np.asarray(row["category_id"], np.int64) - 1          # da 1..11 a 0..10
                raw["label"], raw["block"] = assign(raw["box"], ann, cls)
                save_raw(raw, dst)
                written += 1
                if written % 500 == 0:
                    print(f"{split}: {written} pagine scritte in {time.time() - start:.0f} s")
    print(f"{split}: finito. Scritte {written}, senza testo {skipped}, illeggibili {failed}.")


if __name__ == "__main__":
    fmt = sys.argv[3] if len(sys.argv) > 3 else "xywh"
    space = sys.argv[4] if len(sys.argv) > 4 else "1025"
    for split in ("train", "validation", "test"):
        build_split(sys.argv[1], sys.argv[2], split, fmt, space)
```

## 11. Baseline

Prima della rete, un metodo semplice: alberi di decisione (gradient boosting di scikit-learn) che classificano ogni nodo guardando le sue 42 feature di base più la relazione con il nodo precedente e con il successivo. Non usa il testo e non raggruppa i nodi.

Serve a tre cose.

- **Dà il numero da battere** per la testa di classe. Se la rete non lo supera, c'è un errore nella rete.
- **Verifica le etichette**: se anche la baseline dà risultati casuali, il problema è nelle sezioni 7 e 8, non nel modello.
- **Può bastare**: se i tuoi documenti sono omogenei e ti serve solo la classe, è già una soluzione velocissima.

Si addestra in pochi minuti su 3.000 pagine.

```python
# file: baseline.py
"""Baseline senza rete neurale: classifica ogni nodo dalle sue feature e da quelle dei due nodi adiacenti.
Uso: python baseline.py D:/AI/cache"""
import glob
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from layout.config import CLASSES, Config
from layout.decode import f1_per_class
from layout.features import BASE_FEATS, load_raw, make_example


def page_xy(path, cfg):
    ex = make_example(load_raw(path), cfg)
    n, r = len(ex["feats"]), ex["rel"].shape[-1]
    prev = np.full((n, r), 12, np.uint8)                     # 12 = "nessuna differenza"
    nxt = prev.copy()
    if n > 1:
        step = ex["rel"][np.arange(n - 1), np.arange(1, n)]  # relazione tra ogni nodo e il successivo
        nxt[:-1] = step
        prev[1:] = step
    return np.concatenate([ex["feats"][:, :BASE_FEATS], prev, nxt], 1), ex["label"]


def load(folder, n_pages, cfg):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    order = np.random.default_rng(0).permutation(len(files))[:n_pages]
    xs, ys = zip(*(page_xy(files[i], cfg) for i in order))
    return np.concatenate(xs), np.concatenate(ys)


def main(cache_dir, n_train=3000, n_val=500):
    cfg = Config()
    x, y = load(os.path.join(cache_dir, "train"), n_train, cfg)
    xv, yv = load(os.path.join(cache_dir, "validation"), n_val, cfg)
    print(f"Nodi di training: {len(y)}, di validation: {len(yv)}")
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1)
    clf.fit(x, y)
    f1, support = f1_per_class(yv, clf.predict(xv), len(CLASSES))
    for name, value, count in zip(CLASSES, f1, support):
        print(f"{name:16s} F1 {value:.3f}   nodi {count}")
    print(f"Macro-F1 (sulle classi presenti): {f1[support > 0].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
```

## 12. Dai link ai blocchi, e metriche

**Dai link ai blocchi.** La testa di link dà una probabilità per ogni coppia candidata. Le coppie sopra la soglia (0,5) vengono unite con union-find, e ogni gruppo risultante è un blocco. C'è un rischio: un solo link sbagliato fonde due blocchi interi. Per limitarlo, se le classi predette dei due nodi sono diverse la coppia viene unita solo con probabilità sopra 0,9.

**Classe del blocco.** Si fa la media dei logit dei suoi nodi e si prende la classe più alta. È più robusto del voto a maggioranza, perché tiene conto di quanto ogni nodo è sicuro.

**Metriche.** Sono tre, e misurano cose diverse.

- **Macro-F1 dei nodi**: la testa di classe sbaglia? Risponde al requisito R5a.
- **F1 dei link**: le singole decisioni "stesso blocco" sono giuste?
- **F1 dei blocchi**: un blocco predetto conta come corretto solo se contiene esattamente gli stessi nodi di un blocco vero e ha la stessa classe. È la metrica più severa e quella che corrisponde a ciò che vuoi (R5b, R5c): un paragrafo spezzato in due, o un "2." rimasto staccato, sono errori.

```python
# file: layout/decode.py
"""Dai link ai blocchi, e metriche."""
import numpy as np


def find_blocks(n, pairs, prob, node_cls=None, thr=0.5, thr_cross=0.9):
    """Union-find: unisce i nodi collegati da un link sopra soglia. Ritorna l'id di blocco di ogni nodo."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j), p in zip(pairs, prob):
        same_cls = node_cls is None or node_cls[i] == node_cls[j]
        if p > (thr if same_cls else thr_cross):             # classi diverse: serve un link molto sicuro
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(a) for a in range(n)], np.int64)
    return np.unique(roots, return_inverse=True)[1].reshape(-1)


def block_classes(block, logits):
    """Classe di ogni blocco: quella con la media dei logit più alta tra i suoi nodi."""
    n_blocks = int(block.max()) + 1
    total = np.zeros((n_blocks, logits.shape[1]))
    np.add.at(total, block, logits)
    return (total / np.bincount(block, minlength=n_blocks)[:, None]).argmax(1)


def _block_sets(block, node_cls):
    groups = {}
    for i, b in enumerate(block):
        groups.setdefault(int(b), []).append(i)
    return {(frozenset(members), int(node_cls[members[0]])) for members in groups.values()}


def block_match(pred_block, pred_cls, gt_block, gt_cls):
    """Ritorna [blocchi corretti, blocchi predetti, blocchi veri]. Corretto = stessi nodi e stessa classe."""
    gt_block = np.asarray(gt_block).copy()
    lone = gt_block < 0                                      # i nodi senza annotazione sono blocchi a sé
    gt_block[lone] = gt_block.max() + 1 + np.arange(lone.sum())
    pred, true = _block_sets(pred_block, pred_cls), _block_sets(gt_block, gt_cls)
    return np.array([len(pred & true), len(pred), len(true)], np.float64)


def f1_per_class(y_true, y_pred, n_classes):
    """F1 di ogni classe e numero di nodi veri di quella classe."""
    f1 = np.zeros(n_classes)
    support = np.zeros(n_classes, np.int64)
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1[c] = 2 * tp / max(2 * tp + fp + fn, 1)
        support[c] = np.sum(y_true == c)
    return f1, support
```

## 13. Il modello

### 13.1 Che cosa entra

Per un batch di B pagine, con N nodi nella pagina più lunga e P coppie candidate:

| Nome | Forma | Contenuto |
|---|---|---|
| `feats` | [B, N, 138] | feature di nodo |
| `tok` | [B, N, 32] | id di testo (0 = vuoto) |
| `rel` | [B, N, N, 9] | bucket delle feature di coppia, per tutte le coppie |
| `mask` | [B, N] | vero per i nodi reali, falso per il padding |
| `pairs` | [B, P, 2] | indici dei due nodi di ogni coppia candidata |
| `pair_rel` | [B, P, 9] | bucket delle feature di coppia, solo per le candidate |

```python
# file: layout/data.py
"""Caricamento delle pagine per PyTorch."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import load_raw, make_example


class PageDataset(Dataset):
    def __init__(self, folder, cfg, train):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"Nessun file .npz in {folder}")
        self.cfg, self.train = cfg, train
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        return make_example(load_raw(self.files[k]), self.cfg, self.train, self.rng)


def collate(batch):
    """Unisce più pagine in un batch, riempiendo le più corte con padding."""
    b_size = len(batch)
    n = max(len(e["feats"]) for e in batch)
    p = max(max(len(e["pairs"]) for e in batch), 1)
    n_feats, n_tok, n_rel = batch[0]["feats"].shape[1], batch[0]["tok"].shape[1], batch[0]["rel"].shape[-1]
    out = {
        "feats": torch.zeros(b_size, n, n_feats),
        "tok": torch.zeros(b_size, n, n_tok, dtype=torch.long),
        "rel": torch.zeros(b_size, n, n, n_rel, dtype=torch.uint8),
        "mask": torch.zeros(b_size, n, dtype=torch.bool),
        "pairs": torch.zeros(b_size, p, 2, dtype=torch.long),
        "pair_rel": torch.zeros(b_size, p, n_rel, dtype=torch.long),
        "pair_mask": torch.zeros(b_size, p, dtype=torch.bool),
        "label": torch.full((b_size, n), -100, dtype=torch.long),    # -100 = "ignora nella loss"
        "link": torch.zeros(b_size, p),
    }
    for b, e in enumerate(batch):
        k, q = len(e["feats"]), len(e["pairs"])
        out["feats"][b, :k] = torch.from_numpy(e["feats"])
        out["tok"][b, :k] = torch.from_numpy(e["tok"])
        out["rel"][b, :k, :k] = torch.from_numpy(np.ascontiguousarray(e["rel"]))
        out["mask"][b, :k] = True
        out["pairs"][b, :q] = torch.from_numpy(e["pairs"])
        out["pair_rel"][b, :q] = torch.from_numpy(e["pair_rel"].astype(np.int64))
        out["pair_mask"][b, :q] = True
        if "label" in e:
            out["label"][b, :k] = torch.from_numpy(e["label"])
            out["link"][b, :q] = torch.from_numpy(e["link"])
    out["block"] = [e.get("block") for e in batch]           # lista di array: serve solo alle metriche
    return out


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
```

### 13.2 Che cosa succede dentro

**1. Da nodo a vettore.** Le 138 feature passano in una piccola rete a due strati che produce un vettore di 256 numeri. I 32 id di testo vengono trasformati ciascuno in un vettore di 64 numeri, se ne fa la media (ignorando gli id vuoti) e il risultato viene portato a 256 numeri. I due vettori si sommano. È qui che bbox e testo diventano "un vettore nello spazio latente", come chiedevi.

**2. Bias geometrico.** Per ogni coppia di nodi, i 9 bucket vengono trasformati in 9 gruppi di 8 numeri (uno per testa di attenzione) e sommati. Il risultato è un numero per ogni coppia e per ogni testa, calcolato una volta e usato da tutti i layer. Ai nodi di padding viene assegnato meno infinito, così nessuno li guarda.

**3. Sei layer.** In ogni layer ogni nodo calcola quanto guardare ciascun altro nodo (somiglianza tra i loro vettori più il bias geometrico), aggiorna il proprio vettore con una media pesata degli altri, e poi lo rielabora con una rete a due strati. "Pre-LN" significa che la normalizzazione dei vettori avviene prima di ciascuna delle due operazioni: rende il training più stabile.

**4. Niente posizione nella sequenza.** Il modello non sa in che ordine gli arrivano i nodi: conosce solo la loro geometria. Di conseguenza l'ordine in cui il PDF elenca il testo, che è inaffidabile, non può influenzare il risultato.

**5. Testa di classe.** Uno strato lineare: dal vettore del nodo a 12 punteggi, uno per classe.

**6. Testa di link.** Per ogni coppia candidata (i, j) riceve: i due vettori, la loro differenza in valore assoluto, il loro prodotto, e un embedding delle 9 feature di coppia. Una rete a due strati produce un punteggio: positivo significa "stesso blocco". In ogni coppia i viene prima di j nell'ordine di lettura, quindi il modello sa sempre quale dei due sta sopra o a sinistra.

### 13.3 Dimensioni

| Parte | Parametri |
|---|---|
| Embedding del testo (65.536 × 64) | 4.194.304 |
| Sei layer (789.760 ciascuno) | 4.738.560 |
| Testa di link | 270.849 |
| Rete delle feature di nodo | 101.376 |
| Tutto il resto | 29.748 |
| **Totale** | **9.334.837** |

`smoke_test.py` stampa il conteggio reale: deve coincidere con questo.

**Loss.** Somma di due termini: cross-entropy sulle classi dei nodi (con label smoothing 0,1, ignorando il padding) e BCE sui link delle coppie candidate.

```python
# file: layout/model.py
"""La rete: nodi -> vettori -> transformer con bias geometrico -> classe dei nodi e link tra coppie."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASSES, REL_BUCKETS, REL_FEATS, VOCAB
from .features import NODE_FEATS


class Block(nn.Module):
    """Un layer del transformer (pre-LN): attenzione tra i nodi, poi rielaborazione di ogni nodo."""

    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):                              # x: [B, N, d]   bias: [B, H, N, N]
        b, n, d = x.shape
        d_head = d // self.heads
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # ciascuno [B, H, N, d_head]
        att = q @ k.transpose(-2, -1) / math.sqrt(d_head) + bias
        att = self.drop(att.softmax(-1))                     # quanto ogni nodo guarda ciascun altro
        y = (att @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.norm2(x)))


class LayoutModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.heads = cfg.n_heads
        self.feat_mlp = nn.Sequential(nn.Linear(NODE_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.tok_emb = nn.Embedding(VOCAB, cfg.d_tok, padding_idx=0)
        self.tok_proj = nn.Linear(cfg.d_tok, d)
        self.in_norm = nn.LayerNorm(d)
        self.rel_bias = nn.Embedding(REL_FEATS * REL_BUCKETS, cfg.n_heads)
        self.layers = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.cls_head = nn.Linear(d, len(CLASSES))
        self.pair_emb = nn.Embedding(REL_FEATS * REL_BUCKETS, cfg.d_pair)
        self.link_head = nn.Sequential(nn.Linear(4 * d + cfg.d_pair, d), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        # Ogni feature di coppia ha i suoi 25 bucket: sommando questi scostamenti stanno tutte in una sola tabella.
        self.register_buffer("offsets", torch.arange(REL_FEATS) * REL_BUCKETS, persistent=False)
        nn.init.zeros_(self.rel_bias.weight)                 # all'inizio la geometria non influenza l'attenzione
        nn.init.normal_(self.pair_emb.weight, std=0.1)

    def forward(self, feats, tok, rel, mask, pairs, pair_rel):
        # 1) da nodo a vettore
        n_tok = (tok != 0).sum(-1, keepdim=True).clamp(min=1)
        text = self.tok_emb(tok).sum(2) / n_tok              # media degli embedding degli id non vuoti
        x = self.in_norm(self.feat_mlp(feats) + self.tok_proj(text))            # [B, N, d]

        # 2) bias geometrico, uguale per tutti i layer
        bias = torch.zeros(rel.shape[0], rel.shape[1], rel.shape[2], self.heads, device=feats.device)
        for f in range(rel.shape[-1]):
            bias = bias + self.rel_bias(rel[..., f].long() + self.offsets[f])
        bias = bias.permute(0, 3, 1, 2)                                          # [B, H, N, N]
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))          # nessuno guarda il padding

        # 3) sei layer
        for layer in self.layers:
            x = layer(x, bias)
        h = self.out_norm(x)

        # 4) teste
        cls_logits = self.cls_head(h)                                            # [B, N, 12]
        d = h.shape[-1]
        hi = h.gather(1, pairs[..., 0:1].expand(-1, -1, d))                      # [B, P, d]
        hj = h.gather(1, pairs[..., 1:2].expand(-1, -1, d))
        pe = self.pair_emb(pair_rel + self.offsets).sum(2)                       # [B, P, d_pair]
        link_logits = self.link_head(torch.cat([hi, hj, (hi - hj).abs(), hi * hj, pe], -1)).squeeze(-1)
        return cls_logits, link_logits                                           # [B, N, 12], [B, P]


def run(model, batch):
    return model(batch["feats"], batch["tok"], batch["rel"], batch["mask"], batch["pairs"], batch["pair_rel"])


def compute_losses(model, batch, cfg):
    cls_logits, link_logits = run(model, batch)
    loss_cls = F.cross_entropy(cls_logits.flatten(0, 1), batch["label"].flatten(),
                               ignore_index=-100, label_smoothing=cfg.label_smoothing)
    keep = batch["pair_mask"]
    if keep.any():
        loss_link = F.binary_cross_entropy_with_logits(link_logits[keep], batch["link"][keep])
    else:
        loss_link = link_logits.sum() * 0.0
    return loss_cls + cfg.link_weight * loss_link, loss_cls.item(), loss_link.item()
```

## 14. Prova a secco e training

### 14.1 Prova a secco

Questo script non scarica nulla. Genera pagine sintetiche (titolo, sezioni numerate con "2." separato dal nome, paragrafi, elenchi, una tabella, intestazione e piè di pagina) nello stesso formato che PyMuPDF restituisce, e controlla tutta la catena: estrazione, etichette, feature, candidati, blocchi. Se PyTorch è installato costruisce anche il modello, conta i parametri e fa 150 step di training su 16 pagine per verificare che la loss scenda.

Con un argomento scrive anche una cache finta, su cui puoi lanciare `baseline.py` e `train.py` in pochi minuti prima che il dataset vero sia pronto. Su quei dati conta solo che gli script arrivino in fondo senza errori: le metriche su pagine sintetiche non dicono nulla sulla qualità.

```python
# file: smoke_test.py
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
            save_raw(fake_raw(rng)[0], os.path.join(folder, split, f"fake_{k:04d}.npz"))
    print(f"Cache finta scritta in {folder}: {n_train} pagine di training, {n_val} di validation.")


if __name__ == "__main__":
    generator = np.random.default_rng(0)
    check_pipeline(generator)
    check_model(generator)
    if len(sys.argv) > 1:
        write_cache(sys.argv[1], generator)
```

### 14.2 Training

Che cosa fa lo script, in ordine.

1. Carica la cache di training e di validation.
2. Per ogni batch: calcola la loss, calcola in che direzione modificare i pesi (`backward`), limita l'ampiezza della modifica (`clip_grad_norm_`, evita salti distruttivi), aggiorna i pesi.
3. Ogni 100 step stampa le due loss e il tempo.
4. Alla fine di ogni epoca calcola le tre metriche sulla validation.
5. Salva il modello in `best.pt` solo quando l'F1 dei blocchi sulla validation migliora. È questo che protegge dall'overfitting: il file contiene sempre il modello migliore, non l'ultimo.

**Iperparametri e loro significato pratico.**

| Nome | Valore | Se lo alzi | Se lo abbassi |
|---|---|---|---|
| `lr` | 0,0003 | training più rapido, rischio che la loss esploda | più stabile, più lento |
| `batch_size` | 16 | più memoria, stime più stabili | meno memoria |
| `epochs` | 30 | più tempo; `best.pt` protegge comunque dall'overfitting | rischio di fermarsi presto |
| `dropout` | 0,1 | meno overfitting, apprendimento più lento | il contrario |
| `link_threshold` | 0,5 | blocchi più piccoli (meno fusioni sbagliate, più blocchi spezzati) | blocchi più grandi |

**Come leggere l'output.**

- All'inizio la loss delle classi vale circa 2,5 (il logaritmo di 12: il modello tira a caso tra 12 classi) e quella dei link circa 0,7. Devono scendere entrambe nei primi minuti.
- Con label smoothing 0,1 la loss delle classi non arriva a zero nemmeno con un modello perfetto: si ferma intorno a 0,5. Non è un difetto.
- Le metriche di validation devono salire nelle prime epoche e poi appiattirsi. Se scendono mentre la loss di training continua a calare, è overfitting.

**Tempi e memoria.** Non li ho misurati. Con 69.375 pagine di training e batch da 16, un'epoca sono circa 4.300 step: lo script stampa il tempo ogni 100 step, da cui ricavi la durata di un'epoca. Se la GPU esaurisce la memoria, dimezza `batch_size`; il consumo maggiore viene dalle pagine con molti nodi, perché l'attenzione cresce con il quadrato di N.

**Per iniziare in piccolo.** Non serve tutto il dataset per capire se funziona. Copia 5.000 file di `cache/train` in un'altra cartella e addestra lì: i risultati saranno peggiori ma l'andamento è lo stesso, e un errore si vede in un'ora invece che in un giorno.

```python
# file: train.py
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
from layout.data import PageDataset, collate, to_device
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
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=max(cfg.batch_size // 2, 1), shuffle=False, collate_fn=collate)
    print(f"Dispositivo: {device}. Pagine: {len(train_ds)} di training, {len(val_ds)} di validation.")

    model = LayoutModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(train_dl)
    warmup = min(cfg.warmup_steps, max(total // 10, 1))

    def lr_factor(step):                                     # sale nel warmup, poi scende a coseno fino a zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    best, step = -1.0, 0
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
                      f"{time.time() - start:.0f} s da inizio epoca")
        m = evaluate(model, val_dl, cfg, device)
        print(f"Epoca {epoch + 1}: F1 nodi {m['node_f1']:.3f}  F1 link {m['link_f1']:.3f}  "
              f"F1 blocchi {m['block_f1']:.3f}")
        print("   per classe: " + "  ".join(f"{c} {v:.2f}" for c, v in zip(CLASSES, m["per_class"])))
        if m["block_f1"] > best:
            best = m["block_f1"]
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "metrics": m},
                       os.path.join(out_dir, "best.pt"))
            print("   salvato come modello migliore")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
```

## 15. Uso su un tuo PDF

Lo script applica il modello a ogni pagina di un PDF e produce una lista di blocchi, ciascuno con classe, bbox e testo. È qui che i tre requisiti diventano visibili:

- R5a: un blocco con classe `title` o `section_header`;
- R5b: un blocco con classe `table` che contiene tutte le celle;
- R5c: un blocco `section_header` il cui testo è "2. Introduzione", ricomposto dai due nodi.

Stampa anche il tempo medio per pagina diviso in tre voci: estrazione dal PDF, calcolo delle feature, modello. È la misura del requisito R2, e dice dove conviene ottimizzare.

**Che tempi aspettarsi.** Ho potuto misurare solo il calcolo delle feature, su una macchina con due core: 6 ms per una pagina di 200 nodi, 27 ms per 500 nodi, 98 ms per 1.000 nodi. Cresce con il quadrato dei nodi, perché le feature di coppia riguardano tutte le coppie: sulle pagine molto dense è una voce che conta.

Per il modello ho solo una stima. Con 200 nodi fa circa un miliardo di moltiplicazioni: su una CPU moderna sono alcune decine di millisecondi, su GPU pochi millisecondi. Il tempo di estrazione di PyMuPDF non l'ho misurato. Se il modello risulta la voce più lenta, la prima cosa da provare è ridurlo (`d_model = 192`, `n_layers = 4`) e rimisurare la qualità.

```python
# file: infer.py
"""Applica il modello a un PDF e scrive i blocchi trovati in un file JSON.
Uso: python infer.py runs/v1/best.pt documento.pdf"""
import json
import sys
import time

import numpy as np
import pymupdf
import torch

from layout.config import CLASSES, Config
from layout.data import collate, to_device
from layout.decode import block_classes, find_blocks
from layout.extract import extract_page
from layout.features import add_text_feats, make_example
from layout.model import LayoutModel, run


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    cfg = Config(**ck["cfg"])
    model = LayoutModel(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg


@torch.no_grad()
def predict_page(page, model, cfg, device):
    """Ritorna i blocchi della pagina e i tempi (estrazione, feature, modello) in secondi."""
    t0 = time.perf_counter()
    raw = add_text_feats(extract_page(page, with_graphics=cfg.use_graphics))
    t1 = time.perf_counter()
    ex = make_example(raw, cfg, cap=10 ** 9)                 # in uso reale nessun limite al numero di nodi
    n = len(ex["feats"])
    if n == 0:
        return [], (t1 - t0, 0.0, 0.0)
    batch = to_device(collate([ex]), device)
    t2 = time.perf_counter()
    cls_logits, link_logits = run(model, batch)
    logits = cls_logits[0].cpu().numpy()
    prob = torch.sigmoid(link_logits[0]).cpu().numpy()[:len(ex["pairs"])]
    t3 = time.perf_counter()

    block = find_blocks(n, ex["pairs"], prob, logits.argmax(1), cfg.link_threshold)
    cls = block_classes(block, logits)
    out = []
    for b in range(int(block.max()) + 1):
        idx = np.flatnonzero(block == b)                     # i nodi sono già in ordine di lettura
        box = ex["box"][idx]
        out.append({
            "class": CLASSES[int(cls[b])],
            "bbox": [float(box[:, 0].min()), float(box[:, 1].min()), float(box[:, 2].max()), float(box[:, 3].max())],
            "text": " ".join(ex["texts"][i] for i in idx),
            "nodes": len(idx),
        })
    out.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))     # dall'alto in basso: non è un vero ordine di lettura
    return out, (t1 - t0, t2 - t1, t3 - t2)


def main(ckpt, pdf_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt, device)
    doc = pymupdf.open(pdf_path)
    pages, times = [], np.zeros(3)
    for number, page in enumerate(doc, start=1):
        blocks, t = predict_page(page, model, cfg, device)
        times += t
        pages.append({"page": number, "blocks": blocks})
        print(f"Pagina {number}: {sum(b['nodes'] for b in blocks)} nodi, {len(blocks)} blocchi")
        for b in blocks[:8]:
            print(f"   {b['class']:15s} {b['text'][:70]}")
    with open(pdf_path + ".layout.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    per_page = 1000 * times / max(len(pages), 1)
    print(f"Tempo medio per pagina: estrazione {per_page[0]:.1f} ms, feature {per_page[1]:.1f} ms, "
          f"modello {per_page[2]:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

## 16. Quando qualcosa non va

| Sintomo | Causa più probabile | Che cosa fare |
|---|---|---|
| `check_data.py` dà copertura bassa con tutte e quattro le ipotesi | scala o origine delle coordinate diversa da quella assunta | guarda le immagini `check_NNN.png`: mostrano di quanto e in che direzione i box sono spostati |
| La baseline dà risultati quasi casuali | etichette sbagliate (sezione 8) | come sopra: il problema è prima del modello |
| La loss è `nan` | learning rate troppo alto | dividi `lr` per 3 |
| La loss non scende dal valore iniziale | errore nei dati o learning rate troppo basso | lancia `smoke_test.py`: se lì scende, il modello è a posto e il problema è nella cache |
| F1 dei nodi alto, F1 dei blocchi basso | soglia dei link inadatta | prova `link_threshold` a 0,4 e 0,6 e confronta |
| Blocchi enormi che fondono mezza pagina | un link sbagliato unisce due blocchi | alza `link_threshold` |
| Paragrafi spezzati in più blocchi | link mancanti | abbassa `link_threshold`; se non basta, alza `knn` |
| Le classi rare (`title`, `footnote`, `formula`) hanno F1 vicino a zero | sbilanciamento: il modello ignora le classi con pochi esempi | aggiungi pesi di classe alla cross-entropy (parametro `weight`), proporzionali all'inverso della frequenza |
| Loss di training in calo, metriche di validation in calo | overfitting | alza `dropout` a 0,2 oppure usa più dati |
| Memoria della GPU esaurita | batch o pagine troppo grandi | dimezza `batch_size`; poi abbassa `max_nodes_train` |
| Buoni risultati su DocLayNet, scarsi sui tuoi PDF | i tuoi documenti sono diversi per lingua o impaginazione | sezione 17, primo punto |

## 17. Dopo la v1

In ordine di utilità. Nessuno di questi passi è nel codice qui sopra.

1. **Adattamento ai tuoi documenti.** È il passo che più probabilmente decide la qualità finale. Fai girare Docling sui tuoi PDF (offline la lentezza non conta), prendi i suoi blocchi con classe e bbox, portali in punti con origine in alto a sinistra e passali a `assign` come se fossero annotazioni di DocLayNet. Ottieni una cache con pseudo-etichette su cui continuare il training partendo da `best.pt`, con learning rate più basso (0,0001) e poche epoche. A parte, correggi a mano 200–300 pagine (per esempio con Label Studio) e non usarle mai per il training: sono il tuo vero test set.
2. **Confronto con Docling.** Sugli stessi PDF, misura tempo per pagina e qualità sul tuo test set. È la verifica dei requisiti R2 e R3.
3. **Nodi grafici.** Metti `use_graphics = True` e riaddestra: i filetti e le immagini sono già nella cache. Aiuta soprattutto a riconoscere le tabelle con bordi.
4. **Visione solo dove serve (R7).** La prima versione del "sospetto" non richiede di addestrare nulla: ogni blocco che il modello classifica come `table`, `picture` o `formula` è una regione sospetta, a cui si aggiungono le bbox delle immagini raster che il PDF dichiara. Solo quelle regioni vengono renderizzate (`page.get_pixmap(clip=rettangolo, dpi=...)`) e passate a un modello visivo, per esempio Table Transformer per la struttura delle tabelle. La misura da tenere d'occhio è la quota di tabelle vere che il modello testuale non segnala: quelle non arriveranno mai al secondo stadio.
5. **Struttura delle tabelle senza visione.** Lo stesso modello può imparare link "stessa riga" e "stessa colonna" tra celle, usando PubTables-1M o FinTabNet, che hanno le parole con bbox e la griglia.
6. **Livelli dei titoli e ordine di lettura.** Servono etichette che DocLayNet non ha; HRDoc le fornisce per articoli scientifici.
7. **Esportazione e quantizzazione** (ONNX). Solo se la misura di `infer.py` mostra che il modello è la voce lenta.

A ogni cambiamento, un'ablation: stesso training con e senza la novità, stesse metriche sulla validation.

## 18. Correzioni rispetto a quanto ti avevo scritto

Scrivendo e provando il codice ho cambiato alcune cose dette nei messaggi precedenti.

- **"Mescola i nodi in training".** Non serve: senza posizione nella sequenza il modello è già indifferente all'ordine dei nodi.
- **Token `[PAGE]`.** Tolto dalla v1. Le statistiche di pagina entrano come feature di ogni nodo.
- **"Millisecondi per pagina su CPU".** Era ottimistico: la stima corretta è di alcune decine di millisecondi, da misurare.
- **Spezzare la riga dove cambia il font.** Nella v1 la riga si spezza solo sui grandi spazi orizzontali; il cambio di font resta una feature. Un titolo in grassetto seguito dal testo sulla stessa riga resta quindi un nodo solo.
- **`EmbeddingBag`.** Sostituito da un embedding normale con 32 id fissi per nodo: stessa idea, codice più semplice.
- **LightGBM.** Sostituito dal gradient boosting di scikit-learn, che è della stessa famiglia e toglie una dipendenza.
- **Augmentation.** Ridotta a due trasformazioni (rumore sulle coordinate, pagine senza testo). Le altre che avevo elencato sono esperimenti da fare dopo.

## Appendice: creare i file da questa guida

Per non copiare a mano quattordici file, salva queste righe come `extract_code.py` nella cartella di progetto, accanto a questa guida, e lancia `python extract_code.py layout-model-guide.md`.

```python
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
for match in re.finditer(r"^```python\n# file: (\S+)\n(.*?)\n```$", text, flags=re.S | re.M):
    path = pathlib.Path(match.group(1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(match.group(2) + "\n", encoding="utf-8")
    print("scritto", path)
```
