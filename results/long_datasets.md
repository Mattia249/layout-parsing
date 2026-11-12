# Dataset di PDF lunghi e ricchi di grafica: confronto

Misurato con `compare_datasets.py` (fino a 8 pagine campionate per documento; pagina "grafica" = almeno 1 immagine
incorporata o 20 tracciati vettoriali). Il criterio è grossolano: vale per confrontare, non in assoluto.

| dataset | doc | pag. media | mediana | p90 | min | max | % pag. grafiche | img/pag | vett/pag | % doc >= 20 pag |
|---|---|---|---|---|---|---|---|---|---|---|
| DUDE (campione 400) | 400 | 5,8 | 3 | 16 | 1 | 24 | 79,3 | 2,73 | 206,4 | 7,0 |
| MMLongBench-Doc | 135 | 48,3 | 28 | 107 | 9 | 468 | 72,6 | 2,14 | 283,5 | 84,4 |
| **LongDocURL** | 396 | **85,6** | **78** | 127 | **51** | 149 | 68,5 | **10,93** | 106,1 | **100,0** |

Dai soli metadati (nessun PDF scaricato):

| dataset | doc | pag. media | mediana | note |
|---|---|---|---|---|
| MMDocIR | 313 | ~65 | ~27 | 10 domini, 8,5 layout/pagina (MinerU) con bbox; PDF in archivi rar da 11 GB con le immagini |
| pdfa-eng-wds (campione 450 doc) | 2,16 M | 8,7 | 3 | 1,5 TB; solo il 5,6% dei doc ha >= 15 pagine e grafica |

Scelto: **LongDocURL** (PDF in `datasets/LongDocURL/pdfs`, 396 file; domande in `LongDocURL_public_with_subtask_category.jsonl`).
