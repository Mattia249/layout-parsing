# Baseline (HistGradientBoosting)

Data: 2026-11-12. Comando: `uv run baseline.py cache`.

Impostazioni dello script: 3.000 pagine di training e 500 di validation prese a caso (seed 0) dalla cache
completa; 42 feature di base del nodo + relazione con il nodo precedente e successivo; niente token di testo.

Nodi di training: 204.695, di validation: 33.576.

| Classe | F1 | Nodi in validation |
|---|---|---|
| caption | 0.476 | 248 |
| footnote | 0.157 | 25 |
| formula | 0.738 | 941 |
| list_item | 0.657 | 2777 |
| page_footer | 0.883 | 521 |
| page_header | 0.771 | 555 |
| picture | 0.648 | 2546 |
| section_header | 0.798 | 1552 |
| table | 0.848 | 8095 |
| text | 0.899 | 16049 |
| title | 0.404 | 73 |
| other | 0.060 | 194 |

**Macro-F1 (sulle classi presenti): 0.612** — è il numero che la rete deve battere sulla F1 dei nodi.

Nota: la validation qui sono 500 pagine, quella di `train.py` è l'intero split (6.321 pagine); il confronto
è indicativo, non esatto.
