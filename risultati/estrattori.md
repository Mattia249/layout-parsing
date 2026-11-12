# PyMuPDF e LiteParse a confronto

Data: 2026-11-12. Comando: `uv run confronta_estrattori.py runs/v1/best.pt 500`.

Stesso modello (`runs/v1/best.pt`, addestrato su nodi estratti con PyMuPDF), stesse 503 pagine di validation
prese da tutti i 7 file, metriche contro le annotazioni di DocLayNet. LiteParse 2.18.0, PyMuPDF 2.1.

| Estrattore | F1 nodi | F1 link | F1 blocchi | Nodi/pagina | Copertura | ms/pagina |
|---|---|---|---|---|---|---|
| PyMuPDF | **0.896** | 0.975 | 0.868 | 69.4 | 0.996 | 3.7 |
| LiteParse | 0.862 | 0.973 | 0.868 | 66.8 | 0.997 | 7.1 |

| Classe | PyMuPDF | LiteParse | Nodi |
|---|---|---|---|
| caption | 0.634 | 0.621 | 268 |
| footnote | 0.951 | 0.943 | 40 |
| formula | 0.995 | 0.981 | 1071 |
| list_item | 0.929 | 0.934 | 2096 |
| page_footer | 0.987 | 0.980 | 543 |
| page_header | 0.944 | 0.939 | 769 |
| picture | 0.831 | 0.803 | 2063 |
| section_header | 0.896 | 0.886 | 1810 |
| table | 0.951 | 0.951 | 10549 |
| text | 0.970 | 0.971 | 15464 |
| title | 0.976 | 0.976 | 128 |
| other | 0.684 | 0.364 | 90 |

## Che cosa ne segue

- Sui blocchi i due estrattori sono pari. La differenza sulla F1 dei nodi viene quasi tutta da `other`
  (90 nodi, testo non annotato) e in parte da `picture` e `formula`.
- Il motivo più probabile è che il modello ha imparato sui nodi di PyMuPDF, e LiteParse spezza il testo in
  modo un po' diverso: riscrive le formule in LaTeX (`$A_{0}$`) senza font, e sul testo spaziato lettera per
  lettera mette spazi sbagliati già nel suo testo ("NOT ES TO AN D").
- Dopo questo confronto LiteParse è stato ridotto al solo testo (niente vettori, immagini o blocchi di layout):
  in `annota.py` si comporta come l'OCR, e la modalità `misto` è stata tolta.

## Correzioni fatte nella ricostruzione delle righe (prima: F1 nodi 0.841, `other` 0.125)

- un item molto più alto della mediana (un "25" decorativo di 300 punti) diventa una riga a sé;
- item vicini sulla stessa riga ma non consecutivi nell'ordine del PDF vanno in nodi separati;
- soglia per lo spazio tra item a 0,08 del corpo (con 0,15 si perdevano spazi veri: "equitystatement");
- caratteri di controllo sostituiti da spazi.
