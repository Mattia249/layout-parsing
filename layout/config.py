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
    # risorse
    num_workers: int = 8        # processi che preparano i batch in parallelo (0 = tutto nel processo principale)
