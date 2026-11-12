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
