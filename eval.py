#!/usr/bin/env python3
"""
unsup_test_evaluate.py
======================
Compute test-set accuracy for

    • K-Means (k=2)      → accuracy + ARI
    • GMM (components=2) → accuracy + ARI
    • NN baseline        → accuracy
    • NN + cluster feats → accuracy

Input files required
--------------------
heart_attack_prediction_indonesia_*_train.csv
heart_attack_prediction_indonesia_*_test.csv
heart_attack_prediction_indonesia_*_withclusters_train.csv
heart_attack_prediction_indonesia_*_withclusters_test.csv

selected_datasets/<model>/<dataset>_mask.json   (feature masks)

Output
------
results/unsup_test_scores.csv
    model , dataset , accuracy , ari
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score
from mlp import TorchMLPWrapper

# ------------ configuration ------------------------------------------
STEM = "heart_attack_prediction_indonesia"

DATASETS = {
    "OG":  f"{STEM}_scaled",
    "PCA": f"{STEM}_pca",
    "ICA": f"{STEM}_ica",
}

MASK_DIR   = Path("selected_datasets")
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)
DEVICE     = "cuda"            # or "cpu"
# ---------------------------------------------------------------------


def load_split(stem: str, with_clusters=False):
    """Return X_train, y_train, X_test, y_test, feature names."""
    suffix = "_withclusters" if with_clusters else ""
    tr = pd.read_csv(f"{stem}{suffix}_train.csv")
    te = pd.read_csv(f"{stem}{suffix}_test.csv")
    y_tr = tr.pop("heart_attack").values
    y_te = te.pop("heart_attack").values
    return tr.values, y_tr, te.values, y_te, tr.columns


def mask_indices(model: str, dataset: str, feat_cols):
    """Return list of indices or None if no mask."""
    p = MASK_DIR / model / f"{dataset}_mask.json"
    if p.exists():
        names = json.loads(p.read_text())
        return [feat_cols.get_loc(c) for c in names]
    return None


def eval_cluster(model, X_tr, y_tr, X_te, y_te, cols):
    """Return (accuracy, ari) for KMeans/GMM."""
    if model == "kmeans":
        est = KMeans(n_clusters=2, random_state=0)
    else:  # gmm
        est = GaussianMixture(n_components=2,
                              covariance_type="full",
                              random_state=0)

    Xtr_sel = X_tr[:, cols] if cols is not None else X_tr
    Xte_sel = X_te[:, cols] if cols is not None else X_te

    est.fit(Xtr_sel)
    lbl_tr = est.predict(Xtr_sel)
    lbl_te = est.predict(Xte_sel)

    # majority-vote mapping
    mapping = {}
    for k in np.unique(lbl_tr):
        mapping[k] = int(np.round(y_tr[lbl_tr == k].mean()))
    y_pred = np.vectorize(mapping.get)(lbl_te)

    acc = accuracy_score(y_te, y_pred)
    ari = adjusted_rand_score(y_te, lbl_te)
    return acc, ari


def eval_nn(X_tr, y_tr, X_te, y_te):
    nn = TorchMLPWrapper(device=DEVICE,
                         epochs=100,
                         early_stopping=True,
                         verbose=0,
                         random_state=0)
    nn.fit(X_tr, y_tr)
    return accuracy_score(y_te, nn.predict(X_te))


def main():
    warnings.filterwarnings("ignore")
    rows = []

    for ds, stem in DATASETS.items():
        print(f"\n--- {ds} ---")

        # ---------- clustering models ----------
        X_tr, y_tr, X_te, y_te, feat_cols = load_split(stem)

        for mdl in ("kmeans", "gmm"):
            idx = mask_indices(mdl, ds, feat_cols)
            acc, ari = eval_cluster(mdl, X_tr, y_tr, X_te, y_te, idx)
            rows.append({"model": mdl, "dataset": ds,
                         "accuracy": acc, "ari": ari})
            print(f"{mdl.upper():6}  acc={acc:.3f}  ari={ari:.3f}")

        # ---------- baseline NN ----------
        nn_acc = eval_nn(X_tr, y_tr, X_te, y_te)
        rows.append({"model": "nn", "dataset": ds,
                     "accuracy": nn_acc, "ari": np.nan})
        print(f"NN       acc={nn_acc:.3f}")

        # ---------- NN + cluster features ----------
        X_tr_aug, y_tr_aug, X_te_aug, y_te_aug, _ = load_split(
            stem, with_clusters=True)
        nn_aug_acc = eval_nn(X_tr_aug, y_tr_aug, X_te_aug, y_te_aug)
        rows.append({"model": "nn+clust", "dataset": ds,
                     "accuracy": nn_aug_acc, "ari": np.nan})
        print(f"NN+Clust acc={nn_aug_acc:.3f}")

    out = RESULT_DIR / "unsup_test_scores.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n✓ test accuracies written to", out.resolve())


if __name__ == "__main__":
    main()
