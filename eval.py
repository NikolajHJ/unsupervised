#!/usr/bin/env python3
"""
unsup_test_evaluate.py
======================
Compute test‐set accuracy for

    • K-Means (k=2)      → accuracy + ARI
    • GMM (components=2) → accuracy + ARI
    • NN baseline        → accuracy
    • NN + cluster feats → accuracy

Input files required under datasets/:
    {scaled,pca,ica,rp}/{stem}_train.csv
    {scaled,pca,ica,rp}/{stem}_test.csv
    {scaled,pca,ica,rp}/{stem}_withclusters_train.csv
    {scaled,pca,ica,rp}/{stem}_withclusters_test.csv

Feature masks in selected_datasets/<model>/<DATASET>_mask.json

Output:
    results/unsup_test_scores.csv
      model, dataset, accuracy, ari
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score
from mlp import TorchMLPWrapper

warnings.filterwarnings("ignore")

# ─────── configuration ───────
BASE       = Path("datasets")
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)
DEVICE     = "cuda"  # or "cpu"
# point this at the root of your split CSVs:

# map logical dataset name → subfolder & file‐stem prefix
DATASETS = {
    "OG":     ("scaled", "ha_scaled"),
    "PCA":    ("pca",    "ha_pca"),
    "ICA":    ("ica",    "ha_ica"),
    "RP":     ("rp",     "ha_rp"),
}

MASK_DIR = Path("selected_datasets")
# ──────────────────────────────
# ──────────────────────────────


def load_split(subfolder: str, stem: str, with_clusters: bool=False):
    """
    Load train/test splits from datasets/{subfolder}/{stem}[_withclusters]_{train,test}.csv
    Returns: X_tr, y_tr, X_te, y_te, feat_cols
    """
    suffix = "_withclusters" if with_clusters else ""
    tr = pd.read_csv(BASE / subfolder / f"{stem}{suffix}_train.csv")
    te = pd.read_csv(BASE / subfolder / f"{stem}{suffix}_test.csv")
    y_tr = tr.pop("heart_attack").values
    y_te = te.pop("heart_attack").values
    return tr.values, y_tr, te.values, y_te, tr.columns


def mask_indices(model: str, ds_name: str, feat_cols):
    """
    Load selected_datasets/<model>/<ds_name>_mask.json (if exists),
    return list of integer indices into feat_cols, or None
    """
    p = MASK_DIR / model / f"{ds_name}_mask.json"
    if p.exists():
        names = json.loads(p.read_text())
        return [feat_cols.get_loc(c) for c in names]
    return None


def eval_cluster(model_name, X_tr, y_tr, X_te, y_te, cols):
    """
    Fit clusterer on TRAIN (masked or full), map clusters→classes by majority vote,
    then predict on TEST. Returns (accuracy, ari).
    """
    # apply mask if provided
    Xtr = X_tr[:, cols] if cols is not None else X_tr
    Xte = X_te[:, cols] if cols is not None else X_te

    if model_name == "kmeans":
        est = KMeans(n_clusters=2, random_state=0)
        # fit and predict train in one go:
        lbl_tr = est.fit_predict(Xtr)
    else:  # "gmm"
        est = GaussianMixture(n_components=2,
                              covariance_type="full",
                              random_state=0)
        est.fit(Xtr)
        lbl_tr = est.predict(Xtr)

    # map cluster → class by majority vote on TRAIN
    mapping = {k: int(np.round(y_tr[lbl_tr == k].mean()))
               for k in np.unique(lbl_tr)}

    # predict on TEST
    lbl_te = est.predict(Xte)
    y_pred = np.vectorize(mapping.get)(lbl_te)

    acc = accuracy_score(y_te, y_pred)
    ari = adjusted_rand_score(y_te, lbl_te)
    return acc, ari

def eval_nn(X_tr, y_tr, X_te, y_te):
    """Train a fresh TorchMLPWrapper on (X_tr,y_tr) and eval accuracy on X_te"""
    nn = TorchMLPWrapper(device=DEVICE,
                         epochs=50,
                         early_stopping=True,
                         verbose=0,
                         random_state=0)
    nn.fit(X_tr, y_tr)
    return accuracy_score(y_te, nn.predict(X_te))


def main():
    rows = []

    for ds_name, (subfolder, stem) in DATASETS.items():
        print(f"\n=== {ds_name} ===")
        # load raw splits
        X_tr, y_tr, X_te, y_te, feat_cols = load_split(subfolder, stem)

        # --- clustering models ---
        for mdl in ("kmeans", "gmm"):
            idx = mask_indices(mdl, ds_name, feat_cols)
            acc, ari = eval_cluster(mdl, X_tr, y_tr, X_te, y_te, idx)
            rows.append({"model":   mdl,
                         "dataset": ds_name,
                         "accuracy": acc,
                         "ari":       ari})
            print(f"{mdl.upper():6}  acc={acc:.3f}  ari={ari:.3f}")

        # --- NN baseline ---
        acc_nn = eval_nn(X_tr, y_tr, X_te, y_te)
        rows.append({"model":"nn",
                     "dataset":ds_name,
                     "accuracy":acc_nn,
                     "ari": np.nan})
        print(f"NN       acc={acc_nn:.3f}")

        # --- NN + cluster features ---
        X_tr2, y_tr2, X_te2, y_te2, _ = load_split(subfolder, stem, with_clusters=True)
        acc_aug = eval_nn(X_tr2, y_tr2, X_te2, y_te2)
        rows.append({"model":"nn+clust",
                     "dataset":ds_name,
                     "accuracy":acc_aug,
                     "ari": np.nan})
        print(f"NN+Clust acc={acc_aug:.3f}")

    # dump results
    out = RESULT_DIR / "unsup_test_scores.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n✓ Written test scores to", out.resolve())


if __name__ == "__main__":
    main()
