#!/usr/bin/env python3
"""
add_cluster_features.py
-----------------------
For each data set (OG, PCA, ICA, RP):

  1. Load datasets/{folder}/{stem}_train.csv and _test.csv
  2. Read the backward‐selection masks:
       selected_datasets/kmeans/{DS}_mask.json
       selected_datasets/gmm   /{DS}_mask.json
     (falls back to all columns if a mask is missing)
  3. Fit KMeans(n_clusters=2) and GaussianMixture(n_components=2) on TRAIN
     (masked cols)
  4. Predict cluster labels for TRAIN and TEST
  5. Append two new cols: cluster_kmeans, cluster_gmm
  6. Save:
       datasets/{folder}/{stem}_withclusters_train.csv
       datasets/{folder}/{stem}_withclusters_test.csv
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# ─────── configuration ───────
# point this at the root of your split CSVs:
BASE = Path("datasets")

# map logical dataset name → subfolder & file‐stem prefix
DATASETS = {
    "OG":     ("scaled", "ha_scaled"),
    "PCA":    ("pca",    "ha_pca"),
    "ICA":    ("ica",    "ha_ica"),
    "RP":     ("rp",     "ha_rp"),
}

MASK_DIR = Path("selected_datasets")
# ──────────────────────────────


def load_split(folder: str, stem: str):
    """Load train/test DataFrames from datasets/{folder}/{stem}_*.csv."""
    train_df = pd.read_csv(BASE / folder / f"{stem}_train.csv")
    test_df  = pd.read_csv(BASE / folder / f"{stem}_test.csv")
    return train_df, test_df


def mask_indices(model: str, ds_name: str, feat_cols):
    """Return list of column‐indices to keep, or None for all."""
    p = MASK_DIR / model / f"{ds_name}_mask.json"
    if p.exists():
        cols = json.loads(p.read_text())
        return [feat_cols.get_loc(c) for c in cols]
    return None


def fit_predict_clusters(kind: str, X_tr, X_te):
    """Fit one of 'kmeans' or 'gmm' on X_tr, predict on X_tr & X_te."""
    if kind == "kmeans":
        est = KMeans(n_clusters=2, random_state=0)
    else:
        est = GaussianMixture(n_components=2,
                              covariance_type="full",
                              random_state=0)
    est.fit(X_tr)
    return est.predict(X_tr), est.predict(X_te)


def main():
    for ds_name, (folder, stem) in DATASETS.items():
        # 1) load splits
        tr_df, te_df = load_split(folder, stem)
        feat_cols = tr_df.columns.drop("heart_attack")

        X_tr_full = tr_df[feat_cols].values
        X_te_full = te_df[feat_cols].values

        # 2) K‑means clusters
        km_idx = mask_indices("kmeans", ds_name, feat_cols)
        Xtr_km = X_tr_full[:, km_idx] if km_idx else X_tr_full
        Xte_km = X_te_full[:, km_idx] if km_idx else X_te_full
        km_tr, km_te = fit_predict_clusters("kmeans", Xtr_km, Xte_km)
        tr_df["cluster_kmeans"] = km_tr
        te_df["cluster_kmeans"] = km_te

        # 3) GMM clusters
        gmm_idx = mask_indices("gmm", ds_name, feat_cols)
        Xtr_gm = X_tr_full[:, gmm_idx] if gmm_idx else X_tr_full
        Xte_gm = X_te_full[:, gmm_idx] if gmm_idx else X_te_full
        gm_tr, gm_te = fit_predict_clusters("gmm", Xtr_gm, Xte_gm)
        tr_df["cluster_gmm"] = gm_tr
        te_df["cluster_gmm"] = gm_te

        # 4) save augmented versions
        out_train = BASE / folder / f"{stem}_withclusters_train.csv"
        out_test  = BASE / folder / f"{stem}_withclusters_test.csv"
        tr_df.to_csv(out_train, index=False)
        te_df.to_csv(out_test,  index=False)

        print(f"{ds_name:3} → wrote {out_train.name}, {out_test.name}")

    print("\n✓ All datasets augmented with cluster‐labels.")


if __name__ == "__main__":
    main()
