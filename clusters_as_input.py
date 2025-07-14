#!/usr/bin/env python3
"""
add_cluster_features.py
-----------------------
For each data set (OG, PCA, ICA …):

    1. Load <stem>_train.csv and <stem>_test.csv
    2. Read the backward-selection masks
           selected_datasets/kmeans/<dataset>_mask.json
           selected_datasets/gmm/<dataset>_mask.json
       (falls back to all columns if a mask is missing)
    3. Fit K-means-2 and GMM-2 on TRAIN (masked cols)
    4. Predict cluster labels for TRAIN and TEST
    5. Append two new columns:
           cluster_kmeans , cluster_gmm
    6. Save
           <stem>_withclusters_train.csv
           <stem>_withclusters_test.csv
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# ------------- configuration -----------------------------------------
STEM = "heart_attack_prediction_indonesia"

DATASETS = {
    "OG":  f"{STEM}_scaled",
    "PCA": f"{STEM}_pca",
    "ICA": f"{STEM}_ica",
    # "RP":  f"{STEM}_rp",   # ← add when you have RP files
}

MASK_DIR = Path("selected_datasets")
# ---------------------------------------------------------------------


def load_split(stem: str):
    """Return (train_df, test_df) – label still inside."""
    train_df = pd.read_csv(f"{stem}_train.csv")
    test_df  = pd.read_csv(f"{stem}_test.csv")
    return train_df, test_df


def mask_indices(model: str, dataset: str, feat_cols):
    """Return list of column indices or None."""
    path = MASK_DIR / model / f"{dataset}_mask.json"
    if path.exists():
        names = json.loads(path.read_text())
        return [feat_cols.get_loc(c) for c in names]
    return None


def fit_predict(model_name, X_train, X_test):
    """Fit clustering model and return labels for train & test."""
    if model_name == "kmeans":
        est = KMeans(n_clusters=2, random_state=0)
    else:  # gmm
        est = GaussianMixture(
            n_components=2, covariance_type="full", random_state=0
        )
    est.fit(X_train)
    return est.predict(X_train), est.predict(X_test)


def main():
    warnings.filterwarnings("ignore")

    for ds_name, stem in DATASETS.items():
        tr_df, te_df = load_split(stem)
        feat_cols = tr_df.columns.drop("heart_attack")

        X_tr = tr_df[feat_cols].values
        X_te = te_df[feat_cols].values

        # --- K-means ---------------------------------------------------
        km_idx = mask_indices("kmeans", ds_name, feat_cols)
        km_tr, km_te = fit_predict(
            "kmeans",
            X_tr[:, km_idx] if km_idx else X_tr,
            X_te[:, km_idx] if km_idx else X_te,
        )
        tr_df["cluster_kmeans"] = km_tr
        te_df["cluster_kmeans"] = km_te

        # --- GMM -------------------------------------------------------
        gmm_idx = mask_indices("gmm", ds_name, feat_cols)
        gm_tr, gm_te = fit_predict(
            "gmm",
            X_tr[:, gmm_idx] if gmm_idx else X_tr,
            X_te[:, gmm_idx] if gmm_idx else X_te,
        )
        tr_df["cluster_gmm"] = gm_tr
        te_df["cluster_gmm"] = gm_te

        # --- save augmented CSVs --------------------------------------
        aug_train = f"{stem}_withclusters_train.csv"
        aug_test  = f"{stem}_withclusters_test.csv"
        tr_df.to_csv(aug_train, index=False)
        te_df.to_csv(aug_test,  index=False)

        print(f"{ds_name:3} → wrote {aug_train}  {aug_test}")

    print("\n✓ All augmented files written.")


if __name__ == "__main__":
    main()
