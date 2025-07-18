#!/usr/bin/env python3
"""
silhouette_grid.py
==================
Compute silhouette scores for K-means and GMM with k ∈ {2 … 25}
on each data set (OG, PCA, ICA, RP).

Inputs
------
• datasets/scaled/…_scaled_train.csv
• datasets/pca/…_pca_train.csv
• datasets/ica/…_ica_train.csv
• datasets/rp/…_rp_train.csv
• selected_datasets/kmeans/<DATASET>_mask.json   (optional)
• selected_datasets/gmm/<DATASET>_mask.json     (optional)

Outputs
-------
results/silhouette_scores.csv
plots/silhouette_curves.png
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ------------ configuration ------------------------------------------
BASE       = Path("datasets")
DATASETS   = {
    "OG":  BASE / "scaled" / "ha_scaled_train.csv",
    "PCA": BASE / "pca"    / "ha_pca_train.csv",
    "ICA": BASE / "ica"    / "ha_ica_train.csv",
    "RP":  BASE / "rp"     / "ha_rp_train.csv",
}
MASK_DIR   = Path("selected_datasets")
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR   = Path("plots");   PLOT_DIR.mkdir(exist_ok=True)

K_RANGE = range(2, 26)
# ---------------------------------------------------------------------


def load_matrix(csv_path: Path):
    """Return (X, feat_cols_index)."""
    df = pd.read_csv(csv_path)
    if "heart_attack" in df.columns:
        df = df.drop(columns=["heart_attack"])
    return df.values, df.columns


def mask_indices(model: str, dataset: str, feat_cols):
    """
    Return list of feature‐indices to keep, or None for all.
    """
    p = MASK_DIR / model / f"{dataset}_mask.json"
    if p.exists():
        names = json.loads(p.read_text())
        return [feat_cols.get_loc(c) for c in names]
    return None


def compute_silhouette(model_name: str, X: np.ndarray, k: int) -> float:
    """
    Fit the clusterer on X and return the silhouette score.
    """
    if model_name == "kmeans":
        est = KMeans(n_clusters=k, random_state=0)
        labels = est.fit_predict(X)
    else:  # "gmm"
        est = GaussianMixture(
            n_components=k, covariance_type="full", random_state=0
        )
        est.fit(X)
        labels = est.predict(X)

    # silhouette_score requires at least 2 clusters
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(X, labels, metric="euclidean")


def main():
    warnings.filterwarnings("ignore")
    rows = []

    for ds_name, csv_path in DATASETS.items():
        if not csv_path.exists():
            print(f"Skipping {ds_name}: {csv_path} not found")
            continue

        X_full, feat_cols = load_matrix(csv_path)

        for model in ("kmeans", "gmm"):
            # apply mask if present
            idx = mask_indices(model, ds_name, feat_cols)
            X = X_full[:, idx] if idx is not None else X_full

            for k in tqdm(K_RANGE, desc=f"{model.upper()} {ds_name}", unit="k"):
                sil = compute_silhouette(model, X, k)
                rows.append({
                    "model":    model,
                    "dataset":  ds_name,
                    "k":        k,
                    "silhouette": sil,
                })

    # save CSV
    sil_df = pd.DataFrame(rows)
    out_csv = RESULT_DIR / "silhouette_scores.csv"
    sil_df.to_csv(out_csv, index=False)
    print("✓ silhouette scores written to", out_csv.resolve())

    # quick plot
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=sil_df,
        x="k",
        y="silhouette",
        hue="dataset",
        style="model",
        markers=True,
        dashes=False,
        palette="Set2",
    )
    plt.xlabel("Number of clusters/components (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette curves (training split)")
    plt.legend(title="Data set / model", ncol=2, fontsize=8)
    plt.tight_layout()
    out_fig = PLOT_DIR / "silhouette_curves.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print("✓ silhouette plot saved to", out_fig.resolve())


if __name__ == "__main__":
    main()
