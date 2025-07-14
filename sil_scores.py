#!/usr/bin/env python3
"""
silhouette_grid.py
==================
Compute silhouette scores for K-means and GMM with k ∈ {2 … 25}
on each data set (OG, PCA, ICA, …).

Inputs
------
• <stem>_train.csv  (scaled / PCA / ICA)  from prepare_unsupervised_datasets.py
• selected_datasets/kmeans/<DATASET>_mask.json   (optional)
• selected_datasets/gmm/<DATASET>_mask.json     (optional)

Outputs
-------
results/silhouette_scores.csv
    model , dataset , k , silhouette
plots/silhouette_curves.png
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ------------ configuration ------------------------------------------
STEM = "heart_attack_prediction_indonesia"

DATASETS = {
    "OG":  f"{STEM}_scaled_train.csv",
    # "PCA": f"{STEM}_pca_train.csv",
    # "ICA": f"{STEM}_ica_train.csv",
}

MASK_DIR   = Path("selected_datasets")
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR   = Path("plots");   PLOT_DIR.mkdir(exist_ok=True)

K_RANGE = range(2, 26)
# ---------------------------------------------------------------------


def load_matrix(csv_path):
    """Return X (numeric matrix) and feature-name index."""
    df = pd.read_csv(csv_path)
    if "heart_attack" in df:
        df = df.drop(columns=["heart_attack"])
    return df.values, df.columns


def mask_indices(model, dataset, feat_cols):
    p = MASK_DIR / model / f"{dataset}_mask.json"
    if p.exists():
        names = json.loads(p.read_text())
        return [feat_cols.get_loc(c) for c in names]
    return None


def compute_silhouette(model_name, X, k):
    if model_name == "kmeans":
        mdl = KMeans(n_clusters=k, random_state=0)
    else:
        mdl = GaussianMixture(
            n_components=k, covariance_type="full", random_state=0
        )
    labels = mdl.fit_predict(X)
    # silhouette_score needs >1 cluster label; ensure unique >1
    if len(set(labels)) == 1:
        return np.nan
    return silhouette_score(X, labels, metric="euclidean")


def main():
    warnings.filterwarnings("ignore")
    rows = []

    for ds, csv in DATASETS.items():
        X_full, feat_cols = load_matrix(csv)

        for model in ("kmeans", "gmm"):
            idx = mask_indices(model, ds, feat_cols)
            X = X_full[:, idx] if idx else X_full

            for k in tqdm(K_RANGE):
                sil = compute_silhouette(model, X, k)
                rows.append({
                    "model": model,
                    "dataset": ds,
                    "k": k,
                    "silhouette": sil,
                })
            print(f"[{model.upper()} {ds}] done")

    # ---------- save CSV ----------
    out_csv = RESULT_DIR / "silhouette_scores.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("✓ scores written to", out_csv.resolve())

    # ---------- quick plot ----------
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df,
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
    plt.title("Silhouette curves (train split)")
    plt.legend(title="Data set / model", ncol=2, fontsize=8)
    plt.tight_layout()
    out_fig = PLOT_DIR / "silhouette_curves.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print("✓ plot saved to", out_fig.resolve())


if __name__ == "__main__":
    main()
