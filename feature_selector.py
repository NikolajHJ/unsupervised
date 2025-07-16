#!/usr/bin/env python3
"""
unsup_feature_selection_baseline.py
-----------------------------------
Backward feature elimination on three data sets:

    OG  = *_scaled.csv
    PCA = *_pca.csv
    ICA = *_ica.csv

Models & scoring function
-------------------------
KMeans           → Adjusted Rand Index (vs. heart_attack)
GaussianMixture  → Adjusted Rand Index
Torch-MLP        → classification accuracy

Outputs
-------
selected_datasets/<model>/<dataset>_mask.json
selected_datasets/selection_curves.csv
"""

from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.base import clone

from mlp import TorchMLPWrapper      # ← your supervised NN wrapper

# ------------- configuration -----------------------------------------
OG_TRAIN  = Path("heart_attack_prediction_indonesia_scaled_train.csv")
PCA_TRAIN = Path("heart_attack_prediction_indonesia_pca_train.csv")
ICA_TRAIN = Path("heart_attack_prediction_indonesia_ica_train.csv")

SAMPLE_N  = None      # rows used for feature selection (None = all)
OUT_ROOT  = Path("selected_datasets")
DEVICE    = "cuda"      # or "cpu"
# ---------------------------------------------------------------------


def load_matrix(csv: Path):
    """Return X (n×p), col_names, y_true."""
    df = pd.read_csv(csv)
    y_true = df["heart_attack"].values
    X_df = df.drop(columns=["heart_attack"])
    if SAMPLE_N and SAMPLE_N < len(X_df):
        X_df = X_df.sample(n=SAMPLE_N, random_state=42)
        y_true = y_true[X_df.index]
    return X_df.values, X_df.columns, y_true


def metric(estimator, X, y, cols):
    """
    Choose ARI for clustering (fit_predict) or accuracy for classifiers.
    """
    if hasattr(estimator, "fit_predict"):        # unsupervised
        labels = clone(estimator).fit_predict(X[:, cols])
        # guard against single-cluster collapse
        if len(set(labels)) < 2:
            return -np.inf
        return adjusted_rand_score(y, labels)
    else:                                        # supervised baseline
        mdl = clone(estimator).fit(X[:, cols], y)
        return accuracy_score(y, mdl.predict(X[:, cols]))


def backward_selection(estimator, X, y):
    selected = list(range(X.shape[1]))
    best_score = metric(estimator, X, y, selected)
    best_subset = selected.copy()
    curve = []

    while len(selected) > 1:
        scores = [(metric(estimator, X, y,
                          [c for c in selected if c != f]), f)
                  for f in selected]
        step_score, drop_f = max(scores, key=lambda t: t[0])
        selected.remove(drop_f)
        curve.append(step_score)
        if step_score >= best_score:
            best_score, best_subset = step_score, selected.copy()

    mask = np.zeros(X.shape[1], bool)
    mask[best_subset] = True
    return mask, curve


def main():
    warnings.filterwarnings("ignore")
    OUT_ROOT.mkdir(exist_ok=True)

    datasets = {
        "OG":  OG_TRAIN,
        "PCA": PCA_TRAIN,
        "ICA": ICA_TRAIN,
    }

    models = {
        "kmeans": KMeans(n_clusters=2, random_state=0),
        "gmm":    GaussianMixture(n_components=2,
                                  covariance_type="full",
                                  random_state=0),
        # "mlp":    TorchMLPWrapper(device=DEVICE,
        #                           epochs=25,
        #                           early_stopping=True,
        #                           verbose=0,
        #                           random_state=0),
    }

    rows = []
    # total iterations = (#datasets) × (#models)
    total_steps = len(datasets) * len(models)

    with tqdm(total=total_steps, desc="Backward FS", unit="comb") as pbar:
        for ds_name, csv in datasets.items():
            X, col_names, y_true = load_matrix(csv)

            for mdl_name, est in models.items():
                print(f"{mdl_name.upper():5} on {ds_name:3} "
                    f"(rows={X.shape[0]}, feats={X.shape[1]})")

                mask, curve = backward_selection(est, X, y_true)

                # save mask
                mdl_dir = OUT_ROOT / mdl_name
                mdl_dir.mkdir(exist_ok=True)
                with open(mdl_dir / f"{ds_name}_mask.json", "w") as fp:
                    json.dump(col_names[mask].tolist(), fp, indent=2)

                metric_name = "ari" if mdl_name in {"kmeans", "gmm"} else "accuracy"
                rows.extend(
                    {"model": mdl_name,
                    "dataset": ds_name,
                    "n_features": X.shape[1] - i,
                    metric_name: s}
                    for i, s in enumerate(curve, start=1)
                )

                print(f"  kept {mask.sum():2d} features | "
                    f"best {metric_name} = {max(curve):.3f}")
                pbar.update(1)

    if rows:
        pd.DataFrame(rows).to_csv(
            OUT_ROOT / "selection_curves.csv", index=False)
        print("\nselection_curves.csv written to", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
