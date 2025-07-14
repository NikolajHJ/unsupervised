#!/usr/bin/env python3
"""
Grouped bar-chart: test-set ACCURACY for

    • KMEANS      (clusters→class mapping)
    • GMM
    • NN baseline
    • NN + cluster features

One coloured bar per data set (OG, PCA, ICA, …) in each group.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

RES_DIR = Path("results")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. load test-set scores
# ---------------------------------------------------------------
df = pd.read_csv(RES_DIR / "unsup_test_scores.csv")   # model,dataset,accuracy,ari

# we use ACCURACY for *every* group
name_map = {
    "kmeans":   "KMEANS",
    "gmm":      "GMM",
    "nn":       "NN",
    "nn+clust": "NN+Clust",
}
df["group"] = df["model"].map(name_map)
df["score"] = df["accuracy"]          # a single metric column

# ---------------------------------------------------------------
# 2. order axis categories
# ---------------------------------------------------------------
order_grp  = ["KMEANS", "GMM", "NN", "NN+Clust"]
primary_ds = ["OG", "PCA", "ICA"]
other_ds   = [d for d in df["dataset"].unique() if d not in primary_ds]
order_ds   = primary_ds + other_ds

df["group"]   = pd.Categorical(df["group"],  order_grp,  ordered=True)
df["dataset"] = pd.Categorical(df["dataset"], order_ds, ordered=True)
df = df.sort_values(["group", "dataset"])

# ---------------------------------------------------------------
# 3. plot
# ---------------------------------------------------------------
plt.figure(figsize=(8, 4))
ax = sns.barplot(
    data=df,
    x="group",
    y="score",
    hue="dataset",
    palette="Set2",
    width=0.8,
    edgecolor="black",
)

ax.set_ylabel("Test-set Accuracy")
ax.set_xlabel("")
ax.set_title("Accuracy comparison across models and data sets")
ax.set_ylim(0, 1)
ax.legend(title="Data set", loc="upper left")

plt.tight_layout()
out_path = OUT_DIR / "main_results.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("✓ plot saved to", out_path.resolve())



# ------------------------------------------------------------------
# PCA explained-variance plot
# ------------------------------------------------------------------
import joblib
import numpy as np

pca_path = Path("heart_attack_prediction_indonesia_pca.joblib")
pca = joblib.load(pca_path)

var_ratio = pca.explained_variance_ratio_
cum_var   = np.cumsum(var_ratio)

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.step(range(1, len(cum_var) + 1), cum_var, where="mid")
ax.scatter(range(1, len(cum_var) + 1), cum_var, s=25)

ax.axhline(0.95, ls="--", color="red", lw=1)
ax.text(len(cum_var)*0.8, 0.95 + 0.02, "95 % variance", color="red")

ax.set_xlabel("Number of PCA components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA variance explained curve")
ax.set_ylim(0, 1.01)
ax.set_xlim(1, len(cum_var))

plt.tight_layout()
fig.savefig("plots/pca_variance.png", bbox_inches="tight", dpi=300)
plt.close(fig)