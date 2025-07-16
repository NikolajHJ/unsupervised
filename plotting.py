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


def savefig(fig, name):
    Path("plots").mkdir(exist_ok=True)
    fig.savefig(f"plots/{name}", bbox_inches="tight")
    plt.close(fig)

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

# ------------------------------------------------------------------
# Silhouette-score curves  (k = 2 … 25)
# ------------------------------------------------------------------
sil_csv = RES_DIR / "silhouette_scores.csv"
if sil_csv.exists():
    sil_df = pd.read_csv(sil_csv)

    # put OG, PCA, ICA first in the legend if they exist
    primary_ds = ["OG", "PCA", "ICA"]
    other_ds   = [d for d in sil_df["dataset"].unique() if d not in primary_ds]
    order_ds   = primary_ds + other_ds
    sil_df["dataset"] = pd.Categorical(sil_df["dataset"], order_ds, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=sil_df,
        x="k",
        y="silhouette",
        hue="dataset",
        style="model",
        markers=True,
        dashes=False,
        palette="Set2",
        ax=ax
    )

    ax.set_xlabel("Number of clusters / components (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette curves (train split)")
    ax.set_ylim(0, 1)
    ax.legend(title="Data set / model", ncol=2, fontsize=8, loc="best")

    fig.savefig("plots/silhouette_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
else:
    print("- silhouette_scores.csv not found – skipping silhouette plot")


# ------------------------------------------------------------------
# ARI vs number-of-features (backward elimination curves)
# ------------------------------------------------------------------
curves_csv = Path("selected_datasets/selection_curves.csv")
if curves_csv.exists():
    cur_df = pd.read_csv(curves_csv)

    # sort so that curves run from "all feats" → 1 feature
    cur_df = cur_df.sort_values(["dataset", "model", "n_features"], ascending=[True, True, False])

    # nicer labels for legend
    name_map = {"kmeans": "KMEANS", "gmm": "GMM"}
    cur_df["model"] = cur_df["model"].map(name_map).fillna(cur_df["model"])

    # order datasets (OG, PCA, ICA first)
    primary_ds = ["OG", "PCA", "ICA"]
    other_ds   = [d for d in cur_df["dataset"].unique() if d not in primary_ds]
    order_ds   = primary_ds + other_ds
    cur_df["dataset"] = pd.Categorical(cur_df["dataset"], order_ds, ordered=True)

    # ---- Facet plot: one panel per data set ----
    g = sns.FacetGrid(
        cur_df, col="dataset", col_wrap=3, height=3, sharey=True
    )
    g.map_dataframe(
        sns.lineplot,
        x="n_features",
        y="ari",
        hue="model",
        marker="o",
        dashes=False,
        palette="Set2",
    )
    g.set_axis_labels("Remaining features", "Adjusted Rand Index")
    g.add_legend(title="Model")

    # flip x-axis so leftmost = many feats, rightmost = few feats
    for ax in g.axes.ravel():
        ax.invert_xaxis()

    for ax, title in zip(g.axes.flat, order_ds):
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig("plots/ari_vs_features.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
else:
    print("- selection_curves.csv not found – skipping ARI-vs-features plot")



# ------------------------------------------------------------------
# ARI-vs-features – separate plot for each data set
# ------------------------------------------------------------------
curves_path = Path("selected_datasets/selection_curves.csv")
if curves_path.exists():
    sel_df = pd.read_csv(curves_path)

    # consistent model order / colours
    model_order = ["kmeans", "gmm"]
    palette = sns.color_palette("Set2", n_colors=len(model_order))

    for ds_name in sel_df["dataset"].unique():
        ds_df = sel_df[sel_df["dataset"] == ds_name].copy()

        # sort so x goes from many → few features
        ds_df = ds_df.sort_values(["model", "n_features"], ascending=[True, False])
        ds_df["model"] = pd.Categorical(
            ds_df["model"], categories=model_order, ordered=True
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            data=ds_df,
            x="n_features",
            y="ari",
            hue="model",
            palette=palette,
            marker="o",
            dashes=False,
            ax=ax
        )

        ax.set_xlabel("Number of features kept")
        ax.set_ylabel("Adjusted Rand Index (test)")
        ax.set_title(f"{ds_name} – ARI vs. remaining features")
        ax.invert_xaxis()              # left = many features, right = few
        ax.set_ylim(0, 0.2)
        ax.legend(title="Model")

        savefig(fig, f"ari_vs_features_{ds_name}.png")

else:
    print("selection_curves.csv not found – skipping ARI-vs-features plots")


sns.set(style="whitegrid", font_scale=0.8)

DATA_STEMS = {
    "OG"        : "heart_attack_prediction_indonesia",
    "OG‑scaled" : "heart_attack_prediction_indonesia_scaled",
    "PCA"       : "heart_attack_prediction_indonesia_pca",
    "ICA"       : "heart_attack_prediction_indonesia_ica",
}

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.ravel()

vmax = None                                 # auto‑scale each panel
for ax, (name, stem) in zip(axes, DATA_STEMS.items()):
    csv = Path(f"{stem}_train.csv")         # use TRAIN split
    if stem == "heart_attack_prediction_indonesia":
        csv = Path(f"{stem}.csv")
    
    if not csv.exists():
        ax.axis("off")
        ax.set_title(f"{name} (missing)")
        continue

    df = pd.read_csv(csv)
    X = df.drop(columns=["heart_attack"], errors="ignore").values
    cov = np.cov(X, rowvar=False)

    sns.heatmap(
        cov,
        ax=ax,
        cmap="vlag",
        center=0,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        vmax=vmax,
    )
    ax.set_title(name)

# one common color‑bar
fig.tight_layout(rect=[0, 0, 0.9, 1])
cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
sns.heatmap(
    np.concatenate([np.cov(pd.read_csv(f"{stem}_train.csv")
                           .drop(columns=["heart_attack"], errors="ignore").values,
                           rowvar=False).ravel()[:, None]
                    for stem in DATA_STEMS.values() if Path(f"{stem}_train.csv").exists()]),
    cmap="vlag",
    center=0,
    cbar=True,
    cbar_ax=cax,
    vmin=np.min(cov),
    vmax=np.max(cov),
)
cax.set_ylabel("Covariance")

Path("plots").mkdir(exist_ok=True)
fig.savefig("plots/covariance_matrices.png", dpi=300, bbox_inches="tight")

plt.close(fig)

# --- load the scaled CSV (train or test) ---
# adjust path if needed
df = pd.read_csv("heart_attack_prediction_indonesia_scaled_train.csv")

# --- plot ---
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df,
    x="blood_pressure_systolic",
    y="blood_pressure_diastolic",
    hue="heart_attack",
    palette={0: "C0", 1: "C1"},
    alpha=0.6,
    edgecolor=None,
)
plt.xlabel("Systolic blood pressure (standardized)")
plt.ylabel("Diastolic blood pressure (standardized)")
plt.title("BP Systolic vs Diastolic (scaled data)")
plt.legend(title="Heart attack", loc="upper left")
plt.tight_layout()

# save to your plots folder
plt.savefig("plots/bp_scatter.png", dpi=300)
plt.close()