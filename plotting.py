#!/usr/bin/env python3
"""
Makes all figures used in the report.

 • grouped bar-chart of test accuracy
 • PCA variance-explained curve
 • silhouette–score curves
 • ARI-vs-features curves
 • covariance-matrix heat-maps (OG / scaled / PCA / ICA / RP)
 • BP-scatter plot
 • ICA component-loading heat-map (ordered)

Assumes artefacts are stored in the ‘datasets/…’ folder hierarchy.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.cluster.hierarchy import linkage, leaves_list

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
PLOTS = Path("plots");  PLOTS.mkdir(exist_ok=True)
RES   = Path("results")

# ----------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------
def savefig(fig: plt.Figure, fname: str, **kw):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / fname, bbox_inches="tight", dpi=300, **kw)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════=
# 1) main bar-chart ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════=
df = pd.read_csv(RES / "unsup_test_scores.csv")      # model,dataset,accuracy,ari
name_map = {"kmeans": "KMEANS", "gmm": "GMM",
            "nn": "NN", "nn+clust": "NN+Clust"}
df["group"] = df["model"].map(name_map)
df["score"] = df["accuracy"]

order_grp  = ["KMEANS", "GMM", "NN", "NN+Clust"]
primary_ds = ["OG", "PCA", "ICA"]
other_ds   = [d for d in df["dataset"].unique() if d not in primary_ds]
order_ds   = primary_ds + other_ds

df["group"]   = pd.Categorical(df["group"],  order_grp,  ordered=True)
df["dataset"] = pd.Categorical(df["dataset"], order_ds, ordered=True)
df = df.sort_values(["group", "dataset"])

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=df, x="group", y="score", hue="dataset",
            palette="Set2", width=0.8, edgecolor="black", ax=ax)
ax.set_ylabel("Test-set accuracy"); ax.set_xlabel("")
ax.set_title("Accuracy comparison across models and data sets")
ax.set_ylim(0, 1); ax.legend(title="Data set", loc="upper left")
savefig(fig, "main_results.png")
print("✓ plots/main_results.png")

# ═════════════════════════════════════════════════════════════════════=
# 2) PCA explained-variance curve
# ═════════════════════════════════════════════════════════════════════=
pca = joblib.load("datasets/pca/pca.joblib")
cum = np.cumsum(pca.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.step(range(1, len(cum)+1), cum, where="mid")
ax.scatter(range(1, len(cum)+1), cum, s=25)
ax.axhline(0.95, ls="--", lw=1, c="red")
ax.text(len(cum)*0.8, 0.97, "95 % variance", color="red")
ax.set(xlabel="Number of PCA components",
       ylabel="Cumulative explained variance",
       title="PCA variance explained curve",
       ylim=(0, 1.01), xlim=(1, len(cum)))
savefig(fig, "pca_variance.png")
print("✓ plots/pca_variance.png")

# ═════════════════════════════════════════════════════════════════════=
# 3) silhouette-score curves
# ═════════════════════════════════════════════════════════════════════=
sil_csv = RES / "silhouette_scores.csv"
if sil_csv.exists():
    sil = pd.read_csv(sil_csv)
    primary_ds = ["OG", "PCA", "ICA"]
    other_ds   = [d for d in sil["dataset"].unique() if d not in primary_ds]
    sil["dataset"] = pd.Categorical(sil["dataset"],
                                    primary_ds + other_ds, ordered=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=sil, x="k", y="silhouette",
                 hue="dataset", style="model",
                 markers=True, dashes=False, palette="Set2", ax=ax)
    ax.set(xlabel="Number of clusters / components (k)",
           ylabel="Silhouette score",
           title="Silhouette curves (train split)",
           ylim=(0, 1))
    ax.legend(title="Data set / model", ncol=2, fontsize=8)
    savefig(fig, "silhouette_curves.png")
    print("✓ plots/silhouette_curves.png")
else:
    print("• silhouette_scores.csv not found – skipped curve")

# ═════════════════════════════════════════════════════════════════════=
# 4) ARI-vs-features (facet + per-dataset)
# ═════════════════════════════════════════════════════════════════════=
sel_csv = Path("selected_datasets/selection_curves.csv")
if sel_csv.exists():
    sel = pd.read_csv(sel_csv).sort_values(
        ["dataset", "model", "n_features"], ascending=[True, True, False])

    sel["model"] = sel["model"].map({"kmeans": "KMEANS", "gmm": "GMM"})
    primary_ds = ["OG", "PCA", "ICA", "RP"]
    other_ds   = [d for d in sel["dataset"].unique() if d not in primary_ds]
    sel["dataset"] = pd.Categorical(sel["dataset"],
                                    primary_ds + other_ds, ordered=True)

    # facet
    g = sns.FacetGrid(sel, col="dataset", col_wrap=2, height=2, sharey=True)
    g.map_dataframe(sns.lineplot, x="n_features", y="ari",
                    hue="model", marker="o", dashes=False, palette="Set2")
    g.set_axis_labels("Remaining features", "Adjusted Rand Index")
    g.add_legend(title="Model")
    for ax in g.axes.ravel():
        ax.invert_xaxis()
    savefig(g.fig, "ari_vs_features.png")
    print("✓ plots/ari_vs_features.png")

    # per-dataset
    for dsn in sel["dataset"].unique():
        sub = sel[sel["dataset"] == dsn]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=sub.sort_values(["model", "n_features"],
                                          ascending=[True, False]),
                     x="n_features", y="ari", hue="model",
                     marker="o", dashes=False, palette="Set2", ax=ax)
        ax.invert_xaxis(); ax.set_ylim(0, 0.20)
        ax.set(xlabel="Number of features kept",
               ylabel="Adjusted Rand Index (test)",
               title=f"{dsn} – ARI vs remaining features")
        ax.legend(title="Model")
        savefig(fig, f"ari_vs_features_{dsn}.png")
else:
    print("• selection_curves.csv not found – skipped ARI plots")

# ═════════════════════════════════════════════════════════════════════=
# 5) covariance-matrix heat-maps
# ═════════════════════════════════════════════════════════════════════=
sns.set(font_scale=0.8)
STEMS = {
    "OG"        : "datasets/og/ha",
    "Scaled"    : "datasets/scaled/ha_scaled",
    "PCA"       : "datasets/pca/ha_pca",
    "ICA"       : "datasets/ica/ha_ica",
    "RP"        : "datasets/rp/ha_rp",
}

fig, axes = plt.subplots(2, 3, figsize=(9, 6)); axes = axes.ravel()
cov_arrays, vmin, vmax = [], None, None

for ax, (lbl, stem) in zip(axes, STEMS.items()):
    csv = Path(f"{stem}_train.csv")
    if not csv.exists():
        ax.axis("off"); ax.set_title(f"{lbl} (missing)"); continue
    X = pd.read_csv(csv).drop(columns=["heart_attack"], errors="ignore").values
    cov = np.cov(X, rowvar=False)
    cov_arrays.append(cov.ravel()[:, None])
    vmin = cov.min() if vmin is None else min(vmin, cov.min())
    vmax = cov.max() if vmax is None else max(vmax, cov.max())
    sns.heatmap(cov, cmap="vlag", center=0, square=True,
                cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(lbl)

# colour-bar
if cov_arrays:
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    sns.heatmap(np.concatenate(cov_arrays),
                cmap="vlag", center=0, cbar=True, cbar_ax=cax,
                vmin=vmin, vmax=vmax)
    cax.set_ylabel("Covariance")

savefig(fig, "covariance_matrices.png")
print("✓ plots/covariance_matrices.png")

# ═════════════════════════════════════════════════════════════════════=
# 6) BP scatter (scaled data)
# ═════════════════════════════════════════════════════════════════════=
fp = Path("datasets/scaled/ha_scaled_train.csv")
if fp.exists():
    df = pd.read_csv(fp)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x="blood_pressure_systolic",
                    y="blood_pressure_diastolic",
                    hue="heart_attack",
                    palette={0: "C0", 1: "C1"},
                    alpha=0.6, edgecolor=None, ax=ax)
    ax.set(xlabel="Systolic BP (standardised)",
           ylabel="Diastolic BP (standardised)",
           title="BP systolic vs diastolic (scaled)")
    ax.legend(title="Heart attack", loc="upper left")
    savefig(fig, "bp_scatter.png")
    print("✓ plots/bp_scatter.png")

# ═════════════════════════════════════════════════════════════════════=
# 7) ICA loading heat-map (ordered)
# ═════════════════════════════════════════════════════════════════════=
ica_job = Path("datasets/ica/ica.joblib")
sc_csv  = Path("datasets/scaled/ha_scaled_train.csv")
if ica_job.exists() and sc_csv.exists():
    ica  = joblib.load(ica_job)
    cols = pd.read_csv(sc_csv).drop(columns=["heart_attack"]).columns
    load = pd.DataFrame(np.abs(ica.mixing_), index=cols,
                        columns=[f"IC{i+1}" for i in range(ica.mixing_.shape[1])])

    priority = [
        "age","gender","region","income_level","hypertension","diabetes",
        "cholesterol_level","obesity","waist_circumference","family_history",
        "smoking_status","alcohol_consumption","physical_activity",
        "dietary_habits","air_pollution_exposure","stress_level",
        "sleep_hours","blood_pressure_systolic","blood_pressure_diastolic",
        "fasting_blood_sugar","cholesterol_hdl","cholesterol_ldl",
        "previous_heart_disease","participated_in_free_screening",
        "EKG_results","triglycerides","medication_usage",
    ]
    priority = [f for f in priority if f in load.index]

    remaining = set(load.columns); ordered = []
    for feat in priority:
        if not remaining: break
        best = load.loc[feat, list(remaining)].idxmax()
        ordered.append(best); remaining.remove(best)
    ordered += list(remaining)
    load = load[ordered]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(load, cmap="rocket_r", linewidths=0.3,
                cbar_kws={"label": "|loading|"}, ax=ax)
    ax.set(title="ICA – components ordered by feature priority",
           xlabel="Independent components", ylabel="Original features")
    savefig(fig, "ica_component_loadings_by_priority.png")
    print("✓ plots/ica_component_loadings_by_priority.png")
else:
    print("• ICA artefacts missing – skipped ICA heat-map")


# ═════════════════════════════════════════════════════════════════════
# 8) RP component‑vs‑feature heat‑map  (no special ordering)
# ═════════════════════════════════════════════════════════════════════
rp_job  = Path("datasets/rp/rp.joblib")        # projection artefact
sc_csv  = Path("datasets/scaled/ha_scaled_train.csv")
if rp_job.exists() and sc_csv.exists():
    rp   = joblib.load(rp_job)
    cols = pd.read_csv(sc_csv).drop(columns=["heart_attack"]).columns

    # GaussianRandomProjection: components_.shape = (n_components, n_features)
    load = pd.DataFrame(np.abs(rp.components_.T),     # rows = features
                        index=cols,
                        columns=[f"RP{i+1}" for i in range(rp.components_.shape[0])])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(load, cmap="rocket_r", linewidths=0.3,
                cbar_kws={"label": "|projection weight|"}, ax=ax)
    ax.set(title="Random Projection – component weight magnitudes",
           xlabel="Random‑projection components", ylabel="Original features")

    savefig(fig, "rp_component_weights.png")
    print("✓ plots/rp_component_weights.png")
else:
    print("• RP artefacts missing – skipped RP heat‑map")


# ═════════════════════════════════════════════════════════════════════=
# 9) scatterplot of blood preasure
# ═════════════════════════════════════════════════════════════════════=

# --- load the scaled CSV (train or test) ---
# adjust path if needed
df = pd.read_csv("datasets/scaled/ha_scaled_train.csv")

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