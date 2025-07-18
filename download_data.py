#!/usr/bin/env python3
"""
prepare_unsupervised_datasets.py
--------------------------------
• value‑map categoricals → floats
• save OG csv
• create scaled, PCA‑, ICA‑ and RP‑based train / test splits
  (same   n_components for PCA, ICA, RP)
Files are written under   datasets/<subfolder>/…
"""

from pathlib import Path, PurePath
from shutil import copy2
import kagglehub, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

# ------------ folder helpers -----------------------------------------
SAMPLE_N = 10_000          # e.g. 10 000   |  None = use the whole file

BASE = Path("datasets")
DIRS = {name: BASE / name for name in
        ("og", "scaled", "pca", "ica", "rp")}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------

# 1) download ----------------------------------------------------------
print("Downloading dataset …")
DATASET_REF = "ankushpanday2/heart-attack-prediction-in-indonesia"
dataset_dir = Path(kagglehub.dataset_download(DATASET_REF))
src_csv     = next(dataset_dir.glob("*.csv"))
copy2(src_csv, DIRS["og"] / src_csv.name)

# 2) value‑map ---------------------------------------------------------
df = pd.read_csv(src_csv, keep_default_na=False)
df.replace({'': np.nan}, inplace=True)           # treat empty as NA
if SAMPLE_N is not None and SAMPLE_N < len(df):
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=SAMPLE_N, random_state=42
    )
    idx, _ = next(splitter.split(df, df["heart_attack"]))
    df = df.iloc[idx].reset_index(drop=True)
    print(f"↓  subsampled to {SAMPLE_N:,} rows (stratified)")
VALUE_MAP = {
    "gender":                {"Male": 1., "Female": 0.},
    "region":                {"Rural": 0., "Urban": 1.},
    "income_level":          {"Low": 0., "Middle": .5, "High": 1.},
    "smoking_status":        {"Never": 0., "Past": .5, "Current": 1.},
    "alcohol_consumption":   {"None": 0., "Moderate": .5, "High": 1.},
    "physical_activity":     {"Low": 0., "Moderate": .5, "High": 1.},
    "dietary_habits":        {"Unhealthy": 0., "Healthy": 1.},
    "air_pollution_exposure":{"Low": 0., "Moderate": .5, "High": 1.},
    "stress_level":          {"Low": 0., "Moderate": .5, "High": 1.},
    "EKG_results":           {"Normal": 0., "Abnormal": 1.},
}
for col, mp in VALUE_MAP.items():
    if col in df.columns:
        df[col] = df[col].map(mp)

df = df.dropna().reset_index(drop=True)
og_csv = DIRS["og"] / "heart_attack_prediction_indonesia.csv"
df.to_csv(og_csv, index=False)
print("Saved OG value‑mapped →", PurePath(og_csv))

# 3) split -------------------------------------------------------------
LABEL = "heart_attack"
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[LABEL])

# numeric cols
num_cols = train_df.select_dtypes(
    include=["float64", "float32", "int64", "int32"]
).columns.drop(LABEL)

# 4) scaling -----------------------------------------------------------
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols]  = scaler.transform(test_df[num_cols])

stem = "ha"   # short stem keeps file names short
(train_df
 ).to_csv(DIRS["scaled"] / f"{stem}_scaled_train.csv", index=False)
(test_df
 ).to_csv(DIRS["scaled"] / f"{stem}_scaled_test.csv",  index=False)
joblib.dump(scaler, DIRS["scaled"] / "scaler.joblib")
print("scaled/   train, test, scaler.joblib written")

# ---------- helper to dump component‑based sets ----------------------
def dump_component_set(mat_tr, mat_te, cols, folder: Path, tag: str, obj):
    tr = pd.DataFrame(mat_tr, columns=cols); tr[LABEL] = train_df[LABEL].values
    te = pd.DataFrame(mat_te, columns=cols); te[LABEL] = test_df[LABEL].values
    tr.to_csv(folder / f"{stem}_{tag}_train.csv", index=False)
    te.to_csv(folder / f"{stem}_{tag}_test.csv",  index=False)
    joblib.dump(obj, folder / f"{tag}.joblib")

# 5) PCA ---------------------------------------------------------------
pca = PCA(n_components=0.95, random_state=0)
pca_tr = pca.fit_transform(train_df[num_cols])
pca_te = pca.transform(test_df[num_cols])
p_cols = [f"PC{i+1}" for i in range(pca_tr.shape[1])]
dump_component_set(pca_tr, pca_te, p_cols, DIRS["pca"], "pca", pca)
print("pca/      train, test, pca.joblib written")

# 6) ICA ---------------------------------------------------------------
n_comp = pca_tr.shape[1]
ica = FastICA(n_components=n_comp, random_state=0, max_iter=10000)
ica_tr = ica.fit_transform(train_df[num_cols])
ica_te = ica.transform(test_df[num_cols])
i_cols = [f"IC{i+1}" for i in range(n_comp)]
dump_component_set(ica_tr, ica_te, i_cols, DIRS["ica"], "ica", ica)
print("ica/      train, test, ica.joblib written")

# 7) Gaussian Random Projection ---------------------------------------
rp = GaussianRandomProjection(n_components=n_comp, random_state=0)
rp_tr = rp.fit_transform(train_df[num_cols])
rp_te = rp.transform(test_df[num_cols])
r_cols = [f"RP{i+1}" for i in range(n_comp)]
dump_component_set(rp_tr, rp_te, r_cols, DIRS["rp"], "rp", rp)
print("rp/       train, test, rp.joblib written")

print("\n✓ All artefacts saved under 'datasets/'")
