#!/usr/bin/env python3
"""
prepare_unsupervised_datasets.py
--------------------------------
Creates three numeric data sets (scaled, PCA, ICA) and
splits each of them into an 80/20 stratified train / test pair.
"""

from pathlib import Path
from shutil import copy2
import kagglehub
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
import numpy as np
# ------------------------------------------------------------------
# 1. download raw CSV
print("Downloading dataset …")
DATASET_REF = "ankushpanday2/heart-attack-prediction-in-indonesia"
dataset_dir = Path(kagglehub.dataset_download(DATASET_REF))

csv_files = list(dataset_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV found in the Kaggle folder")
src_csv = csv_files[0]
dst_csv = Path.cwd() / src_csv.name
copy2(src_csv, dst_csv)
print("Copied raw file →", dst_csv.name)

# ------------------------------------------------------------------
# 2. value-map categoricals → numeric df
df = pd.read_csv(dst_csv, keep_default_na=False)       # "None" stays a string
df.replace({'': np.nan}, inplace=True)                 # still mark empty cells as NA

VALUE_MAP = {
    "gender": {"Male": 1.0, "Female": 0.0},
    "region": {"Rural": 0.0, "Urban": 1.0},
    "income_level": {"Low": 0.0, "Middle": 0.5, "High": 1.0},
    "smoking_status": {"Never": 0.0, "Past": 0.5, "Current": 1.0},
    "alcohol_consumption": {"None": 0.0, "Moderate": 0.5, "High": 1.0},
    "physical_activity": {"Low": 0.0, "Moderate": 0.5, "High": 1.0},
    "dietary_habits": {"Unhealthy": 0.0, "Healthy": 1.0},
    "air_pollution_exposure": {"Low": 0.0, "Moderate": 0.5, "High": 1.0},
    "stress_level": {"Low": 0.0, "Moderate": 0.5, "High": 1.0},
    "EKG_results": {"Normal": 0.0, "Abnormal": 1.0},
}
for col, mp in VALUE_MAP.items():
    if col in df.columns:
        df[col] = df[col].map(mp)

df = df.dropna().reset_index(drop=True)

mapped_path = dst_csv.with_name(f"heart_attack_prediction_indonesia.csv")
df.to_csv(mapped_path, index=False)
print("Saved value‑mapped OG CSV →", mapped_path)        # <── NEW

print("Rows after drop-na:", len(df))

# ------------------------------------------------------------------
# 3. split 80 / 20 stratified by heart_attack
LABEL = "heart_attack"
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[LABEL]
)
print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")

# ------------------------------------------------------------------
# 4. StandardScaler (fit on train numerics, apply to both)
num_cols = train_df.select_dtypes(include=[
    "float64", "float32", "int64", "int32"]).columns.drop(LABEL)

scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols]  = scaler.transform(test_df[num_cols])

# save scaled CSVs
stem = dst_csv.stem
scaled_train = dst_csv.with_name(f"{stem}_scaled_train.csv")
scaled_test  = dst_csv.with_name(f"{stem}_scaled_test.csv")
train_df.to_csv(scaled_train, index=False)
test_df.to_csv(scaled_test,  index=False)
joblib.dump(scaler, dst_csv.with_name(f"{stem}_scaler.joblib"))
print("Saved scaled train / test CSVs and scaler.")

# ------------------------------------------------------------------
# 5. PCA (95 % variance) on TRAIN → transform both
pca = PCA(n_components=0.95, random_state=0)
pca_train_mat = pca.fit_transform(train_df[num_cols])
pca_test_mat  = pca.transform(test_df[num_cols])
pca_cols = [f"PC{i+1}" for i in range(pca_train_mat.shape[1])]

pca_train_df = pd.DataFrame(pca_train_mat, columns=pca_cols)
pca_test_df  = pd.DataFrame(pca_test_mat,  columns=pca_cols)
pca_train_df[LABEL] = train_df[LABEL].values
pca_test_df[LABEL]  = test_df[LABEL].values

pca_train = dst_csv.with_name(f"{stem}_pca_train.csv")
pca_test  = dst_csv.with_name(f"{stem}_pca_test.csv")
pca_train_df.to_csv(pca_train, index=False)
pca_test_df.to_csv(pca_test,  index=False)
joblib.dump(pca, dst_csv.with_name(f"{stem}_pca.joblib"))
print("Saved PCA train / test CSVs")

# ------------------------------------------------------------------
# 6. ICA (same n_components) on TRAIN → transform both
# n_comp = pca_train_mat.shape[1]
# ica = FastICA(n_components=n_comp, random_state=0, max_iter=1000)
# ica_train_mat = ica.fit_transform(train_df[num_cols])
# ica_test_mat  = ica.transform(test_df[num_cols])
# ica_cols = [f"IC{i+1}" for i in range(n_comp)]

# ica_train_df = pd.DataFrame(ica_train_mat, columns=ica_cols)
# ica_test_df  = pd.DataFrame(ica_test_mat,  columns=ica_cols)
# ica_train_df[LABEL] = train_df[LABEL].values
# ica_test_df[LABEL]  = test_df[LABEL].values

# ica_train = dst_csv.with_name(f"{stem}_ica_train.csv")
# ica_test  = dst_csv.with_name(f"{stem}_ica_test.csv")
# ica_train_df.to_csv(ica_train, index=False)
# ica_test_df.to_csv(ica_test,  index=False)
# joblib.dump(ica, dst_csv.with_name(f"{stem}_ica.joblib"))
# print("Saved ICA train / test CSVs")

# print("\nArtifacts written:")
# for p in [scaled_train, scaled_test, pca_train, pca_test, ica_train, ica_test]:
#     print(" ", p)
