#!/usr/bin/env python3
"""
prepare_unsupervised_datasets.py
--------------------------------
Download the Indonesia heart-attack CSV, map categoricals to numbers,
standard-scale numerics, then create:

  1. scaled  original dataset   ( *_scaled.csv )
  2. PCA components (95 % var.) ( *_pca.csv    )
  3. ICA components (same dims) ( *_ica.csv    )

and save the fitted StandardScaler, PCA, and ICA objects via joblib.
"""

from pathlib import Path
from shutil import copy2
import kagglehub
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# ------------------------------------------------------------------
# 1. download CSV
print("Downloading dataset ...")
DATASET_REF = "ankushpanday2/heart-attack-prediction-in-indonesia"
dataset_dir = Path(kagglehub.dataset_download(DATASET_REF))

csv_files = list(dataset_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV in Kaggle download")
src_csv = csv_files[0]
dst_csv = Path.cwd() / src_csv.name
copy2(src_csv, dst_csv)
print("Copied", dst_csv.name)

# ------------------------------------------------------------------
# 2. read & map categoricals
df = pd.read_csv(dst_csv)

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

# ------------------------------------------------------------------
# 3. drop remaining NaNs
df = df.dropna().reset_index(drop=True)
print("Rows after drop-na:", len(df))

# ------------------------------------------------------------------
# 4. standard-scale numerics (fit on all rows)
num_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

scaled_path = dst_csv.with_name(f"{dst_csv.stem}_scaled.csv")
df.to_csv(scaled_path, index=False)
joblib.dump(scaler, dst_csv.with_name(f"{dst_csv.stem}_scaler.joblib"))
print("Saved scaled CSV and scaler")

# ------------------------------------------------------------------
# 5. PCA (95 % variance)
pca = PCA(random_state=0)
pca_data = pca.fit_transform(df[num_cols])
pca_cols = [f"PC{i+1}" for i in range(pca_data.shape[1])]
df_pca = pd.DataFrame(pca_data, columns=pca_cols)
pca_path = dst_csv.with_name(f"{dst_csv.stem}_pca.csv")
df_pca.to_csv(pca_path, index=False)
joblib.dump(pca, dst_csv.with_name(f"{dst_csv.stem}_pca.joblib"))
print("Saved PCA CSV (components:", pca_data.shape[1], ")")

# ------------------------------------------------------------------
# 6. ICA (same #components as PCA)
ica = FastICA(random_state=0, max_iter=10000)
ica_data = ica.fit_transform(df[num_cols])
ica_cols = [f"IC{i+1}" for i in range(ica_data.shape[1])]
df_ica = pd.DataFrame(ica_data, columns=ica_cols)
ica_path = dst_csv.with_name(f"{dst_csv.stem}_ica.csv")
df_ica.to_csv(ica_path, index=False)
joblib.dump(ica, dst_csv.with_name(f"{dst_csv.stem}_ica.joblib"))
print("Saved ICA CSV")

print("\nArtifacts:")
print(" ", scaled_path)
print(" ", pca_path)
print(" ", ica_path)