#!/usr/bin/env python3
"""
main.py
========
Run the entire **unsupervised** workflow in one go.

Order
-----
1. prepare_unsupervised_datasets.py   – download, map, scale, PCA / ICA / RP
2. unsup_feature_selection_ari.py     – backward‐elimination masks
3. silhouette_grid.py                 – k→silhouette curves
4. add_cluster_features.py            – append KMeans/GMM labels to CSVs
5. unsup_test_evaluate.py             – final test accuracy / ARI
6. plotting.py                        – all figures (bar chart, PCA var, …)

Each script is executed as a separate process; if one fails, execution
stops so you can read the console traceback.
"""
from pathlib import Path
import subprocess
import sys

# ------------------------------------------------------------------
# helper
# ------------------------------------------------------------------
def run(script: Path | str):
    """Run a Python script and stop the pipeline if it fails."""
    script = Path(script)
    print(f"\n\033[96m➜  Running {script}\033[0m")
    try:
        subprocess.check_call([sys.executable, str(script)])
    except subprocess.CalledProcessError as e:
        print(f"\n\033[91m✗ {script} exited with error code {e.returncode}\033[0m")
        sys.exit(e.returncode)


# ------------------------------------------------------------------
# pipeline
# ------------------------------------------------------------------
SCRIPTS = [
    "download_data.py",
    "feature_selector.py",
    "sil_scores.py",
    "clusters_as_input.py",
    "eval.py",
    "plotting.py",                       # your master plotting script
]

if __name__ == "__main__":
    for s in SCRIPTS:
        run(s)

    print("\n\033[92m✓ Pipeline finished without errors.\033[0m")
