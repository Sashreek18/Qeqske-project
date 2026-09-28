import platform
import os
import sys
import json
import psutil
import numpy
import scipy
import sklearn
try:
    import xgboost
    xgb_version = xgboost.__version__
except Exception:
    xgb_version = "Not Installed"

def collect_info():
    info = {
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgb_version
        },
        "system": {
            "os": platform.platform(),
            "cpu_info": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total
        },
        "random_seeds": {
            "general_seed": 42,
            "attack_a_sequence_holdouts": "42 to 51 (10 repeated holdouts)",
            "all_other_scripts_and_cv": 42,
            "exceptions": "None"
        },
        "cv_procedures": {
            "attack_a_classical": "RepeatedStratifiedKFold(n_splits=5, n_repeats=5)",
            "attack_a_sequence": "10 Repeated Holdout splits (75/25 ratio)",
            "attack_c_combined": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10)",
            "attack_c_progressive": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10)",
            "attack_c_leaky_vs_clean": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10)",
            "system_comparison": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10)"
        },
        "sequence_model_hyperparameters": {
            "hidden_size": 16,
            "learning_rate": 5e-3,
            "epochs": 40,
            "optimizer": "Adam (b1=0.9, b2=0.999, eps=1e-8)",
            "batch_size": 32,
            "early_stopping": "None (trains for all 40 epochs)"
        },
        "qrng_data_collection": {
            "dates": "Not recorded, collected via collect_qrng_data.py across 2 ANU API accounts",
            "source": "Australian National University (ANU) QRNG API"
        },
        "preprocessing": {
            "classical_features_window_size": 8,
            "sequence_model_window_size": 32,
            "sequence_model_normalization": "per-window z-normalization (mean 0, std 1)",
            "classical_features": [
                "mean", "std", "min", "max", "range", "median", "skew",
                "bit_ones_ratio", "autocorr_lag1", "diff_mean", "diff_std",
                "longest_run_bits", "entropy", "mod3_balance", "even_ratio"
            ]
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/reproducibility.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Saved reproducibility info to results/reproducibility.json")

if __name__ == "__main__":
    collect_info()
