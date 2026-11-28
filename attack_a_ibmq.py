#!/usr/bin/env python3
"""
Multi-QRNG Attack A Distinguisher (HCL IBMQ vs. MT19937 and IBMQ vs. ANU)
===========================================================================
Reuses attack_a_distinguisher (15 window features, window_size=8) and
attack_a_sequence + nn_from_scratch (LSTM, GRU, Transformer, 32-step z-norm windows)
to evaluate:
  1. ibmq_vs_mersenne: HCL IBMQ QRNG vs. Mersenne Twister MT19937 (589,193 draws each)
  2. ibmq_vs_anu:      HCL IBMQ Superconducting QRNG vs. ANU Optical QRNG (589,193 draws each)

Outputs: results/attack_a_ibmq.json
"""

import json
import os
import time
import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from attack_a_distinguisher import build_dataset as build_dataset_classical
from attack_a_sequence import build_dataset as build_dataset_sequence
from env_metadata import get_environment_metadata
from nn_from_scratch import MODELS


def load_csv_ints(path: str, limit: int | None = None) -> list[int]:
    with open(path, "r", encoding="ascii") as f:
        vals = [int(x.strip()) for x in f.read().split(",") if x.strip()]
    return vals[:limit] if limit is not None else vals


def calc_stats(scores):
    n = len(scores)
    mean = float(np.mean(scores))
    sd = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    t_crit = float(scipy.stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    margin = t_crit * sd / np.sqrt(n) if n > 0 else 0.0
    return {"mean": mean, "sd": sd, "ci95": [max(0.0, mean - margin), min(1.0, mean + margin)]}


def eval_pair_classical(stream_a: list[int], stream_b: list[int], seed: int = 42) -> dict:
    X, y, _ = build_dataset_classical(stream_a, stream_b, window_size=8)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    scoring = ["accuracy", "roc_auc", "precision", "recall", "f1"]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=False, random_state=seed),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=50, random_state=seed, n_jobs=-1, eval_metric="logloss")

    out = {"total_windows": int(len(X)), "n_class1": int(np.sum(y == 1)), "n_class0": int(np.sum(y == 0)), "models": {}}
    rng = np.random.RandomState(seed)
    svm_idx = rng.permutation(len(X))[:15000]

    for mname, clf in models.items():
        if mname == "SVM":
            Xr, yr = X[svm_idx], y[svm_idx]
        else:
            Xr, yr = X, y
        scores = cross_validate(clf, Xr, yr, cv=cv, scoring=scoring, n_jobs=-1)
        out["models"][mname] = {
            "accuracy": calc_stats(scores["test_accuracy"]),
            "roc_auc": calc_stats(scores["test_roc_auc"]),
            "precision": calc_stats(scores["test_precision"]),
            "recall": calc_stats(scores["test_recall"]),
            "f1": calc_stats(scores["test_f1"]),
            "n_samples_eval": int(len(Xr)),
            "cv": "RepeatedStratifiedKFold(5, 5, seed=42)",
        }
        print(f"    {mname:<14s} acc={out['models'][mname]['accuracy']['mean']*100:.2f}% auc={out['models'][mname]['roc_auc']['mean']:.4f}")
    return out


def eval_pair_sequence(stream_a: list[int], stream_b: list[int], windows_count: int = 400, window_len: int = 32, epochs: int = 40) -> dict:
    a_arr = np.asarray(stream_a, dtype=float)
    b_arr = np.asarray(stream_b, dtype=float)
    out = {}
    for mname, cls in MODELS.items():
        accs, rocs = [], []
        for rep in range(5):
            seed = 42 + rep
            X, y = build_dataset_sequence(a_arr, b_arr, windows_count, window_len, seed=seed, per_window_z=True)
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(X))
            cut = int(0.75 * len(X))
            tr, te = idx[:cut], idx[cut:]
            model = cls(hidden=16, lr=5e-3, epochs=epochs, seed=seed)
            model.fit(X[tr], y[tr])
            yp = model.predict(X[te])
            yprob = model.predict_proba(X[te])
            accs.append(float(accuracy_score(y[te], yp)))
            try:
                rocs.append(float(roc_auc_score(y[te], yprob)))
            except ValueError:
                rocs.append(0.5)
        out[mname] = {
            "accuracy": calc_stats(accs),
            "roc_auc": calc_stats(rocs),
            "cv": "5 Repeated Holdouts (75/25, seed=42..46)",
            "windows_per_class": windows_count,
            "window_len": window_len,
            "epochs": epochs,
        }
        print(f"    {mname:<14s} acc={out[mname]['accuracy']['mean']*100:.2f}% auc={out[mname]['roc_auc']['mean']:.4f}")
    return out


def main():
    t0 = time.perf_counter()
    os.makedirs("results", exist_ok=True)
    ibmq = load_csv_ints("ibmq_qrng_uint16.txt")
    anu = load_csv_ints("qrng_combined.txt")
    mt = load_csv_ints("mersenne_large_baseline.txt")

    n_mt = min(len(ibmq), len(mt))
    n_qrng = min(len(ibmq), len(anu))

    results = {
        "environment": get_environment_metadata(),
        "counts": {
            "ibmq_total_uint16": len(ibmq),
            "anu_total_uint16": len(anu),
            "mt_total_uint16": len(mt),
            "ibmq_vs_mersenne_matched_per_source": n_mt,
            "ibmq_vs_anu_matched_per_source": n_qrng,
        },
        "tasks": {},
    }

    for task_key, label, sa, sb in [
        ("ibmq_vs_mersenne", f"HCL IBMQ QRNG vs. MT19937 ({n_mt:,}/source)", ibmq[:n_mt], mt[:n_mt]),
        ("ibmq_vs_anu", f"HCL IBMQ Superconducting vs. ANU Optical ({n_qrng:,}/source)", ibmq[:n_qrng], anu[:n_qrng]),
    ]:
        print(f"\n=== Attack A ({label}) ===")
        cls_res = eval_pair_classical(sa, sb)
        seq_res = eval_pair_sequence(sa, sb)
        results["tasks"][task_key] = {
            "label": label,
            "matched_draws_per_source": len(sa),
            "classical": cls_res,
            "sequence": seq_res,
        }

    results["total_wall_seconds"] = float(time.perf_counter() - t0)
    out_path = "results/attack_a_ibmq.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path} in {results['total_wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
