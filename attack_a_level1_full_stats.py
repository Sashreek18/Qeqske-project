import json
import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from attack_a_distinguisher import build_dataset as build_dataset_classical
from attack_a_sequence import build_dataset as build_dataset_sequence, load_qrng, mersenne_stream, autocorrelated_stream, holdout_eval
from nn_torch import MODELS

def calc_stats(scores):
    n = len(scores)
    mean = np.mean(scores)
    sd = np.std(scores, ddof=1) if n > 1 else 0.0
    t_crit = scipy.stats.t.ppf(0.975, df=n-1) if n > 1 else 0.0
    margin = t_crit * sd / np.sqrt(n) if n > 0 else 0.0
    ci_low = max(0.0, float(mean - margin))
    ci_high = min(1.0, float(mean + margin))
    return {"mean": float(mean), "sd": float(sd), "ci95": [ci_low, ci_high]}

def run_classical():
    with open("qrng_combined.txt") as f:
        qrng_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]
    with open("mersenne_large_baseline.txt") as f:
        mersenne_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    X, y, feat_names = build_dataset_classical(qrng_numbers, mersenne_numbers, window_size=8)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = ["accuracy", "roc_auc", "precision", "recall", "f1"]
    
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            "n_estimators": 50,
            "cv_procedure": "RepeatedStratifiedKFold(5,5,seed=42)"
        },
        "SVM": {
            "model": SVC(kernel="rbf", probability=False, random_state=42),
            "n_estimators": None,
            "cv_procedure": "RepeatedStratifiedKFold(5,5,seed=42) on 15k subsample"
        }
    }
    
    if HAS_XGB:
        models["XGBoost"] = {
            "model": xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1, eval_metric="logloss"),
            "n_estimators": 50,
            "cv_procedure": "RepeatedStratifiedKFold(5,5,seed=42)"
        }
    else:
        print("XGBoost not installed or missing dependencies (e.g. libomp). Skipping XGBoost.")
        
    results = {}
    for name, config in models.items():
        print(f"Running classical model: {name}")
        model = config["model"]
        
        if name == "SVM":
            # Subsample 15000 for SVM
            np.random.seed(42)
            idx = np.random.permutation(len(X))[:15000]
            X_run, y_run = X[idx], y[idx]
        else:
            X_run, y_run = X, y

        scores = cross_validate(model, X_run, y_run, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Confusion matrix via StratifiedKFold
        cv_cm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name == "SVM":
            y_pred = cross_val_predict(model, X_run, y_run, cv=cv_cm, n_jobs=-1)
        else:
            y_pred = cross_val_predict(model, X_run, y_run, cv=cv_cm, n_jobs=-1)
            
        cm = confusion_matrix(y_run, y_pred).tolist()
        
        results[name] = {
            "accuracy": calc_stats(scores["test_accuracy"]),
            "roc_auc": calc_stats(scores["test_roc_auc"]),
            "precision": calc_stats(scores["test_precision"]),
            "recall": calc_stats(scores["test_recall"]),
            "f1": calc_stats(scores["test_f1"]),
            "confusion_matrix": cm,
            "n_class0": int(np.sum(y_run == 0)),
            "n_class1": int(np.sum(y_run == 1)),
            "total_n": int(len(y_run)),
            "cv_procedure": config["cv_procedure"],
            "n_estimators": config["n_estimators"]
        }
    return results

def holdout_eval_full(model_cls, X, y, seed, epochs=40, hidden=16):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    cut = int(0.75 * len(X))
    tr, te = idx[:cut], idx[cut:]
    model = model_cls(hidden=hidden, lr=5e-3, epochs=epochs, seed=seed)
    model.fit(X[tr], y[tr])
    
    y_true = y[te]
    y_pred = model.predict(X[te])
    y_prob = model.predict_proba(X[te])
    
    acc = accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return acc, roc, prec, rec, f1, y_true, y_pred

def run_sequence():
    qrng = load_qrng()
    mt = mersenne_stream(len(qrng))
    
    windows_count = 400
    window_len = 32
    need = windows_count + window_len + 2
    auto = autocorrelated_stream(need * 4, match=mt)
    
    results = {}
    for name, cls in MODELS.items():
        results[name] = {"acc": [], "roc": [], "prec": [], "rec": [], "f1": []}
        all_y_true = []
        all_y_pred = []
        
        print(f"Running sequence model: {name}")
        for repeat in range(10):
            seed = 42 + repeat
            
            # Positive control check
            margin = 0
            retry_count = 0
            current_seed = seed
            while margin <= 10 and retry_count < 10:
                X_ctrl, y_ctrl = build_dataset_sequence(auto, mt[:len(auto)], min(windows_count, len(auto) // 4), window_len, seed=current_seed)
                acc_ctrl, _, _, _, _, _, _ = holdout_eval_full(cls, X_ctrl, y_ctrl, seed=current_seed, epochs=40)
                base_ctrl = max(np.mean(y_ctrl), 1 - np.mean(y_ctrl))
                margin = (acc_ctrl - base_ctrl) * 100
                
                if margin <= 10:
                    retry_count += 1
                    print(f"  Positive control failed (margin {margin:.2f} <= 10). Retry {retry_count}/10 with new seed...")
                    current_seed += 100
            
            if margin <= 10:
                print(f"  WARNING: Positive control permanently failed after 10 retries. Saving repeat {repeat} anyway but noting poor fit.")
            
            # Real task
            X_real, y_real = build_dataset_sequence(qrng, mt, windows_count, window_len, seed=seed)
            acc, roc, prec, rec, f1, yt, yp = holdout_eval_full(cls, X_real, y_real, seed=seed, epochs=40)
            
            results[name]["acc"].append(acc)
            results[name]["roc"].append(roc)
            results[name]["prec"].append(prec)
            results[name]["rec"].append(rec)
            results[name]["f1"].append(f1)
            all_y_true.extend(yt)
            all_y_pred.extend(yp)
            print(f"  Repeat {repeat}: Acc {acc:.4f}, ROC {roc:.4f}")

        cm = confusion_matrix(all_y_true, all_y_pred).tolist()
        
        final_res = {
            "accuracy": calc_stats(results[name]["acc"]),
            "roc_auc": calc_stats(results[name]["roc"]),
            "precision": calc_stats(results[name]["prec"]),
            "recall": calc_stats(results[name]["rec"]),
            "f1": calc_stats(results[name]["f1"]),
            "confusion_matrix": cm,
            "n_class0": int(np.sum(np.array(all_y_true) == 0)),
            "n_class1": int(np.sum(np.array(all_y_true) == 1)),
            "total_n": len(all_y_true),
            "cv_procedure": "10 Repeated Holdouts (75/25, seed=42..51)",
            "n_estimators": "N/A"
        }
        results[name] = final_res
        
    return results

if __name__ == "__main__":
    print("Starting Task 1: Classical Models")
    res_class = run_classical()
    print("Starting Task 1: Sequence Models")
    res_seq = run_sequence()
    
    final_json = {**res_class, **res_seq}
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/attack_a_level1_full_stats.json", "w") as f:
        json.dump(final_json, f, indent=2)
    print("Saved results/attack_a_level1_full_stats.json")
