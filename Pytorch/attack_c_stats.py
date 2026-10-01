"""
ATTACK C (stats utility)  [PyTorch version]
============================================
Exact copy of sk-learn/attack_c_stats.py.
No PyTorch changes needed here — this module only runs sklearn cross_validate
on sklearn-compatible models and computes scipy statistics.
It is included in the Pytorch/ folder so all PyTorch attack files
can import it from a single location without path jumping.
"""

import json
import os
import numpy as np
import scipy.stats
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import permutation_test_score, cross_validate, StratifiedKFold


def update_json_results(filename, key, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    if key not in results:
        results[key] = {}
    results[key].update(data)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def compute_correlation_stats(x, y, num_signals):
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_r": 0.0,
                "spearman_p": 1.0, "ci95": [0.0, 0.0],
                "effect_size": "negligible", "bonferroni_p": 1.0}

    pearson_r,  p_p = scipy.stats.pearsonr(x, y)
    spearman_r, s_p = scipy.stats.spearmanr(x, y)

    r_z = np.arctanh(np.clip(pearson_r, -0.9999, 0.9999))
    se  = 1 / np.sqrt(len(x) - 3)
    z_crit = scipy.stats.norm.ppf(0.975)
    ci_low  = np.tanh(r_z - z_crit * se)
    ci_high = np.tanh(r_z + z_crit * se)

    abs_r = abs(pearson_r)
    if abs_r < 0.1:   mag = "negligible"
    elif abs_r < 0.3: mag = "small"
    elif abs_r < 0.5: mag = "medium"
    else:             mag = "large"

    bonferroni_p = min(1.0, p_p * num_signals)

    return {
        "pearson_r":   float(pearson_r),
        "pearson_p":   float(p_p),
        "spearman_r":  float(spearman_r),
        "spearman_p":  float(s_p),
        "ci95":        [float(ci_low), float(ci_high)],
        "effect_size": mag,
        "bonferroni_p": float(bonferroni_p)
    }


def compute_ml_stats(model, X, y, cv,
                     scoring=("accuracy", "balanced_accuracy", "roc_auc", "f1")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    res = {}
    n = cv.get_n_splits(X, y)
    for met in scoring:
        key = f"test_{met}"
        if key in scores:
            arr    = scores[key]
            mean   = np.mean(arr)
            sd     = np.std(arr, ddof=1)
            t_crit = scipy.stats.t.ppf(0.975, df=n - 1)
            margin = t_crit * sd / np.sqrt(n)
            ci_low  = max(0.0, float(mean - margin))
            ci_high = min(1.0, float(mean + margin))
            res[met] = {"mean": float(mean), "sd": float(sd),
                        "ci95": [ci_low, ci_high]}

    fast_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score, perm_scores, pvalue = permutation_test_score(
        model, X, y, scoring="accuracy", cv=fast_cv,
        n_permutations=200, n_jobs=-1, random_state=42)
    res["permutation_test"] = {"score": float(score), "pvalue": float(pvalue)}

    return res
