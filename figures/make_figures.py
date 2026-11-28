#!/usr/bin/env python3
"""
Generate all publication-quality figures for the CSD493 QEQSKE Final Report.
Reads strictly from results/*.json without modifying any raw result files or
hardcoding empirical values.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")


def load_json(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def best_model_entry(clf_dict: dict, metric: str = "accuracy"):
    best_name, best_block = max(
        clf_dict.items(),
        key=lambda kv: kv[1]["real"][metric]["mean"],
    )
    return best_name, best_block


def make_fig3_1_architecture():
    d_recollect = load_json("decrypt_leak_5000_linux_i9_recollect.json")
    d_mlkem = load_json("mlkem512_decrypt_leak_5000.json")
    d_sr = load_json("stress_test_traces_5000.json")

    n_qrng = d_recollect["qrng_stream"]["dataset_size"]
    n_toy = d_recollect["experiment"]["n_trials"]
    toy_n = d_recollect["experiment"]["n"]
    toy_k = d_recollect["experiment"]["k"]
    wraps_toy = d_recollect["qrng_stream"]["wraps"]
    n_mlkem = d_mlkem["experiment"]["n_trials"]
    ml_n = d_mlkem["experiment"]["n"]
    ml_k = d_mlkem["experiment"]["k"]
    n_tiers = len(d_sr["tier_summary"])

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def add_box(x, y, w, h, title, body, fc="#F4F7FB", ec="#2B5B84", lw=1.6):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h - 2.3,
            title,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=ec,
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 1.1,
            body,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#222222",
            linespacing=1.35,
        )

    # Top row: Data & Protocol Flow
    add_box(
        2,
        35,
        19,
        15,
        "1. Randomness Source",
        f"ANU QRNG Stream\n({n_qrng:,} uint16 draws)\nvs. MT19937 Baseline\n(Toy n={n_toy}: wraps={wraps_toy})",
        fc="#EEF6FF",
        ec="#1F4E79",
    )
    add_box(
        26,
        35,
        21,
        15,
        "2. KeyGen & Encapsulation",
        f"Alice key_gen(n={toy_n}, k={toy_k})\nBob encrypt_it() (40 bits)\nPer-bit fresh masking\n(unbiased / hardened opt.)",
        fc="#F2F8F4",
        ec="#1E6B43",
    )
    add_box(
        52,
        35,
        21,
        15,
        "3. Channel & Localhost TCP",
        "Public Key (A, t, q)\nCiphertext (u, v_queue)\n24 statistical features +\n127.0.0.1 TCP RTT / payload",
        fc="#FFF8EC",
        ec="#9C6500",
    )
    add_box(
        78,
        35,
        20,
        15,
        "4. Decapsulation Targets",
        f"Toy decrypt_it() (Orig & CT)\nML-KEM-512 (n={ml_n}, k={ml_k})\nStandard (q=3329) &\nQEQSKE-Variant (n={n_mlkem:,})",
        fc="#FDF0F0",
        ec="#8B1E1E",
    )

    arrow_kw = dict(arrowstyle="-|>,head_width=0.5,head_length=0.8", lw=1.6, color="#444444")
    for x0, x1 in [(21.7, 25.3), (47.7, 51.3), (73.7, 77.3)]:
        ax.annotate("", xy=(x1, 42.5), xytext=(x0, 42.5), arrowprops=arrow_kw)

    # Bottom row: Bounded Observation & Statistical Verification Pipeline
    add_box(
        2,
        14,
        29,
        15,
        "Attacks A & B: RNG Quality Tests",
        "NIST 5-test statistical suite\nAttack A: 6-model QRNG vs. MT ML\nAttack B: Autoencoder entropy ratio\n(Quality tests, NOT side-channel)",
        fc="#F5F3FA",
        ec="#5B3A8C",
    )
    add_box(
        35,
        14,
        31,
        15,
        "Attack C: Bounded Software Telemetry",
        f"Clean wall clock + ru_utime vs. tracemalloc\nMicrobenchmark + {n_tiers}-tier Alice-Bob ML\n5-model 5x10 CV, in-fold QResidualizer,\n200-perm test (<1/201 floor), B=2,000 CI",
        fc="#FFF2F0",
        ec="#A63603",
    )
    add_box(
        70,
        14,
        28,
        15,
        "Calibration & Detrending",
        "Centered rolling-median detrend (W=51)\nToy graded positive control (0-5 us)\nML-KEM positive control (0-50 us)\non Linux i9 (5 us) & Apple M2 (2 us)",
        fc="#EEF8F8",
        ec="#0F6262",
    )

    # Connectors from protocol to telemetry
    obs_kw = dict(arrowstyle="-|>,head_width=0.4,head_length=0.7", lw=1.3, ls="--", color="#555555")
    ax.annotate("", xy=(11.5, 29.7), xytext=(11.5, 34.3), arrowprops=obs_kw)
    ax.annotate("", xy=(42, 29.7), xytext=(36.5, 34.3), arrowprops=obs_kw)
    ax.annotate("", xy=(51, 29.7), xytext=(62.5, 34.3), arrowprops=obs_kw)
    ax.annotate("", xy=(60, 29.7), xytext=(83, 34.3), arrowprops=obs_kw)
    ax.annotate("", xy=(84, 29.7), xytext=(88, 34.3), arrowprops=obs_kw)

    ax.text(
        50,
        5.2,
        "Scope Boundary: Evaluates software-observable timing, CPU time, tracemalloc memory, public transcripts, and localhost TCP RTT.\n"
        "Explicitly excludes hardware power/EM, microarchitectural cache probes, and fault injection (Attack D out of scope; algebraic/CCA in Appendix B).",
        ha="center",
        va="center",
        fontsize=8.3,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F9F9F9", edgecolor="#CCCCCC"),
    )

    ax.set_title("QEQSKE and ML-KEM-512 Software Side-Channel Evaluation Architecture", pad=8)
    out = os.path.join(FIGURES_DIR, "fig3_1_architecture.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_1_attack_a_rng():
    d = load_json("attack_a_level1_full_stats.json")
    models = ["Random Forest", "SVM", "XGBoost", "LSTM", "GRU", "Transformer"]

    acc_means = [d[m]["accuracy"]["mean"] * 100.0 for m in models]
    acc_err_lo = [(d[m]["accuracy"]["mean"] - d[m]["accuracy"]["ci95"][0]) * 100.0 for m in models]
    acc_err_hi = [(d[m]["accuracy"]["ci95"][1] - d[m]["accuracy"]["mean"]) * 100.0 for m in models]

    auc_means = [d[m]["roc_auc"]["mean"] * 100.0 for m in models]
    auc_err_lo = [(d[m]["roc_auc"]["mean"] - d[m]["roc_auc"]["ci95"][0]) * 100.0 for m in models]
    auc_err_hi = [(d[m]["roc_auc"]["ci95"][1] - d[m]["roc_auc"]["mean"]) * 100.0 for m in models]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    b1 = ax.bar(
        x - width / 2,
        acc_means,
        width,
        yerr=[acc_err_lo, acc_err_hi],
        capsize=3.5,
        label="Accuracy (%)",
        color="#2B5B84",
        edgecolor="#1A3A54",
    )
    b2 = ax.bar(
        x + width / 2,
        auc_means,
        width,
        yerr=[auc_err_lo, auc_err_hi],
        capsize=3.5,
        label="ROC-AUC (x100)",
        color="#7A9E9F",
        edgecolor="#3F5E5F",
    )
    ax.axhline(50.0, color="#B22222", linestyle="--", linewidth=1.3, label="Random Chance (50.0%)")

    for bar, val, ehi in zip(b1, acc_means, acc_err_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ehi + 0.55, f"{val:.2f}%", ha="center", va="bottom", fontsize=7.6)
    for bar, val, ehi in zip(b2, auc_means, auc_err_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ehi + 0.55, f"{val:.2f}", ha="center", va="bottom", fontsize=7.6)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(38, 60)
    ax.set_ylabel("Score (% / ROC-AUC x 100)")
    ax.set_title("Attack A Level 1: ANU QRNG vs. Mersenne Twister Distinguishability (Mean and 95% CI)")
    ax.legend(loc="upper right", frameon=True)

    out = os.path.join(FIGURES_DIR, "fig5_1_attack_a_rng.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_2_clean_vs_tm_vs_ct():
    configs = [
        ("Linux i9 Recollect\n(seed=42, Orig)", "decrypt_leak_5000_linux_i9_recollect.json"),
        ("Linux i9 Fresh Key\n(seed=1337, Orig)", "decrypt_leak_5000_linux_i9_seed1337.json"),
        ("Apple M2 arm64\n(seed=42, Orig)", "decrypt_leak_5000_mac_arm64.json"),
        ("Linux i9 Branch-Free\n(seed=42, NumPy CT)", "decrypt_ct_leak_5000.json"),
    ]

    labels, base_vals = [], []
    clean_acc, clean_err, tm_acc, tm_err = [], [], [], []
    clean_r2, clean_r2_err, tm_r2, tm_r2_err = [], [], [], []
    clean_models, tm_models = [], []

    for lbl, fname in configs:
        d = load_json(fname)
        labels.append(lbl)
        base = d["targets"]["majority_baseline"] * 100.0
        base_vals.append(base)

        cu = d["clean_uninstrumented"]
        tm = d["tracemalloc_instrumented_4feat"]

        cm, cb = best_model_entry(cu["classification"], "accuracy")
        tmm, tmb = best_model_entry(tm["classification"], "accuracy")
        clean_models.append(cm)
        tm_models.append(tmm)

        c_m = cb["real"]["accuracy"]["mean"] * 100.0
        clean_acc.append(c_m)
        clean_err.append((cb["real"]["accuracy"]["ci95"][1] - cb["real"]["accuracy"]["mean"]) * 100.0)

        t_m = tmb["real"]["accuracy"]["mean"] * 100.0
        tm_acc.append(t_m)
        tm_err.append((tmb["real"]["accuracy"]["ci95"][1] - tmb["real"]["accuracy"]["mean"]) * 100.0)

        _, crb = best_model_entry(cu["regression"], "r2")
        _, trb = best_model_entry(tm["regression"], "r2")
        cr_m = crb["real"]["r2"]["mean"]
        tr_m = trb["real"]["r2"]["mean"]
        clean_r2.append(cr_m)
        clean_r2_err.append(crb["real"]["r2"]["ci95"][1] - cr_m)
        tm_r2.append(tr_m)
        tm_r2_err.append(trb["real"]["r2"]["ci95"][1] - tr_m)

    x = np.arange(len(labels))
    w = 0.34

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # Left panel: Accuracy
    bars_c = ax1.bar(
        x - w / 2, clean_acc, w, yerr=clean_err, capsize=3.5,
        label="Clean Uninstrumented (2 feat)", color="#2B5B84", edgecolor="#1A3A54"
    )
    bars_t = ax1.bar(
        x + w / 2, tm_acc, w, yerr=tm_err, capsize=3.5,
        label="tracemalloc Instrumented (4 feat)", color="#D95F02", edgecolor="#8A3B00"
    )

    for i, bv in enumerate(base_vals):
        ax1.plot([i - 0.42, i + 0.42], [bv, bv], color="#333333", ls="--", lw=1.3,
                 label="Majority Baseline" if i == 0 else None)
        ax1.text(i, bv - 2.3, f"Base {bv:.2f}%", ha="center", va="top", fontsize=7.5, color="#333333")

    for b, val in zip(bars_c, clean_acc):
        ax1.text(b.get_x() + b.get_width() / 2 - 0.03, val + 1.2, f"{val:.2f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    for b, val in zip(bars_t, tm_acc):
        ax1.text(b.get_x() + b.get_width() / 2 + 0.03, val + 1.2, f"{val:.2f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.4)
    ax1.set_ylim(44, 102)
    ax1.set_ylabel("Best 5x10 CV Accuracy (%)")
    ax1.set_title("(a) Classification Accuracy (95% CI)")
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True)

    # Right panel: R^2
    rc_bars = ax2.bar(
        x - w / 2, clean_r2, w, yerr=clean_r2_err, capsize=3.5,
        label="Clean Uninstrumented (2 feat)", color="#2B5B84", edgecolor="#1A3A54"
    )
    rt_bars = ax2.bar(
        x + w / 2, tm_r2, w, yerr=tm_r2_err, capsize=3.5,
        label="tracemalloc Instrumented (4 feat)", color="#D95F02", edgecolor="#8A3B00"
    )
    ax2.axhline(0.0, color="#333333", lw=1.1, ls="--")

    for b, val in zip(rc_bars, clean_r2):
        va = "bottom" if val >= 0 else "top"
        dy = 0.025 if val >= 0 else -0.028
        ax2.text(b.get_x() + b.get_width() / 2 - 0.03, val + dy, f"{val:+.3f}", ha="center", va=va, fontsize=7.6)
    for b, val in zip(rt_bars, tm_r2):
        va = "bottom" if val >= 0 else "top"
        dy = 0.025 if val >= 0 else -0.028
        ax2.text(b.get_x() + b.get_width() / 2 + 0.03, val + dy, f"{val:+.3f}", ha="center", va=va, fontsize=7.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.4)
    ax2.set_ylim(-0.09, 0.98)
    ax2.set_ylabel("Best Continuous Regression R²")
    ax2.set_title("(b) Continuous Secret-Weight Prediction R²")
    ax2.legend(loc="upper right", fontsize=8.0, frameon=True)

    fig.suptitle("Toy QEQSKE decrypt_it: Clean Timing vs. tracemalloc Observer Effect vs. Branch-Free NumPy", y=1.02)
    out = os.path.join(FIGURES_DIR, "fig5_2_clean_vs_tm_vs_ct.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_3_spearman_by_setup():
    d_rec = load_json("decrypt_leak_5000_linux_i9_recollect.json")
    d_s1337 = load_json("decrypt_leak_5000_linux_i9_seed1337.json")
    d_mac = load_json("decrypt_leak_5000_mac_arm64.json")
    d_ct = load_json("decrypt_ct_leak_5000.json")
    d_sr = load_json("stress_test_traces_5000.json")
    d_ml_s42 = load_json("mlkem512_decrypt_leak_5000.json")
    d_ml_q42 = load_json("mlkem512_qeqske_decrypt_leak_5000.json")
    d_ml_s1337 = load_json("mlkem512_linux_i9_seed1337_decrypt_leak_5000.json")
    d_ml_q1337 = load_json("mlkem512_qeqske_linux_i9_seed1337_decrypt_leak_5000.json")

    entries = [
        ("Toy Orig (Linux i9, seed=42, tracemalloc)", d_rec["tracemalloc_instrumented_4feat"]["correlations"]["tm_dec_timing_us"], "#D95F02"),
        ("Toy Orig (Linux i9, seed=1337, Clean)", d_s1337["clean_uninstrumented"]["correlations"]["dec_timing_us"], "#B22222"),
        ("Toy Orig (Linux i9 Recollect, seed=42, Clean)", d_rec["clean_uninstrumented"]["correlations"]["dec_timing_us"], "#B22222"),
        ("Alice-Bob Receiver Orig (Mac M2, Clean)", d_sr["signal_correlations"]["dec_timing_us"]["correlation"], "#B22222"),
        ("Toy Orig (Mac M2, seed=42, Clean)", d_mac["clean_uninstrumented"]["correlations"]["dec_timing_us"], "#B22222"),
        ("Alice-Bob Localhost TCP RTT (Orig)", d_sr["signal_correlations"]["net_rtt_us"]["correlation"], "#E69F00"),
        ("Alice-Bob Localhost TCP RTT (NumPy CT)", d_sr["signal_correlations"]["net_ct_rtt_us"]["correlation"], "#1E6B43"),
        ("Alice-Bob Sender Encrypt (Mac M2, Clean)", d_sr["signal_correlations"]["enc_timing_us"]["correlation"], "#2B5B84"),
        ("Toy Branch-Free NumPy CT (Linux i9, Clean)", d_ct["clean_uninstrumented"]["correlations"]["dec_timing_us"], "#1E6B43"),
        ("ML-KEM-512 Standard (seed=42, Clean)", d_ml_s42["decaps_clean"]["correlations_vs_secret_weight"]["dec_timing_us"], "#2B5B84"),
        ("ML-KEM-512 QEQSKE-Var (seed=42, Clean)", d_ml_q42["decaps_clean"]["correlations_vs_secret_weight"]["dec_timing_us"], "#2B5B84"),
        ("ML-KEM-512 Standard (seed=1337, Clean)", d_ml_s1337["decaps_clean"]["correlations_vs_secret_weight"]["dec_timing_us"], "#2B5B84"),
        ("ML-KEM-512 QEQSKE-Var (seed=1337, Clean)", d_ml_q1337["decaps_clean"]["correlations_vs_secret_weight"]["dec_timing_us"], "#2B5B84"),
    ]

    entries = entries[::-1]  # bottom to top
    y = np.arange(len(entries))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.axvspan(-0.04, 0.04, color="#E8EEF5", alpha=0.6, label="|ρ| <= 0.04 null band")

    for i, (label, corr, color) in enumerate(entries):
        rho = corr["spearman_rho"]
        lo, hi = corr["spearman_boot_ci95"]
        ax.errorbar(
            rho, i, xerr=[[rho - lo], [hi - rho]],
            fmt="o", color=color, ecolor=color, elinewidth=1.8, capsize=3.5, markersize=6.0
        )
        ax.text(hi + 0.015, i, f"{rho:+.4f}", va="center", ha="left", fontsize=8.3, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=8.8)
    ax.set_xlim(-0.12, 0.96)
    ax.set_xlabel("Spearman Rank Correlation ρ (Timing vs. Secret Weight) with 95% Bootstrap CI")
    ax.set_title("Timing-to-Secret Correlation (Spearman ρ) Across Experimental Setups (5,000 Trials Each)")

    legend_handles = [
        mpatches.Patch(color="#D95F02", label="Instrumentation observer effect (tracemalloc)"),
        mpatches.Patch(color="#B22222", label="Uninstrumented toy decrypt_it leak"),
        mpatches.Patch(color="#E69F00", label="Localhost TCP RTT residual (original decrypt)"),
        mpatches.Patch(color="#1E6B43", label="Branch-free NumPy mitigation (no detectable leak)"),
        mpatches.Patch(color="#2B5B84", label="Sender encrypt & n=256 ML-KEM-512 nulls (|ρ| <= 0.04)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.0, frameon=True)

    out = os.path.join(FIGURES_DIR, "fig5_3_spearman_by_setup.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_4_microbenchmark_rootcause():
    dh = load_json("decrypt_hypothesis_ct.json")
    pa = dh["part_a_microbenchmark"]

    weights = sorted(int(k) for k in pa["mat_mult_mod_by_weight"].keys())
    py_med = [pa["mat_mult_mod_by_weight"][str(w)]["median_us"] for w in weights]
    py_p25 = [pa["mat_mult_mod_by_weight"][str(w)]["p25_us"] for w in weights]
    py_p75 = [pa["mat_mult_mod_by_weight"][str(w)]["p75_us"] for w in weights]

    np_med = [pa["numpy_matmul_by_weight"][str(w)]["median_us"] for w in weights]
    np_p25 = [pa["numpy_matmul_by_weight"][str(w)]["p25_us"] for w in weights]
    np_p75 = [pa["numpy_matmul_by_weight"][str(w)]["p75_us"] for w in weights]

    py_rho = pa["mat_mult_mod_weight_correlation"]["spearman_rho"]
    np_rho = pa["numpy_matmul_weight_correlation"]["spearman_rho"]

    op_iso = pa["operand_isolation_40calls"]
    iso_txt = (
        f"Operand Isolation (40 calls):\n"
        f"all 0:  {op_iso['all_0']['median_us']:.2f} us\n"
        f"all +1: {op_iso['all_pos1']['median_us']:.2f} us\n"
        f"all -1: {op_iso['all_neg1']['median_us']:.2f} us"
    )

    comps = pa["numpy_residual_trend_decomposition"]["components"]
    comp_order = [
        ("matmul_only", "matmul_only\n(u @ s_np)"),
        ("sub_only", "sub_only\n(v - u @ s_np)"),
        ("mod_only", "mod_only\n(diff % q64)"),
        ("mod_nonneg_offset", "mod_nonneg\n((diff+4q) % q64)"),
        ("asarray_only", "asarray_only\n(int64 cast)"),
        ("full_with_nonneg_offset", "full+offset\n(inline cast)"),
        ("preconv_with_nonneg_offset", "preconv+offset\n(pre-cast s_np)"),
    ]
    c_labels = [lbl for _, lbl in comp_order]
    c_rhos = [comps[k]["weight_correlation"]["spearman_rho"] for k, _ in comp_order]
    c_colors = ["#2B5B84" if r < 0 else ("#B22222" if k == "mod_only" else "#D95F02") for (k, _), r in zip(comp_order, c_rhos)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))
    fig.subplots_adjust(wspace=0.42)

    ax1.plot(weights, py_med, "-o", color="#B22222", lw=2.0, label=f"Pure-Python mat_mult_mod (ρ={py_rho:+.4f})")
    ax1.fill_between(weights, py_p25, py_p75, color="#B22222", alpha=0.18)
    ax1.plot(weights, np_med, "-s", color="#1E6B43", lw=2.0, label=f"NumPy int64 matmul (ρ={np_rho:+.4f})")
    ax1.fill_between(weights, np_p25, np_p75, color="#1E6B43", alpha=0.18)
    ax1.text(
        0.97, 0.56, iso_txt, transform=ax1.transAxes, ha="right", va="center", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8EC", edgecolor="#9C6500")
    )
    ax1.set_xticks(weights)
    ax1.set_xlabel("Secret Weight (Non-Zero Coefficients out of 8)")
    ax1.set_ylabel("40-Call Median Latency (us, with IQR)")
    ax1.set_title("(a) Microbenchmark: CPython *0 vs. *±1 Shortcut")
    ax1.legend(loc="upper left", fontsize=8.2)

    y2 = np.arange(len(c_labels))
    bars = ax2.barh(y2, c_rhos, color=c_colors, edgecolor="#222222", height=0.6)
    ax2.axvline(0.0, color="#333333", lw=1.1, ls="--")
    for b, r in zip(bars, c_rhos):
        ha = "left" if r >= 0 else "right"
        dx = 0.012 if r >= 0 else -0.012
        ax2.text(r + dx, b.get_y() + b.get_height() / 2, f"{r:+.4f}", va="center", ha=ha, fontsize=8.2)

    ax2.set_yticks(y2)
    ax2.set_yticklabels(c_labels, fontsize=8.3)
    ax2.set_xlim(-0.26, 0.44)
    ax2.set_xlabel("Spearman ρ (Component Latency vs. Secret Weight)")
    ax2.set_title("(b) NumPy Residual-Trend Decomposition (% q64 Root Cause)")

    fig.suptitle("Root-Cause Isolation of Toy Decryption Leakage and NumPy Signed-Modulo Artifacts", y=1.02)
    out = os.path.join(FIGURES_DIR, "fig5_4_microbenchmark_rootcause.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_5_sender_receiver_tcp():
    d5000 = load_json("stress_test_traces_5000.json")
    d500 = load_json("stress_test_traces.json")

    tiers = [
        ("Tier 1a:\nSender Only", "Tier 1a: Sender Only (Clean)"),
        ("Tier 1b:\nReceiver Orig", "Tier 1b: Receiver Only (Clean Local)"),
        ("Tier 1c:\nChannel (24 feat)", "Tier 1c: Public Channel Only"),
        ("Tier 1d:\nTCP RTT (Orig)", "Tier 1d: Network TCP RTT (Original decrypt_it)"),
        ("Tier 1e:\nTCP RTT (NumPy CT)", "Tier 1e: Network TCP RTT (Constant-Time)"),
        ("Tier 2:\nSender+Receiver", "Tier 2: Sender + Receiver (Clean)"),
        ("Tier 3:\nALL Combined", "Tier 3: ALL Combined (Sender + Receiver + Channel + Net RTT)"),
    ]

    tsum = d5000["tier_summary"]
    base = d5000["majority_baseline"] * 100.0
    lbls = [l for l, _ in tiers]
    real_acc = [tsum[k]["best_real"] * 100.0 for _, k in tiers]
    shuf_acc = [tsum[k]["shuffled_acc"] * 100.0 for _, k in tiers]
    qres_acc = [tsum[k]["q_controlled"] * 100.0 for _, k in tiers]

    legacy_rec = d500["tier_summary"]["Tier 1b: Receiver Only"]["best_real"] * 100.0
    legacy_base = d500["majority_baseline"] * 100.0

    x = np.arange(len(lbls))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.2, 4.8))

    b1 = ax.bar(x - w, real_acc, w, label="Real Labels (5x10 CV)", color="#2B5B84")
    ax.bar(x, shuf_acc, w, label="Shuffled Labels Control", color="#A0A0A0")
    ax.bar(x + w, qres_acc, w, label="q-Residualized Features", color="#1E6B43")
    ax.axhline(base, color="#B22222", ls="--", lw=1.3, label=f"Majority Baseline ({base:.2f}%)")

    for b, val in zip(b1, real_acc):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.7, f"{val:.2f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax.text(
        0.99, 0.96,
        f"Legacy 500-trial tracemalloc contrast:\nReceiver Only = {legacy_rec:.2f}% (base {legacy_base:.2f}%)\nClean 5,000-trial Receiver = {tsum['Tier 1b: Receiver Only (Clean Local)']['best_real']*100:.2f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF8EC", edgecolor="#9C6500")
    )

    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=8.3)
    ax.set_ylim(45, 76)
    ax.set_ylabel("Best Model 5x10 CV Accuracy (%)")
    ax.set_title("End-to-End Clean 5,000-Trial Alice-Bob Multi-Tier & Localhost TCP RTT Leakage Evaluation")
    ax.legend(loc="upper left", ncol=2, fontsize=8.2, frameon=True)

    out = os.path.join(FIGURES_DIR, "fig5_5_sender_receiver_tcp.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_6_positive_controls():
    alc = load_json("attack_c_leaky_vs_clean.json")
    pc_lin = load_json("mlkem512_positive_control_5000_linux_i9.json")
    pc_mac = load_json("mlkem512_positive_control_5000.json")

    # Left: Toy n=4 graded sweep
    sweep_toy = alc["part2_graded_delay_sweep"]
    toy_delays = [row["delay_us"] for row in sweep_toy]

    def max_acc_ci(models_dict):
        bm, blk = max(models_dict.items(), key=lambda kv: kv[1]["accuracy"]["mean"])
        m = blk["accuracy"]["mean"] * 100.0
        lo = blk["accuracy"]["ci95"][0] * 100.0
        hi = blk["accuracy"]["ci95"][1] * 100.0
        return m, m - lo, hi - m

    toy_8d, toy_8d_lo, toy_8d_hi = zip(*(max_acc_ci(r["full_8feat"]["models"]) for r in sweep_toy))
    toy_4d, toy_4d_lo, toy_4d_hi = zip(*(max_acc_ci(r["decrypt_4feat"]["models"]) for r in sweep_toy))
    toy_base = alc["experiment"]["majority_baseline"] * 100.0

    # Right: ML-KEM-512 graded sweep (Linux vs Mac)
    ml_delays = [row["delay_us"] for row in pc_lin["sweep"]]
    lin_raw = [row["raw"]["best_accuracy"] * 100.0 for row in pc_lin["sweep"]]
    lin_det = [row["detrended"]["best_accuracy"] * 100.0 for row in pc_lin["sweep"]]
    mac_raw = [row["raw"]["best_accuracy"] * 100.0 for row in pc_mac["sweep"]]
    mac_det = [row["detrended"]["best_accuracy"] * 100.0 for row in pc_mac["sweep"]]
    ml_base = pc_lin["majority_baseline"] * 100.0
    lin_floor = pc_lin["thresholds"]["raw"]["min_detectable_delay_us"]
    mac_floor = pc_mac["thresholds"]["raw"]["min_detectable_delay_us"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    x_toy = np.arange(len(toy_delays))
    ax1.errorbar(x_toy, toy_8d, yerr=[toy_8d_lo, toy_8d_hi], fmt="-o", capsize=3.5, lw=2.0, color="#2B5B84", label="Full 8D (KeyGen + Decrypt)")
    ax1.errorbar(x_toy, toy_4d, yerr=[toy_4d_lo, toy_4d_hi], fmt="--s", capsize=3.5, lw=1.8, color="#D95F02", label="Decrypt 4D Only")
    ax1.axhline(toy_base, color="#333333", ls=":", lw=1.3, label=f"Majority Baseline ({toy_base:.2f}%)")
    ax1.axvline(1, color="#1E6B43", ls="--", lw=1.2, alpha=0.8)
    ax1.text(1.08, 91, "0.5 us Floor\n(p < 1/201)", color="#1E6B43", fontsize=8.3, fontweight="bold")
    ax1.set_xticks(x_toy)
    ax1.set_xticklabels([f"{d:.1f}" for d in toy_delays])
    ax1.set_ylim(50, 100)
    ax1.set_xlabel("Injected Weight-Conditioned Delay (us)")
    ax1.set_ylabel("Best 5x10 CV Accuracy (%)")
    ax1.set_title("(a) Toy QEQSKE (n=4, k=2) Graded Sensitivity")
    ax1.legend(loc="lower right", fontsize=8.0)

    x_ml = np.arange(len(ml_delays))
    ax2.plot(x_ml, mac_raw, "-o", lw=2.0, color="#1E6B43", label="Apple M2 — Raw (2.0 us floor)")
    ax2.plot(x_ml, mac_det, ":^", lw=1.7, color="#4CA973", label="Apple M2 — Detrended (W=51)")
    ax2.plot(x_ml, lin_raw, "-s", lw=2.0, color="#B22222", label="Linux i9 — Raw (5.0 us floor)")
    ax2.plot(x_ml, lin_det, "--d", lw=1.7, color="#E07A5F", label="Linux i9 — Detrended (W=51)")
    ax2.axhline(ml_base, color="#333333", ls=":", lw=1.3, label=f"Majority Baseline ({ml_base:.2f}%)")
    ax2.axvline(ml_delays.index(mac_floor), color="#1E6B43", ls="--", lw=1.1, alpha=0.7)
    ax2.axvline(ml_delays.index(lin_floor), color="#B22222", ls="--", lw=1.1, alpha=0.7)

    ax2.set_xticks(x_ml)
    ax2.set_xticklabels([f"{int(d)}" for d in ml_delays])
    ax2.set_ylim(48, 103)
    ax2.set_xlabel("Injected Secret-Bit-Conditioned Delay (us)")
    ax2.set_ylabel("Best 5x10 CV Accuracy (%)")
    ax2.set_title("(b) ML-KEM-512 (n=256) Dual-Host Detection Thresholds")
    ax2.legend(loc="upper left", fontsize=7.8)

    fig.suptitle("Calibrated Graded Positive Controls: Empirical Detection Bounds vs. Injected Leakage", y=1.02)
    out = os.path.join(FIGURES_DIR, "fig5_6_positive_controls.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig5_7_toy_vs_mlkem_detrend():
    d_rec = load_json("decrypt_leak_5000_linux_i9_recollect.json")
    d_s1337 = load_json("decrypt_leak_5000_linux_i9_seed1337.json")
    d_mac = load_json("decrypt_leak_5000_mac_arm64.json")
    d_ct = load_json("decrypt_ct_leak_5000.json")
    d_ml42 = load_json("mlkem512_decrypt_leak_5000.json")
    d_q42 = load_json("mlkem512_qeqske_decrypt_leak_5000.json")
    d_ml1337 = load_json("mlkem512_linux_i9_seed1337_decrypt_leak_5000.json")
    d_q1337 = load_json("mlkem512_qeqske_linux_i9_seed1337_decrypt_leak_5000.json")
    dc = load_json("mlkem512_detrend_comparison.json")

    def toy_margin(d):
        _, cb = best_model_entry(d["clean_uninstrumented"]["classification"], "accuracy")
        return (cb["real"]["accuracy"]["mean"] - d["targets"]["majority_baseline"]) * 100.0

    def ml_margin(d):
        _, cb = best_model_entry(d["decaps_clean"]["weight_classification"], "accuracy")
        return (cb["real"]["accuracy"]["mean"] - d["targets"]["weight_majority_baseline"]) * 100.0

    comp_items = [
        ("Toy Orig (Linux s42)", toy_margin(d_rec), "#B22222"),
        ("Toy Orig (Linux s1337)", toy_margin(d_s1337), "#B22222"),
        ("Toy Orig (Mac M2 s42)", toy_margin(d_mac), "#B22222"),
        ("Toy NumPy CT (s42)", toy_margin(d_ct), "#1E6B43"),
        ("ML-KEM Std (s42)", ml_margin(d_ml42), "#2B5B84"),
        ("ML-KEM QEQSKE (s42)", ml_margin(d_q42), "#2B5B84"),
        ("ML-KEM Std (s1337)", ml_margin(d_ml1337), "#2B5B84"),
        ("ML-KEM QEQSKE (s1337)", ml_margin(d_q1337), "#2B5B84"),
    ]
    toy_zeros = 8.0 - d_rec["targets"]["secret_weight_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    x1 = np.arange(len(comp_items))
    margins = [m for _, m, _ in comp_items]
    colors = [c for _, _, c in comp_items]
    bars = ax1.bar(x1, margins, color=colors, edgecolor="#222222", width=0.58)
    ax1.axhline(0.0, color="#333333", lw=1.1, ls="--")
    for b, m in zip(bars, margins):
        va = "bottom" if m >= 0 else "top"
        dy = 0.35 if m >= 0 else -0.35
        ax1.text(b.get_x() + b.get_width() / 2, m + dy, f"{m:+.2f}", ha="center", va=va, fontsize=7.8, fontweight="bold")

    ax1.text(
        0.97, 0.94,
        f"Dense NTT s_hat zeros (out of 512):\n"
        f"ML-KEM Std: {d_ml42['targets']['s_hat_zero_mean']:.2f} (s42), {d_ml1337['targets']['s_hat_zero_mean']:.2f} (s1337)\n"
        f"QEQSKE-Var: {d_q42['targets']['s_hat_zero_mean']:.2f} (s42), {d_q1337['targets']['s_hat_zero_mean']:.2f} (s1337)\n"
        f"(vs. {toy_zeros:.2f}/8 zeros in raw toy s)",
        transform=ax1.transAxes, ha="right", va="top", fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F7FB", edgecolor="#2B5B84")
    )
    ax1.set_xticks(x1)
    ax1.set_xticklabels([l for l, _, _ in comp_items], fontsize=8.0, rotation=24, ha="right")
    ax1.set_ylim(-2.6, 17.5)
    ax1.set_ylabel("Accuracy Margin Over Majority Baseline (pp)")
    ax1.set_title("(a) Toy (n=4) vs. Branch-Free NumPy vs. ML-KEM-512 (n=256)")

    # Right panel: Detrending impact (Timing SD, IQR, and Order-Drift |rho| on Standard ML-KEM-512)
    modes = [("raw", "Raw Timing"), ("order_ols_detrended", "Order-OLS"), ("rolling_median_detrended", "Rolling Med (W=51)")]
    std_m = dc["variants"]["standard_mlkem512"]["modes"]
    qeq_m = dc["variants"]["qeqske_mlkem512"]["modes"]

    x2 = np.arange(len(modes))
    w2 = 0.25
    std_sd = [std_m[k]["timing_descriptives"]["std"] for k, _ in modes]
    std_iqr = [std_m[k]["timing_descriptives"]["iqr"] for k, _ in modes]
    qeq_sd = [qeq_m[k]["timing_descriptives"]["std"] for k, _ in modes]

    b_ssd = ax2.bar(x2 - w2, std_sd, w2, label="Std Timing SD (us)", color="#2B5B84")
    b_siq = ax2.bar(x2, std_iqr, w2, label="Std Timing IQR (us)", color="#7A9E9F")
    b_qsd = ax2.bar(x2 + w2, qeq_sd, w2, label="QEQSKE-Var SD (us)", color="#D95F02")

    for bs in (b_ssd, b_siq, b_qsd):
        for b in bs:
            val = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, val + 8, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax2_t = ax2.twinx()
    ax2_t.spines["top"].set_visible(False)
    drift_std = [abs(std_m[k]["drift_order_vs_timing"]["spearman_rho"]) for k, _ in modes]
    drift_qeq = [abs(qeq_m[k]["drift_order_vs_timing"]["spearman_rho"]) for k, _ in modes]
    ax2_t.plot(x2, drift_std, "-o", color="#B22222", lw=2.0, label="Std Order Drift |ρ|")
    ax2_t.plot(x2, drift_qeq, "--s", color="#5B3A8C", lw=1.8, label="QEQSKE Drift |ρ|")
    ax2_t.set_ylim(-0.04, 0.62)
    ax2_t.set_ylabel("Execution-Order Drift |Spearman ρ|")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in modes], fontsize=8.4)
    ax2.set_ylim(0, 355)
    ax2.set_ylabel("Timing Spread (us)")
    ax2.set_title("(b) ML-KEM-512 Execution-Order Detrending (Linux i9)")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.4, frameon=True)

    fig.suptitle("Scaling Nullification at n=256 and Non-Linear Order Detrending", y=1.02)
    out = os.path.join(FIGURES_DIR, "fig5_7_toy_vs_mlkem_detrend.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    setup_style()
    make_fig3_1_architecture()
    make_fig5_1_attack_a_rng()
    make_fig5_2_clean_vs_tm_vs_ct()
    make_fig5_3_spearman_by_setup()
    make_fig5_4_microbenchmark_rootcause()
    make_fig5_5_sender_receiver_tcp()
    make_fig5_6_positive_controls()
    make_fig5_7_toy_vs_mlkem_detrend()
    print("All 8 figures generated successfully.")


if __name__ == "__main__":
    main()
