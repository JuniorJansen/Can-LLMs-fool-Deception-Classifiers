"""Generate the three result figures embedded in the thesis.

Reads per-sample results from ``results/`` and writes PDFs to ``figures/``:
  A. asymmetry.pdf       per-direction ASR diverging-bar chart
  B. robust_accuracy.pdf robust accuracy with baseline and chance reference lines
  C. asr_heatmap.pdf     ASR matrix as a heatmap

Run from the project root: ``python -m src.make_figures``
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

STRATEGIES = ["zero_shot", "one_shot", "few_shot"]
STRATEGY_LABEL = {"zero_shot": "ZS", "one_shot": "OS", "few_shot": "FS"}

SMALL_MODELS = [
    ("llama3.2",   "Llama 3.2 (3B)"),
    ("mistral",    "Mistral (7B)"),
    ("gemma3_4b",  "Gemma 3 (4B)"),
    ("qwen2.5_7b", "Qwen 2.5 (7B)"),
]
LARGE_MODELS = [
    ("meta-llama_Llama-3.3-70B-Instruct", "Llama 3.3 (70B)"),
    ("openai_gpt-oss-120b",                "GPT-OSS (120B)"),
]
ALL_MODELS = SMALL_MODELS + LARGE_MODELS

BASELINE_ACC = 0.78
BASELINE_AUC = 0.853
CHANCE = 0.50

# Colours
COL_ZS = "#1E2761"   # main blue
COL_OS = "#4A90D9"   # accent
COL_FS = "#7FB3E0"   # lighter blue
COL_D2T = "#1E2761"  # deceptive->truthful direction
COL_T2D = "#D9534F"  # truthful->deceptive direction
COL_BASELINE = "#444444"
COL_CHANCE = "#999999"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def load(model_key, strategy):
    path = RESULTS_DIR / f"{model_key}_{strategy}.json"
    with open(path) as f:
        return json.load(f)


def asr(records):
    return sum(r["success"] for r in records) / len(records) if records else 0.0


def asr_by_direction(records):
    """Returns (asr_d_to_t, asr_t_to_d) using the classifier's original prediction."""
    dec = [r for r in records if r["original_label"] == 1]
    tru = [r for r in records if r["original_label"] == 0]
    return asr(dec), asr(tru)


def robust_accuracy(records):
    """Final prediction equals true label."""
    correct = 0
    for r in records:
        if r["success"]:
            final = 1 - r["original_label"]
        else:
            final = r["original_label"]
        if final == r["true_label"]:
            correct += 1
    return correct / len(records)


def auc_score(records):
    """AUC of P(deceptive | post-attack text) against the ground-truth label."""
    y_true = [r["true_label"] for r in records]
    p_deceptive = [
        r["final_confidence"] if r["new_label"] == 1 else 1.0 - r["final_confidence"]
        for r in records
    ]
    return roc_auc_score(y_true, p_deceptive)


# --- Figure A: per-direction asymmetry (diverging bars) ---

def fig_asymmetry():
    rows = []
    for model_key, model_name in ALL_MODELS:
        for strat in STRATEGIES:
            data = load(model_key, strat)
            d2t, t2d = asr_by_direction(data)
            rows.append((model_name, STRATEGY_LABEL[strat], d2t, t2d))

    labels = [f"{m}, {s}" for m, s, _, _ in rows]
    d2t_vals = np.array([r[2] for r in rows])
    t2d_vals = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    y = np.arange(len(rows))

    ax.barh(y, d2t_vals, color=COL_D2T, label=r"D$\to$T (deceptive flipped to truthful)",
            edgecolor="white", linewidth=0.4)
    ax.barh(y, -t2d_vals, color=COL_T2D, label=r"T$\to$D (truthful flipped to deceptive)",
            edgecolor="white", linewidth=0.4)

    # Annotate values
    for i, (d, t) in enumerate(zip(d2t_vals, t2d_vals)):
        if d > 0.05:
            ax.text(d + 0.015, i, f"{d:.2f}", va="center", ha="left", fontsize=8, color=COL_D2T)
        if t > 0.05:
            ax.text(-t - 0.015, i, f"{t:.2f}", va="center", ha="right", fontsize=8, color=COL_T2D)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(np.arange(-1.0, 1.05, 0.25))
    ax.set_xticklabels([f"{abs(x):.2f}" for x in np.arange(-1.0, 1.05, 0.25)])
    ax.set_xlabel("Attack success rate")

    # Group separators between models
    for i in range(3, len(rows), 3):
        ax.axhline(i - 0.5, color="lightgray", linewidth=0.6, linestyle="--")
    # Tier separator: between Qwen few-shot and Llama 3.3 zero-shot
    tier_split = len(SMALL_MODELS) * 3 - 0.5
    ax.axhline(tier_split, color="black", linewidth=1.2)

    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Per-direction attack success rate, by attacker model and prompting strategy")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "asymmetry.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote figures/asymmetry.pdf")


# --- Figure B: robust accuracy ---

def fig_robust_accuracy():
    model_names = [name for _, name in ALL_MODELS]
    matrix = np.zeros((len(ALL_MODELS), len(STRATEGIES)))
    for i, (mkey, _) in enumerate(ALL_MODELS):
        for j, strat in enumerate(STRATEGIES):
            data = load(mkey, strat)
            matrix[i, j] = robust_accuracy(data)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(ALL_MODELS))
    width = 0.27
    colours = [COL_ZS, COL_OS, COL_FS]
    labels_s = ["Zero-shot", "One-shot", "Few-shot"]

    for j in range(len(STRATEGIES)):
        bars = ax.bar(x + (j - 1) * width, matrix[:, j], width,
                      color=colours[j], label=labels_s[j], edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.axhline(BASELINE_ACC, color=COL_BASELINE, linewidth=1.2, linestyle="--",
               label=f"Baseline ({BASELINE_ACC:.2f})")
    ax.axhline(CHANCE, color=COL_CHANCE, linewidth=1.2, linestyle=":",
               label=f"Chance ({CHANCE:.2f})")

    # Tier separator
    tier_split = len(SMALL_MODELS) - 0.5
    ax.axvline(tier_split, color="black", linewidth=1.0, alpha=0.4)
    ax.text(tier_split - 0.05, 1.02, "smaller tier", ha="right", va="bottom",
            fontsize=8, color="black", style="italic")
    ax.text(tier_split + 0.05, 1.02, "larger tier", ha="left", va="bottom",
            fontsize=8, color="black", style="italic")

    ax.set_xticks(x, model_names, rotation=20, ha="right")
    ax.set_ylabel("Robust accuracy of DistilBERT classifier")
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.legend(loc="lower left", ncol=2, frameon=True)
    ax.set_title("Robust accuracy under attack, by attacker model and prompting strategy")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robust_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote figures/robust_accuracy.pdf")


# --- Figure B2: robust AUC ---

def fig_robust_auc():
    model_names = [name for _, name in ALL_MODELS]
    matrix = np.zeros((len(ALL_MODELS), len(STRATEGIES)))
    for i, (mkey, _) in enumerate(ALL_MODELS):
        for j, strat in enumerate(STRATEGIES):
            data = load(mkey, strat)
            matrix[i, j] = auc_score(data)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(ALL_MODELS))
    width = 0.27
    colours = [COL_ZS, COL_OS, COL_FS]
    labels_s = ["Zero-shot", "One-shot", "Few-shot"]

    for j in range(len(STRATEGIES)):
        bars = ax.bar(x + (j - 1) * width, matrix[:, j], width,
                      color=colours[j], label=labels_s[j], edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.axhline(BASELINE_AUC, color=COL_BASELINE, linewidth=1.2, linestyle="--",
               label=f"Baseline ({BASELINE_AUC:.3f})")
    ax.axhline(CHANCE, color=COL_CHANCE, linewidth=1.2, linestyle=":",
               label=f"Random ranker ({CHANCE:.2f})")

    tier_split = len(SMALL_MODELS) - 0.5
    ax.axvline(tier_split, color="black", linewidth=1.0, alpha=0.4)
    ax.text(tier_split - 0.05, 1.02, "smaller tier", ha="right", va="bottom",
            fontsize=8, color="black", style="italic")
    ax.text(tier_split + 0.05, 1.02, "larger tier", ha="left", va="bottom",
            fontsize=8, color="black", style="italic")

    ax.set_xticks(x, model_names, rotation=20, ha="right")
    ax.set_ylabel("AUC of DistilBERT classifier")
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.legend(loc="lower left", ncol=2, frameon=True)
    ax.set_title("AUC under attack, by attacker model and prompting strategy")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robust_auc.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote figures/robust_auc.pdf")


# --- Figure C: ASR heatmap ---

def fig_asr_heatmap():
    model_names = [name for _, name in ALL_MODELS]
    matrix = np.zeros((len(ALL_MODELS), len(STRATEGIES)))
    for i, (mkey, _) in enumerate(ALL_MODELS):
        for j, strat in enumerate(STRATEGIES):
            data = load(mkey, strat)
            matrix[i, j] = asr(data)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=0.75)

    ax.set_xticks(np.arange(len(STRATEGIES)),
                  [STRATEGY_LABEL[s] for s in STRATEGIES])
    ax.set_yticks(np.arange(len(ALL_MODELS)), model_names)

    # Tier separator
    tier_split = len(SMALL_MODELS) - 0.5
    ax.axhline(tier_split, color="black", linewidth=1.5)

    # Annotate cells
    for i in range(len(ALL_MODELS)):
        for j in range(len(STRATEGIES)):
            val = matrix[i, j]
            colour = "white" if val > 0.40 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=colour, fontsize=10, fontweight="bold")

    ax.set_xlabel("Prompting strategy")
    ax.set_title("Attack success rate by attacker model and prompting strategy")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("ASR")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "asr_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote figures/asr_heatmap.pdf")


# --- main ---

def main():
    fig_asymmetry()
    fig_robust_accuracy()
    fig_robust_auc()
    fig_asr_heatmap()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
