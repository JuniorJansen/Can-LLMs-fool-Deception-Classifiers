"""Compute every numeric value in the thesis tables directly from ``results/``.

Single source of truth for the table numbers, so the thesis can never drift
from the raw experiment output again. Prints plain values (ASR, robust
accuracy, AUC, mean confidence change, iterations, per-direction rates,
similarity) together with Wilson 95% confidence intervals and the H1-H4
statistics. Uses the same logic as analyze_stats.py / make_figures.py.

Run from the project root:  python -m src.make_tables
"""

import json
import math
from pathlib import Path
from statistics import mean, median

from scipy import stats
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("results")
STRATEGIES = ["zero_shot", "one_shot", "few_shot"]
SHORT = {"zero_shot": "ZS", "one_shot": "OS", "few_shot": "FS"}

SMALL_MODELS = [
    ("llama3.2", "Llama 3.2 (3B)"),
    ("mistral", "Mistral (7B)"),
    ("gemma3_4b", "Gemma 3 (4B)"),
    ("qwen2.5_7b", "Qwen 2.5 (7B)"),
]
LARGE_MODELS = [
    ("meta-llama_Llama-3.3-70B-Instruct", "Llama 3.3 (70B)"),
    ("openai_gpt-oss-120b", "GPT-OSS (120B)"),
]
ALL_MODELS = SMALL_MODELS + LARGE_MODELS
Z = 1.96

# --- loading + metrics ---

def load(mk, s):
    with open(RESULTS_DIR / f"{mk}_{s}.json") as f:
        return json.load(f)

def asr(rs):
    return sum(r["success"] for r in rs) / len(rs)

def robust_acc(rs):
    return sum(((1 - r["original_label"]) if r["success"] else r["original_label"]) == r["true_label"]
               for r in rs) / len(rs)

def mean_dc(rs):
    return mean(r["max_confidence_change"] for r in rs)

def mean_iters(rs):
    succ = [r["iterations"] for r in rs if r["success"]]
    return mean(succ) if succ else float("nan")

def sim_stats(rs):
    s = [r["final_similarity"] for r in rs]
    return mean(s), median(s)

def auc(rs):
    y = [r["true_label"] for r in rs]
    p = [r["final_confidence"] if r["new_label"] == 1 else 1 - r["final_confidence"] for r in rs]
    return roc_auc_score(y, p)

def baseline_auc():
    rs = load("llama3.2", "zero_shot")
    y = [r["true_label"] for r in rs]
    p = [r["original_confidence"] if r["original_label"] == 1 else 1 - r["original_confidence"] for r in rs]
    return roc_auc_score(y, p)

def directions(rs):
    dec = [r for r in rs if r["original_label"] == 1]
    tru = [r for r in rs if r["original_label"] == 0]
    return asr(dec), asr(tru), len(dec), len(tru)

def correctness_split(rs):
    """ASR split by whether the classifier's original prediction was correct."""
    corr = [r for r in rs if r["original_label"] == r["true_label"]]
    wrong = [r for r in rs if r["original_label"] != r["true_label"]]
    asr_c = asr(corr) if corr else float("nan")
    asr_w = asr(wrong) if wrong else float("nan")
    return asr(rs), asr_c, asr_w, len(corr), len(wrong)

# --- statistics ---

def wilson(p, n, z=Z):
    k = round(p * n); ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = (z / den) * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)

def mcnemar(a, b):
    nb = sum(1 for x, y in zip(a, b) if x and not y)
    nc = sum(1 for x, y in zip(a, b) if not x and y)
    if nb + nc == 0:
        return float("nan"), float("nan"), nb, nc
    chi2 = (abs(nb - nc) - 1) ** 2 / (nb + nc)
    return chi2, 1 - stats.chi2.cdf(chi2, 1), nb, nc

def chi2_prop(n1, s1, n2, s2):
    t = [[s1, n1 - s1], [s2, n2 - s2]]
    chi2, pc, _, _ = stats.chi2_contingency(t, correction=False)
    v = math.sqrt(chi2 / (n1 + n2))
    if min(s1, n1 - s1, s2, n2 - s2) < 5:
        _, p = stats.fisher_exact(t)
        return chi2, p, v, "Fisher"
    return chi2, pc, v, "Chi2"

def ci(p, n):
    lo, hi = wilson(p, n)
    return f"[{lo:.3f}, {hi:.3f}]"

# --- output ---

def main():
    N = 200
    print("=" * 92)
    print("PER-CONDITION VALUES  (computed from results/, N=200 per condition)")
    print("=" * 92)
    print(f"{'Model':16s} {'St':3s} {'ASR':>6s} {'ASR 95%CI':>16s} {'RobAcc':>7s} "
          f"{'RobAcc 95%CI':>16s} {'AUC':>5s} {'MeanDc':>7s} {'Iters':>6s}")
    for mk, mn in ALL_MODELS:
        for s in STRATEGIES:
            d = load(mk, s)
            a, ra = asr(d), robust_acc(d)
            print(f"{mn:16s} {SHORT[s]:3s} {a:6.3f} {ci(a, N):>16s} {ra:7.3f} "
                  f"{ci(ra, N):>16s} {auc(d):5.3f} {mean_dc(d):7.3f} {mean_iters(d):6.2f}")
        print("-" * 92)

    print("\n" + "=" * 70)
    print("PER-DIRECTION ASYMMETRY (D->T vs T->D)")
    print("=" * 70)
    print(f"{'Model':16s} {'St':3s} {'D->T':>6s} {'T->D':>6s} {'chi2':>8s} {'p':>9s} {'V':>5s} {'test':>7s}")
    for mk, mn in ALL_MODELS:
        for s in STRATEGIES:
            d = load(mk, s)
            d2t, t2d, nd, nt = directions(d)
            chi2, p, v, test = chi2_prop(nd, round(d2t * nd), nt, round(t2d * nt))
            print(f"{mn:16s} {SHORT[s]:3s} {d2t:6.3f} {t2d:6.3f} {chi2:8.2f} {p:9.4f} {v:5.2f} {test:>7s}")

    print("\n" + "=" * 70)
    print("FINAL-PARAPHRASE SIMILARITY (over all 200 narratives)")
    print("=" * 70)
    print(f"{'Model':16s} {'St':3s} {'mean':>6s} {'median':>7s}")
    for mk, mn in ALL_MODELS:
        for s in STRATEGIES:
            m, md = sim_stats(load(mk, s))
            print(f"{mn:16s} {SHORT[s]:3s} {m:6.3f} {md:7.3f}")

    print("\n" + "=" * 70)
    print("ASR BY ORIGINAL CORRECTNESS (robustness check)")
    print("=" * 70)
    print(f"{'Model':16s} {'St':3s} {'ASR_all':>8s} {'ASR_corr':>9s} {'ASR_wrong':>10s} {'n_corr':>7s} {'n_wrong':>8s}")
    for mk, mn in ALL_MODELS:
        for s in STRATEGIES:
            a, ac, aw, nc, nw = correctness_split(load(mk, s))
            print(f"{mn:16s} {SHORT[s]:3s} {a:8.3f} {ac:9.3f} {aw:10.3f} {nc:7d} {nw:8d}")

    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)
    print(f"  Baseline accuracy = 0.78   Baseline AUC = {baseline_auc():.3f}")

    print("\n" + "=" * 70)
    print("H2  Llama 3.3 70B vs GPT-OSS 120B (paired McNemar)  Bonferroni alpha=.0167")
    print("    OR = (GPT-OSS only) / (Llama only)")
    print("=" * 70)
    for s in STRATEGIES:
        la = [r["success"] for r in load("meta-llama_Llama-3.3-70B-Instruct", s)]
        gp = [r["success"] for r in load("openai_gpt-oss-120b", s)]
        chi2, p, b, c = mcnemar(la, gp)
        orr = c / b if b else float("inf")
        print(f"  {SHORT[s]:3s} chi2={chi2:6.2f} p={p:.4f} OR={orr:.2f} "
              f"{'SIG' if p < 0.0167 else 'ns'}")

    print("\n" + "=" * 70)
    print("H3  pairwise McNemar within each model  Bonferroni alpha=.0167")
    print("    OR = (first-strategy only) / (second-strategy only)")
    print("=" * 70)
    pairs = [("zero_shot", "one_shot"), ("zero_shot", "few_shot"), ("one_shot", "few_shot")]
    for mk, mn in ALL_MODELS:
        d = {s: [r["success"] for r in load(mk, s)] for s in STRATEGIES}
        print(f"  {mn}")
        for s1, s2 in pairs:
            chi2, p, b, c = mcnemar(d[s1], d[s2])
            orr = b / c if c else float("inf")
            print(f"    {SHORT[s1]} vs {SHORT[s2]}: chi2={chi2:6.2f} p={p:.4f} OR={orr:.2f} "
                  f"{'SIG' if p < 0.0167 else 'ns'}")

    print("\n" + "=" * 70)
    print("H1  small vs large (unpaired chi-squared, N=400)  Bonferroni alpha=.0021")
    print("=" * 70)
    bonf = 0.05 / (len(SMALL_MODELS) * len(LARGE_MODELS) * 3)
    for s in STRATEGIES:
        for sk, sn in SMALL_MODELS:
            sd = load(sk, s)
            for lk, ln in LARGE_MODELS:
                ld = load(lk, s)
                chi2, p, v, test = chi2_prop(len(sd), sum(r["success"] for r in sd),
                                             len(ld), sum(r["success"] for r in ld))
                print(f"  {SHORT[s]:3s} {sn:14s} vs {ln:16s} chi2={chi2:7.2f} p={p:.5f} V={v:.2f} "
                      f"{'SIG' if p < bonf else 'ns'}")

if __name__ == "__main__":
    main()
