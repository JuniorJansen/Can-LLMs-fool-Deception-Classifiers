import argparse
import json
import os
from scipy import stats
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"
STRATEGIES = ["zero_shot", "one_shot", "few_shot"]

# Robustness check: when True, every test runs only on the narratives the
# classifier originally got right. Flipped on by the --correct flag in main().
CORRECT_ONLY = False

SMALL_MODELS = {
    "llama3.2":    "Llama 3.2 (3B)",
    "mistral":     "Mistral (7B)",
    "gemma3_4b":   "Gemma 3 (4B)",
    "qwen2.5_7b":  "Qwen 2.5 (7B)",
}
LARGE_MODELS = {
    "meta-llama_Llama-3.3-70B-Instruct": "Llama 3.3 (70B)",
    "openai_gpt-oss-120b":                    "GPT-OSS (120B)",
}
ALL_MODELS = {**SMALL_MODELS, **LARGE_MODELS}


def load_results(model_key, strategy):
    path = os.path.join(RESULTS_DIR, f"{model_key}_{strategy}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if CORRECT_ONLY:
        # keep only the narratives the classifier originally labelled correctly
        data = [r for r in data if r["original_label"] == r["true_label"]]
    return data


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mcnemar_test(success_a, success_b):
    """McNemar's test with continuity correction for paired binary data."""
    b = sum(1 for a, b in zip(success_a, success_b) if a and not b)
    c = sum(1 for a, b in zip(success_a, success_b) if not a and b)
    if b + c == 0:
        return float("nan"), float("nan")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p


def chi2_proportions(n1, s1, n2, s2):
    """Chi-squared test comparing two independent proportions."""
    table = [[s1, n1 - s1], [s2, n2 - s2]]
    # Use Fisher's exact if any cell < 5
    if min(s1, n1 - s1, s2, n2 - s2) < 5:
        _, p = stats.fisher_exact(table)
        return float("nan"), p, "Fisher"
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return chi2, p, "Chi2"


# --- H1: Model Size Effect ---

def h1_model_size_test():
    print("\n" + "=" * 60)
    print("H1: MODEL SIZE EFFECT (small vs large, unpaired Chi-squared)")
    print("=" * 60)

    n_tests = len(SMALL_MODELS) * len(LARGE_MODELS) * len(STRATEGIES)
    bonferroni = 0.05 / n_tests

    for strategy in STRATEGIES:
        print(f"\n  Strategy: {strategy}")
        for small_key, small_name in SMALL_MODELS.items():
            small_data = load_results(small_key, strategy)
            if small_data is None:
                continue
            n_small = len(small_data)
            s_small = sum(r["success"] for r in small_data)

            for large_key, large_name in LARGE_MODELS.items():
                large_data = load_results(large_key, strategy)
                if large_data is None:
                    continue
                n_large = len(large_data)
                s_large = sum(r["success"] for r in large_data)

                chi2, p, test = chi2_proportions(n_small, s_small, n_large, s_large)
                adj = "*" if p < bonferroni else "ns"
                asr_s = round(s_small / n_small, 3)
                asr_l = round(s_large / n_large, 3)
                print(f"    {small_name} (ASR={asr_s}) vs {large_name} (ASR={asr_l}): "
                      f"{test} p={p:.4f} {sig_stars(p)} [Bonferroni adj: {adj}]")


# --- H2: Architecture Effect ---

def h2_architecture_test():
    print("\n" + "=" * 60)
    print("H2: ARCHITECTURE EFFECT (Llama 70B vs GPT-OSS 120B, paired McNemar)")
    print("=" * 60)

    llama_key = "meta-llama_Llama-3.3-70B-Instruct"
    gptoss_key = "openai_gpt-oss-120b"
    bonferroni = 0.05 / len(STRATEGIES)

    for strategy in STRATEGIES:
        llama_data = load_results(llama_key, strategy)
        gptoss_data = load_results(gptoss_key, strategy)
        if llama_data is None or gptoss_data is None:
            print(f"  {strategy}: missing data, skipping")
            continue

        llama_success = [r["success"] for r in llama_data]
        gptoss_success = [r["success"] for r in gptoss_data]

        chi2, p = mcnemar_test(llama_success, gptoss_success)
        adj = "*" if p < bonferroni else "ns"
        asr_l = round(sum(llama_success) / len(llama_success), 3)
        asr_g = round(sum(gptoss_success) / len(gptoss_success), 3)
        print(f"  {strategy}: Llama ASR={asr_l} vs GPT-OSS ASR={asr_g} | "
              f"McNemar χ²={chi2:.3f}, p={p:.4f} {sig_stars(p)} [Bonferroni adj: {adj}]")


# --- H3: Strategy Effect ---

def h3_strategy_test():
    print("\n" + "=" * 60)
    print("H3: STRATEGY EFFECT (pairwise McNemar per model, Bonferroni across 3 contrasts)")
    print("=" * 60)

    pairs = [("zero_shot", "one_shot"), ("zero_shot", "few_shot"), ("one_shot", "few_shot")]
    bonferroni = 0.05 / len(pairs)

    for model_key, model_name in ALL_MODELS.items():
        datasets = {s: load_results(model_key, s) for s in STRATEGIES}
        if any(v is None for v in datasets.values()):
            print(f"\n  {model_name}: missing data, skipping")
            continue

        asrs = [round(sum(r["success"] for r in datasets[s]) / len(datasets[s]), 3) for s in STRATEGIES]
        print(f"\n  {model_name}: ASR zero={asrs[0]}, one={asrs[1]}, few={asrs[2]}")

        for s1, s2 in pairs:
            chi2, p_mc = mcnemar_test(
                [r["success"] for r in datasets[s1]],
                [r["success"] for r in datasets[s2]],
            )
            adj = "*" if p_mc < bonferroni else "ns"
            print(f"    {s1} vs {s2}: McNemar χ²={chi2:.3f}, p={p_mc:.4f} "
                  f"{sig_stars(p_mc)} [Bonferroni adj: {adj}]")


# --- H4: Class Asymmetry ---

def h4_class_asymmetry_test():
    print("\n" + "=" * 60)
    print("H4: CLASS ASYMMETRY (deceptive→truthful vs truthful→deceptive)")
    print("=" * 60)

    for model_key, model_name in ALL_MODELS.items():
        print(f"\n  {model_name}")
        for strategy in STRATEGIES:
            data = load_results(model_key, strategy)
            if data is None:
                continue

            dec = [r for r in data if r["original_label"] == 1]
            tru = [r for r in data if r["original_label"] == 0]

            n_dec, s_dec = len(dec), sum(r["success"] for r in dec)
            n_tru, s_tru = len(tru), sum(r["success"] for r in tru)

            if n_dec == 0 or n_tru == 0:
                continue

            chi2, p, test = chi2_proportions(n_dec, s_dec, n_tru, s_tru)
            asr_dec = round(s_dec / n_dec, 3)
            asr_tru = round(s_tru / n_tru, 3)
            print(f"    {strategy}: deceptive ASR={asr_dec} (n={n_dec}) vs "
                  f"truthful ASR={asr_tru} (n={n_tru}) | "
                  f"{test} p={p:.4f} {sig_stars(p)}")


# --- Baseline + post-attack AUC ---

def _p_deceptive(label, confidence):
    """Recover P(deceptive=1) from a (predicted label, max-class probability)."""
    return confidence if label == 1 else 1.0 - confidence


def auc_summary():
    print("\n" + "=" * 60)
    print("AUC (threshold-independent ranking of inputs by P(deceptive))")
    print("=" * 60)

    base = load_results("llama3.2", "zero_shot")  # any file works; the sample is fixed
    y_true = [s["true_label"] for s in base]
    base_auc = roc_auc_score(
        y_true,
        [_p_deceptive(s["original_label"], s["original_confidence"]) for s in base],
    )
    print(f"\n  Baseline AUC (on original narratives): {base_auc:.3f}")

    print(f"\n  Robust AUC (on post-attack narratives):")
    print(f"  {'Model':25s}  {'ZS':>6s}  {'OS':>6s}  {'FS':>6s}")
    for model_key, model_name in ALL_MODELS.items():
        cells = []
        for strategy in STRATEGIES:
            data = load_results(model_key, strategy)
            if data is None:
                cells.append("--")
                continue
            y = [s["true_label"] for s in data]
            p = [_p_deceptive(s["new_label"], s["final_confidence"]) for s in data]
            cells.append(f"{roc_auc_score(y, p):.3f}")
        print(f"  {model_name:25s}  {cells[0]:>6s}  {cells[1]:>6s}  {cells[2]:>6s}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Run the H1-H4 significance tests from the thesis on results/.")
    parser.add_argument(
        "--correct", action="store_true",
        help="Robustness check: restrict every test to the narratives the "
             "classifier originally classified correctly.")
    args = parser.parse_args()

    global CORRECT_ONLY
    CORRECT_ONLY = args.correct

    print("\nADVERSARIAL PARAPHRASE ATTACK: STATISTICAL ANALYSIS")
    if CORRECT_ONLY:
        print("Subset: originally-correct predictions only (robustness check)")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")

    h1_model_size_test()
    h2_architecture_test()
    h3_strategy_test()
    h4_class_asymmetry_test()
    auc_summary()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
