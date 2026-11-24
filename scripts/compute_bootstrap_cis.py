"""
Compute bootstrap 95% CIs on Krippendorff Alpha for all experiment cells.
Also computes three-rater Alpha (K, L, LLM) for Tier A dimensions.
Outputs: updated results CSV and summary text.
"""
import json, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from krippendorff import alpha as kripp_alpha

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility", "Semantic_Uniqueness"]
HUMAN_DIM_MAP = {
    "Relevance": ("Relevance_KC", "Relevance_LH"),
    "Clarity": ("Clarity_KC", "Clarity_LH"),
    "Linguistic_Feature": ("Linguistic_Feature_KC", "Linguistic_Feature_LH"),
    "Semantic_Plausibility": ("Semantic_Plausibility_KC", "Semantic_Plausibility_LH"),
    "Semantic_Uniqueness": ("Semantic_Unique_KC", "Semantic_Unique_LH"),
}

OUTPUT_DIR = ROOT / "Data" / "results" / "neurips_experiments"


def load_human_data():
    df = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    for dim, (ck, cl) in HUMAN_DIM_MAP.items():
        df[f"{dim}_consensus"] = df[[ck, cl]].mean(axis=1).round().astype(int)
    return df


def get_llm_median_scores(raw_path: Path, n_items: int):
    """Load raw_scores.json and compute median per dim per item."""
    with open(raw_path) as f:
        raw = json.load(f)
    medians = {}  # uuid -> {dim: median}
    for uuid, dims in raw.items():
        medians[uuid] = {}
        for dim, runs in dims.items():
            vals = [v for v in runs if v is not None]
            medians[uuid][dim] = float(np.median(vals)) if vals else np.nan
    return medians


def bootstrap_alpha(rater1, rater2, n_boot=2000, ci=0.95):
    """Bootstrap CI for Krippendorff alpha between two rater arrays."""
    n = len(rater1)
    alphas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        r1 = rater1[idx]
        r2 = rater2[idx]
        # Skip if no variance
        if np.std(r1) == 0 and np.std(r2) == 0:
            continue
        try:
            a = kripp_alpha([r1, r2], level_of_measurement='ordinal')
            if not np.isnan(a):
                alphas.append(a)
        except:
            continue
    if len(alphas) < 100:
        return np.nan, np.nan, np.nan
    point = np.median(alphas)
    lower = np.percentile(alphas, (1 - ci) / 2 * 100)
    upper = np.percentile(alphas, (1 + ci) / 2 * 100)
    return point, lower, upper


def three_rater_alpha(rater_k, rater_l, rater_llm):
    """Compute three-rater Krippendorff alpha."""
    try:
        return kripp_alpha([rater_k, rater_l, rater_llm], level_of_measurement='ordinal')
    except:
        return np.nan


def main():
    np.random.seed(42)
    df_human = load_human_data()
    n_items = 100
    df_h = df_human.head(n_items)
    uuids = df_h["question_uuid"].tolist()

    # Discover all experiment cells
    cells = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and (d / "raw_scores.json").exists():
            parts = d.name.split("_", 1)
            if len(parts) == 2:
                # Handle model names with underscores
                for model in ["gemini_lite", "gemini_flash", "gpt54", "llama70b", "llama4scout"]:
                    if d.name.startswith(model + "_"):
                        prompt = d.name[len(model) + 1:]
                        cells.append((model, prompt, d / "raw_scores.json"))
                        break

    print(f"Found {len(cells)} experiment cells")
    print(f"Computing bootstrap CIs (2000 resamples each)...\n")

    results = []
    for model, prompt, raw_path in cells:
        medians = get_llm_median_scores(raw_path, n_items)
        for dim in DIMS:
            consensus_col = f"{dim}_consensus"
            k_col, l_col = HUMAN_DIM_MAP[dim]

            llm_scores = []
            human_scores = []
            k_scores = []
            l_scores = []

            for _, row in df_h.iterrows():
                uuid = row["question_uuid"]
                if uuid in medians and dim in medians[uuid]:
                    llm_val = medians[uuid][dim]
                    human_val = row[consensus_col]
                    if not np.isnan(llm_val) and not np.isnan(human_val):
                        llm_scores.append(llm_val)
                        human_scores.append(human_val)
                        k_scores.append(row[k_col])
                        l_scores.append(row[l_col])

            llm_arr = np.array(llm_scores)
            human_arr = np.array(human_scores)
            k_arr = np.array(k_scores)
            l_arr = np.array(l_scores)

            # Point estimate
            try:
                alpha_point = kripp_alpha([llm_arr, human_arr], level_of_measurement='ordinal')
            except:
                alpha_point = np.nan

            # Bootstrap CI
            _, ci_lo, ci_hi = bootstrap_alpha(llm_arr, human_arr, n_boot=2000)

            # Three-rater alpha
            alpha_3r = three_rater_alpha(k_arr, l_arr, llm_arr)

            # Human baseline
            try:
                alpha_kl = kripp_alpha([k_arr, l_arr], level_of_measurement='ordinal')
            except:
                alpha_kl = np.nan

            bias = float(np.mean(llm_arr - human_arr))

            tier = "A" if alpha_point >= 0.50 else ("B" if alpha_point >= 0.30 else "C")
            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"

            results.append({
                "model": model, "prompt": prompt, "dimension": dim,
                "alpha": round(alpha_point, 4),
                "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
                "alpha_3rater": round(alpha_3r, 4),
                "alpha_kl": round(alpha_kl, 4),
                "bias": round(bias, 3),
                "tier": tier,
                "n_items": len(llm_scores),
            })

            flag = " ***" if tier == "A" else ""
            print(f"  {model:15s} {prompt:15s} {dim:25s}  alpha={alpha_point:+.4f} {ci_str:20s}  3r={alpha_3r:+.4f}  bias={bias:+.3f}  {tier}{flag}")

    # Save
    df_results = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "results_with_cis.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary of Tier A dimensions with CIs
    print("\n" + "=" * 80)
    print("TIER A VALIDATION (with 95% CIs)")
    print("=" * 80)
    tier_a = df_results[df_results["tier"] == "A"].sort_values("alpha", ascending=False)
    for _, row in tier_a.iterrows():
        ci_contains_05 = row["ci_lower"] < 0.50
        warning = " [WARN] CI crosses 0.50" if ci_contains_05 else " [OK] CI above 0.50"
        print(f"  {row['model']:15s} {row['prompt']:15s} {row['dimension']:25s}  "
              f"α={row['alpha']:.4f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]  "
              f"3-rater={row['alpha_3rater']:.4f}{warning}")

    # Summary of three-rater validation
    print("\n" + "=" * 80)
    print("THREE-RATER ALPHA VALIDATION (K, L, LLM)")
    print("=" * 80)
    for dim in DIMS:
        dim_data = df_results[df_results["dimension"] == dim]
        best = dim_data.loc[dim_data["alpha"].idxmax()] if len(dim_data) > 0 else None
        if best is not None:
            kl = best["alpha_kl"]
            a3r = best["alpha_3rater"]
            delta = a3r - kl
            status = "[OK] LLM adds value" if delta >= -0.05 else "[FAIL] LLM degrades agreement"
            print(f"  {dim:25s}  alpha(K,L)={kl:.4f}  alpha(K,L,LLM)={a3r:.4f}  delta={delta:+.4f}  {status}")


if __name__ == "__main__":
    main()
