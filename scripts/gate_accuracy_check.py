"""Quick check: Is the confidence gate's auto-accept rate backed by actual accuracy?"""
import sys, ast
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.dimension_router import DimensionRouter
from src.pipeline.confidence_gate import ConfidenceGate

df_human = pd.read_csv(ROOT / "Data/processed/full_scores_both_annotators.csv")
df_human["LF_K"] = df_human["Linguistic_Feature_KC"]
df_human["LF_L"] = df_human["Linguistic_Feature_LH"]
df_human["LF_consensus"] = ((df_human["LF_K"] + df_human["LF_L"]) / 2).round().astype(int)

DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility", "Semantic_Uniqueness"]

def load_and_gate(condition):
    csv_path = ROOT / "Data/results/phase1_question_level" / condition / f"{condition}_results.csv"
    df_llm = pd.read_csv(csv_path)
    engine_items = []
    for _, row in df_llm.iterrows():
        uuid = str(row["question_uuid"])
        runs = []
        for ri in range(3):
            scores = {}
            for dim in DIMS:
                col = f"{dim}_all_runs"
                if col in df_llm.columns:
                    try:
                        rl = ast.literal_eval(str(row[col]))
                        scores[dim] = rl[ri] if ri < len(rl) else np.nan
                    except:
                        scores[dim] = np.nan
                else:
                    scores[dim] = row.get(dim, np.nan)
            runs.append({"scores": scores})
        engine_items.append({"question_uuid": uuid, "runs": runs})

    router = DimensionRouter()
    gate = ConfidenceGate()
    routed = router.route_batch(engine_items)
    gated = gate.evaluate_batch(routed)

    rows = []
    for item in gated:
        dec = item["gate_decisions"].get("Linguistic_Feature", {})
        rows.append({
            "question_uuid": item["question_uuid"],
            "decision": dec.get("decision", "unknown"),
            "accepted_score": dec.get("accepted_score"),
            "run_scores": item["tier_a"].get("Linguistic_Feature", []),
        })
    return pd.DataFrame(rows)


print("=" * 80)
print("CONFIDENCE GATE: SELF-CONSISTENCY vs ACTUAL ACCURACY")
print("=" * 80)

for cond in ["zero_shot_qlevel", "rubric_guided_qlevel", "few_shot_qlevel"]:
    g = load_and_gate(cond)
    m = g.merge(df_human[["question_uuid", "LF_K", "LF_L", "LF_consensus"]], on="question_uuid")

    a3 = m[m["decision"] == "auto_accept"]
    a2 = m[m["decision"] == "auto_accept_flagged"]
    aa = m[m["decision"].isin(["auto_accept", "auto_accept_flagged"])]
    sp = m[m["decision"] == "route_to_human"]

    print(f"\n{'='*60}")
    print(f"  {cond}")
    print(f"  Total={len(m)} | 3/3={len(a3)} | 2/3={len(a2)} | Split={len(sp)}")
    print(f"  Auto-accept rate: {len(aa)/len(m)*100:.1f}%")
    print(f"{'='*60}")
    
    print(f"  {'Group':<18} {'N':>4} {'ExactCon':>9} {'ExactK':>7} {'ExactL':>7} {'W/in1':>6} {'Bias':>6}")
    print(f"  {'-'*58}")

    for label, sub in [("3/3 agree", a3), ("2/3 agree", a2), ("ALL auto", aa)]:
        if len(sub) == 0:
            continue
        ec = (sub["accepted_score"] == sub["LF_consensus"]).mean() * 100
        ek = (sub["accepted_score"] == sub["LF_K"]).mean() * 100
        el = (sub["accepted_score"] == sub["LF_L"]).mean() * 100
        w1 = (abs(sub["accepted_score"] - sub["LF_consensus"]) <= 1).mean() * 100
        bias = sub["accepted_score"].mean() - sub["LF_consensus"].mean()
        print(f"  {label:<18} {len(sub):>4} {ec:>8.1f}% {ek:>6.1f}% {el:>6.1f}% {w1:>5.1f}% {bias:>+5.2f}")

    h_ex = (m["LF_K"] == m["LF_L"]).mean() * 100
    h_w1 = (abs(m["LF_K"] - m["LF_L"]) <= 1).mean() * 100
    print(f"  {'K vs L (human)':<18} {len(m):>4} {h_ex:>8.1f}% {'':>7} {'':>7} {h_w1:>5.1f}%")

    # Score distributions
    if len(aa) > 0:
        llm_d = aa["accepted_score"].value_counts().sort_index()
        hum_d = aa["LF_consensus"].value_counts().sort_index()
        print(f"\n  Score distributions (auto-accepted items only):")
        print(f"  {'Score':<8} {'LLM':>6} {'Human':>6}")
        for s in sorted(set(llm_d.index) | set(hum_d.index)):
            print(f"  {s:<8} {llm_d.get(s,0):>6} {hum_d.get(s,0):>6}")
        print(f"  Mean LLM: {aa['accepted_score'].mean():.2f}  Mean Human: {aa['LF_consensus'].mean():.2f}  Bias: {aa['accepted_score'].mean()-aa['LF_consensus'].mean():+.2f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
1. AUTO-ACCEPT RATE (87-94%) measures LLM SELF-CONSISTENCY, not accuracy.
   The 3 Gemini runs tend to agree with each other because:
   - Temperature 0.3 is fairly deterministic
   - Linguistic Feature is the most surface-level dimension
   - The LLM has a systematic positive bias (tends toward 3-4)

2. ACCURACY (exact match vs human) tells us if self-consistency is meaningful.
   Compare LLM exact match to human K-vs-L exact match as ceiling.

3. Phase 1 Alpha 0.58 was not "bad"; it was the best dimension.
   Alpha < 0.3 on Relevance, Clarity, etc.: those were bad.
   That's exactly why only Linguistic Feature is in Tier A.

4. The gate IS useful if 3/3-agree items have higher accuracy than split items.
   Split items are harder; even humans may disagree on them.
""")
