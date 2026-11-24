"""Generate dashboard/src/data/findingsData.js from the real experiment CSVs.

The Findings tab must never depend on the API server. If the backend fails during a
demo, this data is already compiled into the bundle and the tab still renders.

Run from the repo root:
    python scripts/build_findings_data.py

Every number in the Findings tab traces back through this script to a CSV under
Data/results/neurips_experiments/. Nothing is typed by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Data" / "results" / "neurips_experiments"
OUT = ROOT / "dashboard" / "src" / "data" / "findingsData.js"

DIM_ORDER = [
    "Linguistic_Feature",
    "Relevance",
    "Semantic_Plausibility",
    "Semantic_Uniqueness",
    "Clarity",
]


def read(name: str) -> pd.DataFrame:
    path = EXP / name
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path)


def best_per_dimension(cis: pd.DataFrame) -> list[dict]:
    rows = []
    for dim in DIM_ORDER:
        g = cis[cis.dimension == dim]
        if g.empty:
            continue
        b = g.loc[g.alpha.idxmax()]
        rows.append(
            {
                "dimension": dim,
                "alpha": round(float(b.alpha), 3),
                "ciLower": round(float(b.ci_lower), 3),
                "ciUpper": round(float(b.ci_upper), 3),
                "model": str(b.model),
                "prompt": str(b.prompt),
                "n": int(b.n_items),
                "tier": str(b.tier),
                # Honest flags. The UI renders these, it does not recompute them.
                "excludesZero": bool(b.ci_lower > 0),
                "crossesHalf": bool(b.ci_lower <= 0.5 <= b.ci_upper),
            }
        )
    return rows


def prompt_matrix(cis: pd.DataFrame, model: str) -> dict:
    g = cis[cis.model == model]
    piv = g.pivot_table(index="dimension", columns="prompt", values="alpha")
    piv = piv.reindex([d for d in DIM_ORDER if d in piv.index])
    return {
        "model": model,
        "prompts": list(piv.columns),
        "rows": [
            {"dimension": dim, "values": [round(float(v), 3) for v in piv.loc[dim]]}
            for dim in piv.index
        ],
    }


def routing_paradox(routing: pd.DataFrame, cis: pd.DataFrame) -> list[dict]:
    """Auto-accept rate against actual agreement, per dimension.

    This is the evidence that a confidence gate alone is unsafe: Clarity reaches
    near-total run consensus while its agreement with humans is below chance.
    """
    r = routing[(routing.model == "gemini_flash") & (routing.prompt == "P0_baseline")]
    c = cis[(cis.model == "gemini_flash") & (cis.prompt == "P0_baseline")]
    merged = r.merge(c[["dimension", "alpha", "tier"]], on="dimension", how="left")
    out = []
    for dim in DIM_ORDER:
        row = merged[merged.dimension == dim]
        if row.empty:
            continue
        row = row.iloc[0]
        out.append(
            {
                "dimension": dim,
                "pct3of3": round(float(row.pct_3of3), 1),
                "pctAutoAccept": round(float(row.pct_auto_accept), 1),
                "alpha": round(float(row.alpha), 3),
                "tier": str(row.tier),
                "unsafe": bool(row.pct_auto_accept >= 90 and row.alpha < 0.3),
            }
        )
    return out


def active_learning() -> list[dict]:
    curves = []
    for dim in ("Relevance", "Linguistic_Feature"):
        path = EXP / f"active_learning_{dim}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        curves.append(
            {
                "dimension": dim,
                "points": [
                    {
                        "budgetPct": float(r.budget_pct),
                        "active": round(float(r.alpha_active), 3),
                        "random": round(float(r.alpha_random), 3),
                        "gain": round(float(r.alpha_gain), 3),
                    }
                    for r in df.itertuples()
                ],
            }
        )
    return curves


def fairness(fair: pd.DataFrame) -> list[dict]:
    out = []
    for (dim, analysis), g in fair.groupby(["dimension", "analysis"], sort=False):
        if dim not in DIM_ORDER:
            continue
        out.append(
            {
                "dimension": dim,
                "analysis": analysis,
                "maxGap": round(float(g.max_gap.iloc[0]), 3),
                "flagged": not bool(g.fair.iloc[0]),
                "subgroups": [
                    {
                        "name": str(r.subgroup),
                        "alpha": round(float(r.alpha), 3),
                        "n": int(r.n),
                    }
                    for r in g.itertuples()
                ],
            }
        )
    return out


def human_ceiling() -> list[dict]:
    """Human-human agreement, produced by scripts/compute_human_baseline.py.

    Both annotators rated all 200 items. The project long assumed otherwise, so this
    is the denominator every agreement number should have been reported against.
    """
    path = ROOT / "Data" / "results" / "human_baseline.csv"
    comp = ROOT / "Data" / "results" / "llm_vs_ceiling.csv"
    if not path.exists() or not comp.exists():
        return []
    base = pd.read_csv(path)
    llm = pd.read_csv(comp)
    rows = []
    for dim in DIM_ORDER:
        b = base[base.dimension == dim]
        g = llm[llm.dimension == dim]
        if b.empty or g.empty:
            continue
        b = b.iloc[0]
        # Best config by matched-item agreement. Never rank by the consensus-restricted
        # number: it is computed on a filtered subset and is not comparable across
        # dimensions or against a full-set ceiling.
        best = g.loc[g.alpha_vs_mean.idxmax()]
        rows.append(
            {
                "dimension": dim,
                "humanAlpha": round(float(b.alpha_ordinal), 3),
                "humanCiLower": round(float(b.ci_lower), 3),
                "humanCiUpper": round(float(b.ci_upper), 3),
                "humanExactPct": float(b.exact_match_pct),
                "nTotal": int(b.n_both_rated),
                "ceilingMatched": round(float(best.human_ceiling_matched), 3),
                "llmAlpha": round(float(best.alpha_vs_mean), 3),
                "pctOfCeiling": (
                    round(float(best.pct_of_ceiling), 0) if pd.notna(best.pct_of_ceiling) else None
                ),
                "humanModalPct": float(best.human_modal_pct),
                "llmModalPct": float(best.llm_modal_pct),
                "llmCollapsed": bool(best.llm_collapsed),
                "bestConfig": f"{best.model}/{best.prompt}",
            }
        )
    return rows


def main() -> None:
    cis = read("results_with_cis.csv")
    routing = read("item_level_routing.csv")
    thresh = read("threshold_sensitivity.csv")
    power = read("power_analysis.csv")
    fair = read("fairness_audit.csv")
    summary = json.loads((EXP / "experiment_summary.json").read_text(encoding="utf-8"))

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "sourceDir": "Data/results/neurips_experiments/",
            "generator": "scripts/build_findings_data.py",
            "experimentRun": summary.get("timestamp"),
            "models": summary.get("models", {}),
            "prompts": summary.get("prompts", {}),
            "promptHashes": summary.get("prompt_hashes", {}),
            "dataChecksums": summary.get("data_checksums", {}),
            "config": summary.get("config", {}),
        },
        "scale": {
            "nConfigs": int(len(cis)),
            "nModels": int(cis.model.nunique()),
            "nPrompts": int(cis.prompt.nunique()),
            "nDimensions": int(cis.dimension.nunique()),
            "nNegative": int((cis.alpha < 0).sum()),
        },
        "humanCeiling": human_ceiling(),
        "bestPerDimension": best_per_dimension(cis),
        "promptMatrix": prompt_matrix(cis, "gemini_flash"),
        "routingParadox": routing_paradox(routing, cis),
        "thresholdSensitivity": [
            {
                "threshold": float(r.threshold),
                "nTierA": int(r.n_tier_a),
                "nTotal": int(r.n_total),
                "workloadPct": float(r.workload_reduction_pct),
                "meanAlpha": round(float(r.mean_tier_a_alpha), 3),
            }
            for r in thresh.itertuples()
        ],
        "power": [
            {
                "dimension": str(r.dimension),
                "bestAlpha": round(float(r.best_alpha), 3),
                "ciWidth100": round(float(r.ci_width_at_100), 3),
                "ciWidth300": round(float(r.ci_width_at_300), 3),
                "requiredN": str(r.required_n_for_target),
            }
            for r in power.itertuples()
        ],
        "activeLearning": active_learning(),
        "fairness": fairness(fair),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    OUT.write_text(
        "// GENERATED FILE. Do not edit by hand.\n"
        "// Regenerate: python scripts/build_findings_data.py\n"
        "// Every number here traces to a CSV in Data/results/neurips_experiments/.\n\n"
        f"export const findings = {body};\n\nexport default findings;\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  configs={payload['scale']['nConfigs']} "
          f"models={payload['scale']['nModels']} "
          f"negative_alpha={payload['scale']['nNegative']}")


if __name__ == "__main__":
    main()
