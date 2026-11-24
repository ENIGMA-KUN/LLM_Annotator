"""Compute the human-human agreement ceiling and LLM performance relative to it.

Both raters scored all 200 items on all five dimensions;
Data/processed/full_scores_both_annotators.csv holds those paired scores.

Outputs:
    Data/results/human_baseline.csv        human-human alpha per dimension, with CIs
    Data/results/llm_vs_ceiling.csv        LLM alpha against that ceiling, overall
                                           and restricted to human-consensus items

Run from the repo root:
    python scripts/compute_human_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Data" / "results" / "neurips_experiments"
OUT = ROOT / "Data" / "results"

# Rubric dimension -> column stem in full_scores_both_annotators.csv.
# Semantic_Uniqueness is stored as Semantic_Unique. Keep this map in one place.
DIM_COLUMN = {
    "Linguistic_Feature": "Linguistic_Feature",
    "Relevance": "Relevance",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
    "Clarity": "Clarity",
}

# The scale is ordinal 1-4. Nominal alpha treats 3-vs-4 as identically wrong to
# 1-vs-4 and materially understates agreement here. Ordinal is the correct default.
LEVEL = "ordinal"
N_BOOT = 2000


def alpha(x: np.ndarray, y: np.ndarray, level: str = LEVEL) -> float:
    """Pairwise Krippendorff alpha over aligned vectors, ignoring incomplete pairs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    try:
        return float(
            krippendorff.alpha(
                reliability_data=np.vstack([x[mask], y[mask]]),
                level_of_measurement=level,
            )
        )
    except Exception:
        return float("nan")


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """Percentile CI, resampling items. Items are the independent unit, not ratings."""
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = alpha(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def human_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, col in DIM_COLUMN.items():
        lh = df[f"{col}_LH"].to_numpy(dtype=float)
        kc = df[f"{col}_KC"].to_numpy(dtype=float)
        both = ~(np.isnan(lh) | np.isnan(kc))
        lo, hi = bootstrap_ci(lh, kc)
        rows.append(
            {
                "dimension": dim,
                "alpha_ordinal": round(alpha(lh, kc), 4),
                "alpha_nominal": round(alpha(lh, kc, "nominal"), 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "exact_match_pct": round(float((lh[both] == kc[both]).mean() * 100), 1),
                "n_both_rated": int(both.sum()),
                "n_consensus": int((lh[both] == kc[both]).sum()),
            }
        )
    return pd.DataFrame(rows)


def llm_vs_ceiling(df: pd.DataFrame, ceiling: dict[str, float]) -> pd.DataFrame:
    """LLM against each annotator, and restricted to items the humans agree on.

    The consensus-restricted column is the interesting one: where the two humans
    agree there is a defensible single answer, and that is the regime in which an
    automated label can be trusted. Where they disagree, human label variation is
    signal rather than noise and a single machine label is not the right output.
    """
    rows = []
    for cfg_dir in sorted(p for p in EXP.iterdir() if (p / "raw_scores.json").exists()):
        raw = json.loads((cfg_dir / "raw_scores.json").read_text(encoding="utf-8"))
        model, prompt = cfg_dir.name.rsplit("_", 2)[0], "_".join(cfg_dir.name.rsplit("_", 2)[1:])
        uuids = [u for u in raw if u in df.index]
        if not uuids:
            continue
        sub = df.loc[uuids]
        for dim, col in DIM_COLUMN.items():
            runs = [[v for v in raw[u].get(dim, []) if v is not None] for u in uuids]
            llm = np.array([np.median(r) if r else np.nan for r in runs], dtype=float)
            lh = sub[f"{col}_LH"].to_numpy(dtype=float)
            kc = sub[f"{col}_KC"].to_numpy(dtype=float)
            consensus = np.where(lh == kc, lh, np.nan)
            a_all = alpha(llm, (lh + kc) / 2)
            a_cons = alpha(llm, consensus)

            # Human ceiling recomputed ON THESE SAME ITEMS. Comparing an LLM alpha
            # from a filtered subset against a human alpha from the full 200 is not a
            # valid ratio: different marginals give different expected disagreement,
            # so the denominators are not the same quantity.
            ceil_matched = alpha(lh, kc)

            # Marginal concentration. Krippendorff alpha collapses when one rater puts
            # nearly all mass on a single category, regardless of how well the other
            # rater discriminates. This is the Feinstein-Cicchetti prevalence paradox
            # and it is the diagnostic that separates "no signal in the task" from
            # "the model stopped discriminating".
            h_all = np.concatenate([lh, kc])
            h_modal = float(pd.Series(h_all).value_counts(normalize=True).iloc[0])
            l_modal = float(pd.Series(llm).value_counts(normalize=True).iloc[0])

            rows.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "dimension": dim,
                    "n_items": len(uuids),
                    "alpha_vs_LH": round(alpha(llm, lh), 4),
                    "alpha_vs_KC": round(alpha(llm, kc), 4),
                    "alpha_vs_mean": round(a_all, 4),
                    "human_ceiling_matched": round(ceil_matched, 4),
                    # The defensible headline: same items, same marginals.
                    "pct_of_ceiling": (
                        round(a_all / ceil_matched * 100, 1)
                        if ceil_matched > 0 and np.isfinite(a_all)
                        else None
                    ),
                    # Conditional result. Report it as "performance given human
                    # agreement", never as a percentage of a full-set ceiling.
                    "alpha_on_consensus": round(a_cons, 4),
                    "n_consensus": int(np.isfinite(consensus).sum()),
                    "human_modal_pct": round(h_modal * 100, 1),
                    "llm_modal_pct": round(l_modal * 100, 1),
                    # Humans discriminate, model does not: a rubric failure, not an
                    # impossible construct.
                    "llm_collapsed": bool(l_modal >= 0.75 and h_modal < 0.75),
                    "human_ceiling_full": round(ceiling.get(dim, float("nan")), 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    src = ROOT / "Data" / "processed" / "full_scores_both_annotators.csv"
    df = pd.read_csv(src).set_index("question_uuid")
    print(f"Loaded {len(df)} items from {src.relative_to(ROOT)}")

    base = human_baseline(df)
    OUT.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT / "human_baseline.csv", index=False)
    print(f"\nHuman-human ceiling (n={base.n_both_rated.iloc[0]}, ordinal alpha):")
    for r in base.itertuples():
        print(
            f"  {r.dimension:24} {r.alpha_ordinal:+.3f}  "
            f"[{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]  "
            f"nominal {r.alpha_nominal:+.3f}  consensus on {r.n_consensus}/{r.n_both_rated}"
        )

    ceiling = dict(zip(base.dimension, base.alpha_ordinal))
    comp = llm_vs_ceiling(df, ceiling)
    comp.to_csv(OUT / "llm_vs_ceiling.csv", index=False)

    print("\nBest configuration per dimension, matched items, full set:")
    for dim in DIM_COLUMN:
        g = comp[comp.dimension == dim]
        if g.empty:
            continue
        b = g.loc[g.alpha_vs_mean.idxmax()]
        pct = f"{b.pct_of_ceiling:.0f}% of ceiling" if b.pct_of_ceiling else "n/a"
        print(
            f"  {dim:24} {b.alpha_vs_mean:+.3f} vs ceiling {b.human_ceiling_matched:+.3f}  "
            f"({pct})  {b.model}/{b.prompt}"
            + ("   [LLM COLLAPSED onto one category]" if b.llm_collapsed else "")
        )

    print(f"\nWrote {(OUT / 'human_baseline.csv').relative_to(ROOT)}")
    print(f"Wrote {(OUT / 'llm_vs_ceiling.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
