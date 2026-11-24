"""
Calibration Monitoring & Drift Detection Module.
Tracks LLM confidence calibration and detects performance drift over time.
"""

from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    ANNOTATION_DIMENSIONS,
    RESULTS_DIR,
    setup_logging,
    save_results,
)
from src.metadata import get_metadata_logger

logger = setup_logging()


# ---------------------------------------------------------------------------
# Confidence Calibration
# ---------------------------------------------------------------------------

def compute_calibration_curve(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data for reliability diagrams.

    Returns bin-level accuracy vs confidence data.
    """
    correct = (true_labels == predicted_labels).astype(float)
    conf_normalized = confidences / 100.0 if confidences.max() > 1 else confidences

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf_normalized >= lower) & (conf_normalized <= upper)
        else:
            mask = (conf_normalized >= lower) & (conf_normalized < upper)

        if mask.sum() > 0:
            bin_centers.append((lower + upper) / 2)
            bin_accuracies.append(float(correct[mask].mean()))
            bin_confidences.append(float(conf_normalized[mask].mean()))
            bin_counts.append(int(mask.sum()))

    ece = sum(
        (count / len(true_labels)) * abs(acc - conf)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
    )

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": round(float(ece), 4),
        "n_total": len(true_labels),
    }


def plot_reliability_diagram(
    calibration_data: dict,
    dimension: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a reliability diagram (calibration plot).
    Shows actual accuracy vs predicted confidence.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    centers = calibration_data["bin_centers"]
    accuracies = calibration_data["bin_accuracies"]
    confidences = calibration_data["bin_confidences"]
    counts = calibration_data["bin_counts"]

    # Main calibration plot
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax1.bar(centers, accuracies, width=0.08, alpha=0.7, label="Model", color="steelblue", edgecolor="navy")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Fraction Correct", fontsize=12)
    ax1.set_title(
        f"Reliability Diagram - {dimension}\nECE = {calibration_data['ece']:.4f}",
        fontsize=14,
    )
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.bar(centers, counts, width=0.08, color="steelblue", alpha=0.7, edgecolor="navy")
    ax2.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"reliability_diagram_{dimension}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram: {save_path}")


# ---------------------------------------------------------------------------
# Drift Detection
# ---------------------------------------------------------------------------

def detect_drift(
    baseline_results: dict,
    current_results: dict,
    threshold: float = 0.1,
) -> dict:
    """
    Detect performance drift by comparing current run against baseline.

    Flags alert if any dimension's Kappa drops by more than `threshold`.
    """
    alerts = []
    status = "OK"

    for dim in ANNOTATION_DIMENSIONS:
        if dim not in baseline_results.get("dimensions", {}):
            continue
        if dim not in current_results.get("dimensions", {}):
            continue

        baseline_kappa = baseline_results["dimensions"][dim]["cohens_kappa"]["kappa"]
        current_kappa = current_results["dimensions"][dim]["cohens_kappa"]["kappa"]

        if np.isnan(baseline_kappa) or np.isnan(current_kappa):
            continue

        drift = baseline_kappa - current_kappa

        if drift > threshold:
            alerts.append({
                "dimension": dim,
                "baseline_kappa": baseline_kappa,
                "current_kappa": current_kappa,
                "drift": round(drift, 4),
                "severity": "HIGH" if drift > 2 * threshold else "MEDIUM",
            })
            status = "DRIFT_DETECTED"

    return {
        "status": status,
        "alerts": alerts,
        "threshold": threshold,
        "num_dimensions_checked": len(
            [d for d in ANNOTATION_DIMENSIONS if d in baseline_results.get("dimensions", {})]
        ),
    }


# ---------------------------------------------------------------------------
# Self-Consistency Analysis
# ---------------------------------------------------------------------------

def analyze_self_consistency(df_llm: pd.DataFrame) -> dict:
    """
    Analyze LLM self-consistency from multi-run data.

    Expects df_llm to have columns like '{dim}_consistency' and '{dim}_all_runs'.
    """
    results = {}

    for dim in ANNOTATION_DIMENSIONS:
        cons_col = f"{dim}_consistency"
        runs_col = f"{dim}_all_runs"

        if cons_col not in df_llm.columns:
            continue

        consistencies = df_llm[cons_col].dropna()

        results[dim] = {
            "mean_consistency": round(float(consistencies.mean()), 4),
            "std_consistency": round(float(consistencies.std()), 4),
            "perfect_consistency_pct": round(
                float((consistencies == 1.0).mean() * 100), 2
            ),
            "n": len(consistencies),
        }

    # Overall
    all_cons = []
    for dim in ANNOTATION_DIMENSIONS:
        cons_col = f"{dim}_consistency"
        if cons_col in df_llm.columns:
            all_cons.extend(df_llm[cons_col].dropna().tolist())

    results["overall"] = {
        "mean_consistency": round(float(np.mean(all_cons)), 4) if all_cons else np.nan,
        "std_consistency": round(float(np.std(all_cons)), 4) if all_cons else np.nan,
        "perfect_consistency_pct": round(
            float(sum(1 for c in all_cons if c == 1.0) / len(all_cons) * 100), 2
        ) if all_cons else np.nan,
    }

    # Log self-consistency metadata
    meta = get_metadata_logger()
    meta.log_provenance("self_consistency", {
        "overall_mean": results["overall"]["mean_consistency"],
        "overall_std": results["overall"]["std_consistency"],
        "perfect_pct": results["overall"]["perfect_consistency_pct"],
        "per_dimension": {
            dim: results[dim]["mean_consistency"]
            for dim in ANNOTATION_DIMENSIONS if dim in results
        },
    })

    return results


# ---------------------------------------------------------------------------
# Summary Plots
# ---------------------------------------------------------------------------

def plot_dimension_comparison(
    results_list: list[dict],
    metric: str = "kappa",
    save_path: Optional[Path] = None,
) -> None:
    """
    Bar chart comparing metric across conditions and dimensions.
    """
    data = []
    for result in results_list:
        condition = result["condition"]
        for dim, metrics in result.get("dimensions", {}).items():
            if metric == "kappa":
                val = metrics["cohens_kappa"]["kappa"]
            elif metric == "exact_match":
                val = metrics["exact_match"]["exact_match_pct"]
            elif metric == "alpha":
                val = metrics["krippendorffs_alpha"]["alpha"]
            else:
                continue
            data.append({"Condition": condition, "Dimension": dim, "Value": val})

    if not data:
        logger.warning("No data to plot.")
        return

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Dimension", y="Value", hue="Condition", ax=ax)
    ax.set_title(f"Comparison: {metric.replace('_', ' ').title()} by Dimension", fontsize=14)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_xlabel("Annotation Dimension", fontsize=12)
    ax.legend(title="Condition")
    ax.grid(True, alpha=0.3, axis="y")

    # Add threshold lines for kappa
    if metric == "kappa":
        ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.7, label="Moderate (0.6)")
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.7, label="Substantial (0.8)")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"comparison_{metric}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison plot: {save_path}")
