"""
FastAPI backend for the React annotation dashboard.

Replaces the Streamlit console: the React app (dashboard/) is the only UI;
this server exposes the Python pipeline over HTTP.

    Upload CSV  →  POST /api/upload          (or /api/upload-sample)
    Annotate    →  POST /api/annotate        (replay cached runs or live API)
    Queue       →  GET  /api/queue
    Review      →  POST /api/review
    Finalize    →  POST /api/finalize        (hybrid or full_auto)
    Export      →  GET  /api/export.csv

Barrier-breaking frameworks:
    Active learning →  GET  /api/active-learning?dimension=...
    Fairness audit  →  GET  /api/fairness
    Pairwise Elo    →  POST /api/elo/run

Run:  uvicorn src.api.server:app --port 8000
"""

import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.input_handler import InputHandler
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.pipeline.confidence_gate import AGREE_3_OF_3, AGREE_2_OF_3
from src.pipeline.dimension_router import (
    TIER_A_DIMENSIONS,
    TIER_C_DIMENSIONS,
    ALL_PIPELINE_DIMENSIONS,
)
from src.pipeline.prompt_payloads import build_strict_prompt
from src.pipeline.active_selector import (
    rank_items_by_uncertainty,
    simulate_active_selection,
    compute_optimal_budget,
)
from src.pipeline.pairwise_elo import EloTournament
from src.utils import setup_logging

logger = setup_logging()

DATA_DIR = _ROOT / "Data"
SAMPLE_CSV = DATA_DIR / "processed" / "pipeline_upload_200items.csv"
HUMAN_CSV = DATA_DIR / "processed" / "full_scores_both_annotators.csv"
EXPERIMENTS_DIR = DATA_DIR / "results" / "neurips_experiments"
REPLAY_SCORES = EXPERIMENTS_DIR / "gemini_flash_P0_baseline" / "raw_scores.json"

app = FastAPI(title="LLM as Judge (Annotating) API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory pipeline session (single-user research tool)
# ---------------------------------------------------------------------------
STATE: dict = {
    "items": [],          # uploaded items (list of dicts)
    "pipeline": None,     # output of orchestrator.run()
    "reviews": [],        # human reviews
    "finalized": None,    # output of orchestrator.finalize()
    "annotate_mode": None,
}
ORCH = HybridOrchestrator()


def _df_to_items(df: pd.DataFrame) -> list[dict]:
    items = []
    for _, row in df.iterrows():
        items.append({
            "question_uuid": str(row["question_uuid"]),
            "text": str(row.get("text", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "textID": str(row.get("textID", "")),
        })
    return items


def _load_replay_scores() -> dict:
    if REPLAY_SCORES.exists():
        with open(REPLAY_SCORES, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _runs_for_item(uuid: str, replay: dict, rng: random.Random) -> dict:
    """Real cached 3-run scores when available, seeded synthetic otherwise."""
    if uuid in replay:
        return {d: list(v) for d, v in replay[uuid].items()}
    runs = {}
    for dim in ALL_PIPELINE_DIMENSIONS:
        base = rng.randint(1, 4)
        roll = rng.random()
        if roll < 0.55:
            runs[dim] = [base] * 3
        elif roll < 0.85:
            other = base % 4 + 1
            runs[dim] = rng.sample([base, base, other], 3)
        else:
            runs[dim] = rng.sample([1, 2, 3, 4], 3)[:3]
    return runs


def _scores_by_uuid() -> dict:
    """uuid -> {dim: [r1, r2, r3]} from the current pipeline state."""
    out = {}
    for item in (STATE["pipeline"] or {}).get("gated_items", []):
        merged = {}
        merged.update(item.get("tier_a", {}))
        merged.update(item.get("tier_b", {}))
        merged.update(item.get("tier_c", {}))
        out[item["question_uuid"]] = merged
    return out


def _json_safe(obj):
    """Recursively convert numpy types / NaN for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Core pipeline endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    import os
    return {
        "ok": True,
        "live_key_present": bool(os.environ.get("GEMINI_API_KEY")),
        "replay_available": REPLAY_SCORES.exists(),
        "sample_available": SAMPLE_CSV.exists(),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    tmp = _ROOT / "Data" / "results" / "_api_upload.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    return _ingest(tmp, source=file.filename)


@app.post("/api/upload-sample")
def upload_sample():
    if not SAMPLE_CSV.exists():
        raise HTTPException(404, f"Sample CSV not found: {SAMPLE_CSV}")
    return _ingest(SAMPLE_CSV, source=SAMPLE_CSV.name)


def _ingest(path: Path, source: str) -> dict:
    try:
        df = InputHandler().load_csv(Path(path))
    except ValueError as e:
        raise HTTPException(422, str(e))
    STATE["items"] = _df_to_items(df)
    STATE["pipeline"] = None
    STATE["reviews"] = []
    STATE["finalized"] = None
    preview = [
        {k: (v[:160] + "…" if isinstance(v, str) and len(v) > 160 else v)
         for k, v in item.items()}
        for item in STATE["items"][:5]
    ]
    return {
        "source": source,
        "n_items": len(STATE["items"]),
        "preview": preview,
        "tier_a_dimensions": TIER_A_DIMENSIONS,
        "tier_c_dimensions": TIER_C_DIMENSIONS,
    }


@app.post("/api/annotate")
def annotate(body: dict):
    """body: {mode: 'replay' | 'live'}"""
    if not STATE["items"]:
        raise HTTPException(409, "No batch uploaded yet")
    mode = body.get("mode", "replay")
    items = STATE["items"]

    if mode == "live":
        from src.pipeline.annotation_engine import AnnotationEngine
        try:
            ORCH.engine = AnnotationEngine()
        except Exception as e:
            raise HTTPException(424, f"Live mode unavailable (API key?): {e}")
        state = ORCH.run(items, build_strict_prompt(), condition_name="react_live")
    else:
        replay = _load_replay_scores()
        rng = random.Random(42)
        raw_results = []
        for it in items:
            runs = _runs_for_item(it["question_uuid"], replay, rng)
            raw_results.append({
                "question_uuid": it["question_uuid"],
                "runs": [
                    {"scores": {d: v[i] for d, v in runs.items()}}
                    for i in range(3)
                ],
            })
        state = ORCH.run(
            items, build_strict_prompt(),
            condition_name="react_replay",
            precomputed_results=raw_results,
        )

    STATE["pipeline"] = state
    STATE["annotate_mode"] = mode
    STATE["finalized"] = None
    n_replayed = len([1 for it in items if it["question_uuid"] in _load_replay_scores()]) if mode == "replay" else 0
    return _json_safe({
        "mode": mode,
        "n_items": len(items),
        "n_replayed_real": n_replayed,
        "gate_stats": state["gate_stats"],
    })


@app.get("/api/state")
def get_state():
    pipeline = STATE["pipeline"]
    reviewed = {(r["question_uuid"], r["dimension"]) for r in STATE["reviews"]}
    return _json_safe({
        "n_items": len(STATE["items"]),
        "annotated": pipeline is not None,
        "annotate_mode": STATE["annotate_mode"],
        "gate_stats": pipeline["gate_stats"] if pipeline else None,
        "n_reviews": len(STATE["reviews"]),
        "n_reviewed_decisions": len(reviewed),
        "finalized": STATE["finalized"] is not None,
        "workload_report": (STATE["finalized"] or {}).get("workload_report"),
    })


@app.get("/api/queue")
def queue():
    if not STATE["pipeline"]:
        raise HTTPException(409, "Run annotation first")
    lookup = {it["question_uuid"]: it for it in STATE["items"]}
    qdf = ORCH.export_human_queue(STATE["pipeline"], items_lookup=lookup, save=False)
    reviewed = {(r["question_uuid"], r["dimension"]) for r in STATE["reviews"]}
    rows = []
    for _, r in qdf.iterrows():
        rows.append({
            "question_uuid": r["question_uuid"],
            "dimension": r["dimension"],
            "tier": r["tier"],
            "queue_reason": r["queue_reason"],
            "llm_runs_visible": r["llm_runs_visible"],
            "priority": int(r["priority"]),
            "question": r["question"],
            "answer": r["answer"],
            "reviewed": (r["question_uuid"], r["dimension"]) in reviewed,
        })
    return _json_safe({"rows": rows, "n_pending": len([r for r in rows if not r["reviewed"]])})


@app.post("/api/review")
def review(body: dict):
    for field in ("question_uuid", "dimension", "score"):
        if field not in body:
            raise HTTPException(422, f"Missing field: {field}")
    score = int(body["score"])
    if not 1 <= score <= 4:
        raise HTTPException(422, "Score must be 1-4")
    # replace any previous review of the same decision
    STATE["reviews"] = [
        r for r in STATE["reviews"]
        if not (r["question_uuid"] == body["question_uuid"] and r["dimension"] == body["dimension"])
    ]
    STATE["reviews"].append({
        "question_uuid": str(body["question_uuid"]),
        "dimension": str(body["dimension"]),
        "score": score,
        "annotator_id": str(body.get("annotator_id", "reviewer_1")),
    })
    return {"ok": True, "n_reviews": len(STATE["reviews"])}


@app.post("/api/finalize")
def finalize(body: dict):
    if not STATE["pipeline"]:
        raise HTTPException(409, "Run annotation first")
    full_auto = bool(body.get("full_auto", False))
    result = ORCH.finalize(STATE["pipeline"], human_reviews=STATE["reviews"], full_auto=full_auto)
    STATE["finalized"] = result
    df = result["assembled"]
    by_source = df["source"].value_counts().to_dict()
    return _json_safe({
        "full_auto": full_auto,
        "workload_report": result["workload_report"],
        "by_source": by_source,
        "n_rows": len(df),
    })


@app.get("/api/export.csv")
def export_csv():
    if not STATE["finalized"]:
        raise HTTPException(409, "Finalize first")
    df = STATE["finalized"]["assembled"].copy()
    df["llm_runs"] = df["llm_runs"].apply(json.dumps)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotations_final.csv"},
    )


# ---------------------------------------------------------------------------
# Barrier-breaking framework endpoints
# ---------------------------------------------------------------------------

@app.get("/api/active-learning")
def active_learning(dimension: str = "Linguistic_Feature"):
    """Budget curve (precomputed from the real experiment when available)
    plus a live uncertainty ranking of the current batch."""
    if dimension not in ALL_PIPELINE_DIMENSIONS:
        raise HTTPException(422, f"Unknown dimension: {dimension}")

    curve_path = EXPERIMENTS_DIR / f"active_learning_{dimension}.csv"
    curve, curve_source = [], None
    if curve_path.exists():
        curve = pd.read_csv(curve_path).to_dict("records")
        curve_source = "real_experiment"
    else:
        scores = _scores_by_uuid() or _replay_scores_dict()
        consensus = _human_consensus(dimension)
        if scores and consensus:
            sim = simulate_active_selection(scores, consensus, dimension)
            curve = sim.to_dict("records")
            curve_source = "computed_now"

    optimal = None
    if curve:
        optimal = compute_optimal_budget(pd.DataFrame(curve))

    ranked_rows = []
    scores_now = _scores_by_uuid() or _replay_scores_dict()
    if scores_now:
        ranked = rank_items_by_uncertainty(scores_now, dimension).head(12)
        ranked_rows = ranked.to_dict("records")

    return _json_safe({
        "dimension": dimension,
        "curve": curve,
        "curve_source": curve_source,
        "optimal_budget": optimal,
        "most_uncertain": ranked_rows,
    })


def _replay_scores_dict() -> dict:
    replay = _load_replay_scores()
    return {u: {d: list(v) for d, v in dims.items()} for u, dims in replay.items()}


def _human_consensus(dimension: str) -> dict:
    if not HUMAN_CSV.exists():
        return {}
    df = pd.read_csv(HUMAN_CSV)
    hd = "Semantic_Unique" if dimension == "Semantic_Uniqueness" else dimension
    k_col, l_col = f"{hd}_KC", f"{hd}_LH"
    if k_col not in df.columns or l_col not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        k, l = row.get(k_col), row.get(l_col)
        if pd.notna(k) and pd.notna(l):
            out[str(row["question_uuid"])] = round((float(k) + float(l)) / 2)
    return out


@app.get("/api/fairness")
def fairness():
    """Real fairness audit results from the experiment run."""
    path = EXPERIMENTS_DIR / "fairness_audit.csv"
    if not path.exists():
        raise HTTPException(404, "fairness_audit.csv not found; run scripts/run_experiments.py first")
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    flags = [
        r for r in rows
        if (not r.get("fair", True)) or (r.get("max_gap") or 0) > 0.15
    ]
    return _json_safe({
        "rows": rows,
        "n_flags": len({(f['dimension'], f['analysis']) for f in flags}),
        "threshold": 0.15,
    })


@app.post("/api/elo/run")
def elo_run(body: dict):
    """Run a pairwise Elo tournament with a simulated judge.

    The judge prefers the item whose (replayed) LLM median is higher on the
    chosen dimension, with seeded noise, demonstrating the framework without
    API spend. A live judge plugs in via PairwiseJudge(engine).
    """
    dimension = body.get("dimension", "Semantic_Plausibility")
    budget = int(body.get("budget", 150))
    if dimension not in ALL_PIPELINE_DIMENSIONS:
        raise HTTPException(422, f"Unknown dimension: {dimension}")

    scores = _scores_by_uuid() or _replay_scores_dict()
    if not scores:
        raise HTTPException(409, "No scores available: upload a batch and annotate, or keep the replay data in place")

    items = [{"question_uuid": u} for u in scores.keys()]
    tournament = EloTournament(items, dimension=dimension, seed=42)
    pairs = tournament.schedule_pairs(budget=budget)

    rng = random.Random(7)
    def median_of(u):
        runs = [s for s in scores[u].get(dimension, []) if not (isinstance(s, float) and np.isnan(s))]
        return float(np.median(runs)) if runs else 2.5

    n_run = 0
    for a, b in pairs:
        ua, ub = a["question_uuid"], b["question_uuid"]
        qa = median_of(ua) + rng.gauss(0, 0.6)
        qb = median_of(ub) + rng.gauss(0, 0.6)
        outcome = "tie" if abs(qa - qb) < 0.15 else ("a" if qa > qb else "b")
        tournament.record(ua, ub, outcome)
        n_run += 1

    ordinal = tournament.to_ordinal_scores()
    hist = {s: 0 for s in (1, 2, 3, 4)}
    for v in ordinal.values():
        hist[int(v)] += 1
    leaderboard = [
        {"question_uuid": u, "elo": round(e, 1), "ordinal": ordinal.get(u)}
        for u, e in tournament.leaderboard(top=10)
    ]
    return _json_safe({
        "dimension": dimension,
        "n_items": len(items),
        "n_comparisons": n_run,
        "judge": "simulated (LLM-median + noise, seeded)",
        "leaderboard": leaderboard,
        "ordinal_histogram": hist,
    })


# ---------------------------------------------------------------------------
# Reference data for the React UI
# ---------------------------------------------------------------------------

@app.get("/api/reference")
def reference():
    """Tier assignments and headline alphas for UI display.

    Values are copied from Data/results/neurips_experiments/results_with_cis.csv
    (best setup per dimension) and Data/results/human_baseline.csv, which are not
    distributed. Update them when those files change.
    """
    return {
        "tier_a": TIER_A_DIMENSIONS,
        "tier_c": TIER_C_DIMENSIONS,
        "headline_alphas": {
            "Linguistic_Feature": 0.536,
            "Relevance": 0.505,
            "Semantic_Uniqueness": 0.259,
            "Semantic_Plausibility": 0.219,
            "Clarity": -0.088,
        },
        "human_alphas": {
            "Linguistic_Feature": 0.835,
            "Relevance": 0.741,
            "Semantic_Uniqueness": 0.855,
            "Semantic_Plausibility": 0.769,
            "Clarity": 0.873,
        },
    }
