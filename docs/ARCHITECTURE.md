# Architecture

> Pipeline stages, routing logic and data flow. Read `RESEARCH_STATUS.md` first for what is
> actually working.

---

## Flow

```
intake CSV
    |
    v
InputHandler            validate schema, batch
    |
    v
AnnotationEngine        3 independent runs per item at temperature 0.3,
    |                   full metadata logged per call
    v
DimensionRouter         assign each dimension to a tier from its measured alpha
    |
    +--> Tier A --> ConfidenceGate --> 3/3 agree ---> auto-accept
    |                                  2/3 agree ---> accept, flagged
    |                                  split     ---> human
    |
    +--> Tier C ------------------------------------> human, always, blind
    |
    v
HybridOrchestrator      build the review queue, enforce blind review
    |
    v
Aggregator              merge machine and human labels, compute agreement
    |
    v
metrics / evaluation    ordinal Krippendorff alpha vs the measured human ceiling
```

---

## Stages

### InputHandler (`src/pipeline/input_handler.py`)

Validates the intake CSV against the expected schema and batches items. Failing rows are
reported with their ids and the rule broken; nothing is silently dropped or imputed.

### AnnotationEngine (`src/pipeline/annotation_engine.py`)

Three independent runs per item. Temperature 0.3, not 0, because eliminating variance removes the
uncertainty signal the router depends on.

Every call logs model, model version, provider, temperature, prompt template version, input
and output tokens, cost, UTC timestamp, latency, item id, run number, condition, raw
response, and parse status. A run completing without this metadata has to be repeated, so
missing fields are treated as failures rather than follow-ups.

Providers sit behind LiteLLM, so adding a model is a config entry. Currently registered:
Gemini Flash and Flash Lite, GPT, Groq (Llama 3.3 70B, Llama 4 Scout), and NVIDIA NIM
(Llama, Qwen, DeepSeek, Nemotron).

### DimensionRouter (`src/pipeline/dimension_router.py`)

Assigns each rubric dimension a tier from its measured agreement against the human ceiling.

| Tier | Rule | Behaviour |
|---|---|---|
| A | alpha >= 0.50 | eligible for auto-accept via the confidence gate |
| B | 0.30 <= alpha < 0.50 | requires 3/3 agreement, otherwise human |
| C | alpha < 0.30 | always human, blind. LLM score never shown |

Current assignment: Tier A holds Linguistic Feature and Relevance; Tier C holds Clarity,
Semantic Plausibility and Semantic Uniqueness. Tier B is presently empty, so that branch is
unreachable in practice.

Why the router exists at all. A confidence gate on its own would auto-accept over 90%
of items on four of five dimensions, and three of those carry no usable signal. On Clarity
the model reaches 99% self-consistency while scoring near zero against humans.
Self-consistency is not correctness. The router is the guard that makes the gate safe.

### ConfidenceGate (`src/pipeline/confidence_gate.py`)

Pure vote counting over the three runs. No numeric threshold.

- all three agree -> auto-accept
- two agree -> accept, flagged for audit
- all differ -> route to human

Note the gate yields only three distinguishable states, which is too coarse to draw a
useful risk-coverage curve. Using ordinal dispersion across runs rather than the exact
agreement pattern gives more levels, and is an open item.

### HybridOrchestrator (`src/pipeline/hybrid_orchestrator.py`)

Builds the human review queue and enforces blind review: for Tier C the LLM's runs are
withheld from the reviewer entirely. Anchoring is a measured effect, and this is where it
happens or does not. Priority: split items first, then Tier C, then flagged.

### Aggregator (`src/pipeline/aggregator.py`)

Merges tiers into a final label set and computes three-rater agreement. Note this module
does not implement Dawid-Skene; that lives separately in `src/aggregator.py` (via
crowd-kit) and is currently reachable only from `scripts/run_phase1.py`.

### Supporting modules

| Module | Role |
|---|---|
| `active_selector.py` | uncertainty-based selection of items for human review |
| `fairness_audit.py` | subgroup alpha by subject, difficulty, passage length, human agreement |
| `quality_monitor.py` | drift detection against a fixed gold set. Implemented but not wired to the API |
| `pairwise_elo.py` | Elo tournament. `EloTournament` is used; `PairwiseJudge` is not instantiated anywhere |
| `metrics.py` | Krippendorff alpha, defaults to ordinal |

---

## API

FastAPI, `src/api/server.py`, serving the dashboard. Routes cover upload, annotate, queue,
review, finalize, export, plus the analysis tabs.

Two integrity gaps, both open and documented in the README:

- `_runs_for_item()` generates synthetic 3-run scores from a seeded RNG for items outside
  the replay cache.
- `/api/elo/run` uses `median + gaussian noise` rather than a real judge.

Neither should be presented as a measured result.

---

## Known structural debt

- No config file. Tier thresholds, model ids, prices and run counts are Python literals
  spread across at least six modules. The best-config map is duplicated three times, and
  two conflicting price tables exist.
- `DimensionRouter.from_alphas()` derives tiers dynamically but has no caller, so the
  static list always wins.
- `dashboard/src/data/mockData.js` uses dimension names that do not match the pipeline
  (`Topic_Relevance`, `Writing_Clarity` against `Relevance`, `Clarity`).
- No tests for `annotation_engine`, `hybrid_orchestrator`, `active_selector`,
  `fairness_audit`, or the API.
