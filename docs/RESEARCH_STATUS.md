# Research status

> Living document. Updated 2026-08-04.
> Numbers reproduce with `python scripts/compute_human_baseline.py`.

This is the file to read first if you are joining the project or returning after a break.
It says what is settled, what is being worked on, what is blocked, and what was tried and
did not work.

---

## The one-paragraph version

We annotate MCQ items against a five-dimension expert rubric using an LLM, three runs per
item, and route anything the model cannot be trusted on to a human. Human-to-human
agreement is measured and high (alpha 0.74 to 0.88). The model reaches roughly two thirds
of that ceiling on two dimensions and fails on three. The failures are diagnosable: the
model collapses onto a single score where humans discriminate. Current work is fixing the
rubric so the model discriminates, not tuning the model.

---

## Settled

| Finding | Evidence | Where |
|---|---|---|
| Human ceiling is measurable and high | Both annotators rated all 200 items, all 5 dimensions. Ordinal alpha 0.741 to 0.873 with bootstrap CIs | `scripts/compute_human_baseline.py` |
| Two dimensions are usable as a first pass | Linguistic Feature 69% of ceiling, Relevance 62%, on identical items | `Data/results/llm_vs_ceiling.csv` |
| Model failure is distribution collapse | Clarity: model 95% on one score, humans 45%. Linguistic Feature: 37% vs 38%, and it works | same |
| Self-reported confidence is useless | Clusters 90-100, no discriminative value | Phase 1 |
| Inter-run agreement alone is unsafe as a gate | Would auto-accept over 90% of items on 4 of 5 dimensions, 3 of which carry no signal | `Data/results/neurips_experiments/item_level_routing.csv` |
| Automated prompt optimisation degrades results | Worse than zero-shot on 4 of 5 dimensions | `master_results.csv` |
| Targeted human review beats random | Relevance alpha 0.762 vs 0.659 at 30% budget | `active_learning_Relevance.csv` |
| The design is underpowered at n=100 | CI width about 0.32; roughly 300 items needed to confirm the two Tier A dimensions | `power_analysis.csv` |

---

## In progress

1. Prompt content verification. Blocking everything else.

The rubric mixes three units of analysis: Clarity is defined on the question stem,
Linguistic Feature and Semantic Plausibility on the distractor, Relevance on response
options. Semantic Uniqueness is absent from `rubric.json` entirely. The intake CSV is one
row per question with only the correct answer; the distractors are not in it.

If the model is asked to judge a distractor it cannot see, it has to invent one, and
inventing a different one per run is exactly what produces collapse.

Action: print one fully rendered prompt from `src/pipeline/prompt_payloads.py` and confirm
what it contains. Then confirm what the two raters were shown. If they differ, no
agreement number is interpretable.

2. Rubric rewrite. Blocked on (1). Plan and template exist. The design moves from
holistic 1-4 Likert descriptors to atomic binary checks with evidence spans, from which the
score is computed rather than emitted. CheckEval (EMNLP 2025) reports +0.45 mean agreement
for this change across 12 evaluator models.

3. Data re-export. One row per option instead of one per question, adding
`option_letter`, `option_text`, `is_correct`, `domain`. Takes 200 rows to 800 and makes the
distractor dimensions well-posed for the first time.

---

## Open, not started

| Item | Why it matters | Blocked by |
|---|---|---|
| Third annotator, 50-100 items | Unlocks the alt-test (Calderon et al., ACL 2025), the current standard for justifying LLM substitution. Two annotators cannot support it | scheduling |
| Risk-coverage curve and AURC | The deployable claim is "at X% coverage, alpha >= Y", not a single accuracy number | needs gold set |
| Contamination-stratified reanalysis | OpenStax is public and likely in pre-training. Reviewers will ask | nothing |
| Real Elo judge | `/api/elo/run` is `median + gaussian noise` | nothing |
| Remove synthetic replay path | `server.py` `_runs_for_item()` fabricates scores outside the cache | nothing |
| Config file | Thresholds, model ids and prices are Python literals in 6+ files, best-config map duplicated 3 times | nothing |
| Ordinal alpha everywhere | Confirmed the pipeline defaults to ordinal. Any older figure computed nominally understates by 0.10-0.30 | nothing |

---

## Tried, did not work

Recorded so nobody repeats them. Each carries a re-test trigger rather than a permanent
verdict, because "did not work once" is evidence, not proof.

| Attempt | Result | Re-test when |
|---|---|---|
| Evidently automated prompt optimisation | Degraded agreement on 4 of 5 dimensions | a different optimiser, or after a gold set exists to optimise against |
| Self-reported model confidence for routing | No discriminative value | never, unless a model exposes calibrated confidence |
| Gemini Lite as primary | Negative alpha on most dimensions | not planned |
| Hand-written prompt P1 as a global replacement | More than doubled Relevance but collapsed Linguistic Feature from 0.536 to 0.091 | prompts are dimension-specific; do not seek one global best |

---

## Known limitations, stated up front

1. Two annotators only. No third rater, so no alt-test.
2. n=100 per configuration. Bootstrap CI width about 0.32; the two Tier A dimensions cannot
   be confirmed above the routing gate at this sample size.
3. OpenStax source. Contamination not excluded.
4. Clarity carries no usable model signal under the current rubric.
5. Workload figures on the threshold sweep count dimensions exempted from scoring, not
   annotator time. Reading the item is a fixed cost routing does not remove.
6. Two dashboard tabs show synthetic data, documented in the main README.

---

## How to read the numbers

- Always against the ceiling. An alpha of 0.558 means nothing alone. Against a measured
  human ceiling of 0.809 on identical items, it means 69%.
- Ordinal, not nominal. The scale is 1-4 ordinal. Nominal alpha treats 3-vs-4 as
  identically wrong to 1-vs-4 and understates agreement substantially.
- Matched items only. Do not divide an alpha computed on a filtered subset by one
  computed on the full set. Different marginals give different expected disagreement, so
  the ratio is not meaningful. This error was made and corrected on 2026-08-04.
- Check the modal share. If the model puts most items on one score while humans spread,
  alpha collapses for reasons that have nothing to do with the model understanding the task.
