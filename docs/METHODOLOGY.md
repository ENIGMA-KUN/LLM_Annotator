# Methodology

> Why each measurement choice was made, and what it does and does not license.

---

## The estimand

For each rubric dimension: how much of the agreement a second human annotator would
contribute does the LLM recover, on identical items?

Not "is the LLM accurate." Accuracy against a noisy single human label is not a meaningful
quantity on a subjective construct.

---

## Estimator: ordinal Krippendorff alpha

The scale is ordinal 1 to 4. Nominal alpha treats a 3-vs-4 disagreement as identically
wrong to 1-vs-4, which understates agreement substantially here. Measured on this data:

| Dimension | Nominal | Ordinal |
|---|---|---|
| Linguistic Feature | 0.543 | 0.835 |
| Relevance | 0.526 | 0.741 |
| Semantic Plausibility | 0.495 | 0.769 |

A factor of that size is not a rounding difference. Always report which was used.

`src/pipeline/metrics.py` defaults to ordinal. Any figure from an older document computed
nominally is wrong and low.

---

## The human ceiling

Both annotators rated all 200 items on all five dimensions, so human-to-human agreement is
directly measurable. This corrects a belief held across several planning documents that the
annotators covered non-overlapping sets.

Every LLM number is reported against that ceiling on identical items, never in
isolation. An alpha of 0.558 is meaningless alone; against a ceiling of 0.809 it is 69%.

The comparison must be matched. Do not divide an alpha computed on a filtered subset by
one computed on the full set. Different item sets give different marginal distributions and
therefore different expected disagreement, so the denominators are not the same quantity
and the ratio is not interpretable. This error was made on 2026-08-03 (reporting 86% and
81%) and corrected on 2026-08-04 (69% and 62%).

---

## Intervals

Bootstrap, 2000 resamples, resampled at the item level. Items are the independent unit;
ratings within an item are not.

A threshold is not met if the interval crosses it. Both Tier A dimensions sit above the
0.50 routing gate on the point estimate while their intervals still span it, so the correct
statement is "we cannot reject the null that this dimension falls below the gate," not
"this dimension passes."

---

## Marginal distributions: the diagnostic that matters

Reporting alpha alone is insufficient under skewed marginals. When one rater concentrates
nearly all mass on a single category, chance-expected disagreement collapses and alpha
falls toward or below zero regardless of how well the other rater discriminates. This is
the Feinstein-Cicchetti prevalence paradox.

So every dimension reports the modal share for both the humans and the model:

| Dimension | Human modal | LLM modal | Alpha | Reading |
|---|---|---|---|---|
| Linguistic Feature | 38% | 37% | +0.558 | distributions match, works |
| Relevance | 52% | 66% | +0.461 | mild over-concentration |
| Semantic Plausibility | 41% | 45% | +0.225 | distribution fine, signal wrong |
| Semantic Uniqueness | 80% | 93% | +0.172 | both concentrated, low signal |
| Clarity | 45% | 95% | -0.082 | collapsed |

This single comparison separates "the task carries no information" from "the model stopped
discriminating." Humans discriminate on Clarity (45% modal, alpha 0.873). The model does
not. That is a rubric failure with a known fix, not an unmeasurable construct.

On the negative value: -0.082 sits in the band conventionally read as no agreement, not
systematic inversion. Do not describe it as worse than random.

---

## The uncertainty signal

Three independent runs at temperature 0.3. Temperature 0 was rejected: eliminating variance
removes the signal the router depends on.

Self-reported model confidence was tested and discarded. It clusters at 90 to 100 with no
discriminative value, so every item reads as high confidence and nothing routes.

Self-consistency is not correctness. Three runs agree perfectly on a collapsed
dimension while carrying zero information. Clarity reaches 99% run consensus and near-zero
agreement with humans. This is why the dimension router sits above the confidence gate
rather than the gate acting alone.

---

## Separating correct answers from distractors

The model recognises correct answers and scores them differently from distractors. Any
pooled number mixing them is confounded, so evaluations report all-items and
distractors-only as separate rows, plus the correct-answer identification rate as its own
auxiliary result.

---

## What this design cannot support

Stated now, while it is honest, rather than during review.

1. No third annotator, so the alt-test (Calderon, Reichart & Dror, ACL 2025) is
   unavailable. There is no published threshold for "an LLM may replace one of two human
   annotators"; the alt-test is the nearest standard and it needs three raters.
2. n=100 per configuration. Bootstrap CI width about 0.32. Roughly 300 items would be
   needed to confirm the two Tier A dimensions above the gate.
3. Contamination not excluded. OpenStax is public and likely in pre-training.
4. Unit-of-analysis mismatch, unresolved. The rubric defines Clarity on the question
   stem, Linguistic Feature and Semantic Plausibility on the distractor, and Relevance on
   response options; Semantic Uniqueness is absent from `rubric.json`. The intake file is
   one row per question containing only the correct answer. Until it is confirmed what the
   prompt and the human annotators each saw, agreement numbers on the distractor
   dimensions are not fully interpretable.
5. Workload figures count dimensions exempted from scoring, not annotator time. Reading
   the item is a fixed cost routing does not remove, so per-dimension exemption saves far
   less than the dimension count implies.

---

## Operational bar

Williamson, Xi & Breyer (2012), the standard in educational measurement, require QWK >= 0.70
and machine-human within 0.10 of human-human. Against that rule, Linguistic Feature at
0.558 versus a ceiling of 0.809 is a 0.25 gap: well outside. That is the concrete target to
work toward rather than an arbitrary threshold.

---

## References

- Krippendorff. Reliability in Content Analysis. Human Communication Research 30(3), 2004.
- Williamson, Xi & Breyer. A Framework for Evaluation and Use of Automated Scoring.
  Educational Measurement: Issues and Practice 31(1), 2012.
- Calderon, Reichart & Dror. The Alternative Annotator Test for LLM-as-a-Judge. ACL 2025.
- Feinstein & Cicchetti. High agreement but low kappa. J Clin Epidemiol, 1990.
- Dettori & Norvell. Kappa and Beyond. Global Spine J, 2020.
- Plank. The "Problem" of Human Label Variation. EMNLP 2022.
- Richie et al. Inter-annotator agreement is not the ceiling of machine learning
  performance. BioNLP 2022.
