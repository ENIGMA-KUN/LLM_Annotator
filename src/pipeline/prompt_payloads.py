"""
Prompt Payloads: Prometheus-Eval-style strict rubric construction.

Design principles (adapted from Prometheus-Eval):
    1. Mathematical boundary heuristics the model must check BEFORE scoring.
       Each heuristic is a hard ceiling: if the condition fires, the score
       for that dimension is capped regardless of holistic impression.
    2. A mandatory anti-bias calibration preamble injected into every
       system prompt (LLMs systematically over-rate quality).
    3. Evidence-before-score ordering: the JSON schema forces the model to
       emit measured evidence fields before it is allowed to emit a score.

Usage:
    from src.pipeline.prompt_payloads import build_strict_prompt
    prompt = build_strict_prompt()           # all 5 dimensions
    prompt = build_strict_prompt(["Linguistic_Feature"])  # subset
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Mandatory anti-bias preamble (verbatim per protocol)
# ---------------------------------------------------------------------------
ANTI_BIAS_PREAMBLE = (
    "IMPORTANT CALIBRATION NOTE: LLM annotators systematically over-rate "
    "quality. You MUST actively look for flaws. A score of 4 should be rare "
    "and reserved for truly excellent items. Use the full 1-4 range. If in "
    "doubt, score lower rather than higher."
)

# ---------------------------------------------------------------------------
# Boundary heuristics: hard ceilings checked before holistic judgment.
# Each entry: (heuristic_id, condition, max_score)
# ---------------------------------------------------------------------------
BOUNDARY_HEURISTICS: dict[str, list[dict]] = {
    "Linguistic_Feature": [
        {
            "id": "LF-H1",
            "rule": "If the correct answer is more than 1.5x the mean word "
                    "length of the distractors (or vice versa), the maximum "
                    "score is 2.",
            "max_score": 2,
        },
        {
            "id": "LF-H2",
            "rule": "If any independent clause in an option exceeds 25 words "
                    "without a coordinating conjunction or terminal "
                    "punctuation (run-on indicator), the maximum score is 2.",
            "max_score": 2,
        },
        {
            "id": "LF-H3",
            "rule": "If two or more options begin with a different grammatical "
                    "category than the correct answer (e.g., gerund vs. noun "
                    "phrase), the maximum score is 3.",
            "max_score": 3,
        },
    ],
    "Relevance": [
        {
            "id": "REL-H1",
            "rule": "If the question stem contains zero content words "
                    "(nouns/verbs) that appear in, or are synonyms of words "
                    "in, the passage's first or last paragraph, the maximum "
                    "score is 2.",
            "max_score": 2,
        },
        {
            "id": "REL-H2",
            "rule": "If the question can be answered correctly without "
                    "reading the passage at all, the maximum score is 2.",
            "max_score": 2,
        },
    ],
    "Clarity": [
        {
            "id": "CLA-H1",
            "rule": "If the question stem contains a pronoun whose antecedent "
                    "could be two or more different nouns, the maximum score "
                    "is 2.",
            "max_score": 2,
        },
        {
            "id": "CLA-H2",
            "rule": "If the stem exceeds 35 words OR contains a double "
                    "negative, the maximum score is 3.",
            "max_score": 3,
        },
    ],
    "Semantic_Plausibility": [
        {
            "id": "SP-H1",
            "rule": "If any distractor is a category error relative to the "
                    "question (e.g., a date offered when a mechanism is "
                    "asked), the maximum score is 2.",
            "max_score": 2,
        },
        {
            "id": "SP-H2",
            "rule": "If two or more distractors do not appear in, and cannot "
                    "be inferred from, the passage, the maximum score is 3.",
            "max_score": 3,
        },
    ],
    "Semantic_Uniqueness": [
        {
            "id": "SU-H1",
            "rule": "If any distractor is a paraphrase of the correct answer "
                    "(shares its core proposition), the maximum score is 1.",
            "max_score": 1,
        },
        {
            "id": "SU-H2",
            "rule": "If a well-prepared student could cite a passage sentence "
                    "defending a distractor as correct, the maximum score "
                    "is 2.",
            "max_score": 2,
        },
    ],
}

# Holistic rubric anchors used AFTER heuristics pass
RUBRIC_ANCHORS: dict[str, str] = {
    "Linguistic_Feature": (
        "1 = correct answer identifiable by form alone; "
        "2 = noticeable structural asymmetry; "
        "3 = mostly parallel, minor asymmetries; "
        "4 = options indistinguishable by surface features."
    ),
    "Relevance": (
        "1 = tests trivia unrelated to main ideas; "
        "2 = peripheral detail; "
        "3 = important supporting concept; "
        "4 = central argument / primary learning objective."
    ),
    "Clarity": (
        "1 = unanswerable as written (multiple readings); "
        "2 = requires re-reading; "
        "3 = clear with minor stylistic issues; "
        "4 = precise and unambiguous on first pass."
    ),
    "Semantic_Plausibility": (
        "1 = distractors obviously wrong; "
        "2 = only one genuine competitor; "
        "3 = multiple distractors require thought; "
        "4 = all distractors are plausible misconceptions."
    ),
    "Semantic_Uniqueness": (
        "1 = two or more defensible answers; "
        "2 = correct answer arguable; "
        "3 = correct answer clearly best; "
        "4 = uniquely supported, distractors clearly contradicted."
    ),
}

ALL_DIMENSIONS = list(BOUNDARY_HEURISTICS.keys())


def build_strict_prompt(dimensions: list[str] | None = None) -> str:
    """
    Assemble a Prometheus-style strict system prompt.

    Structure: anti-bias preamble → per-dimension boundary heuristics →
    rubric anchors → evidence-first JSON schema.

    Args:
        dimensions: subset of dimensions to include (default: all 5).

    Returns:
        Complete system prompt string.
    """
    dims = dimensions or ALL_DIMENSIONS
    unknown = set(dims) - set(ALL_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown dimensions: {unknown}")

    sections = [
        "You are a strict educational assessment auditor evaluating "
        "multiple-choice questions (MCQs) for a Grammatical Error "
        "Correction research dataset.",
        "",
        ANTI_BIAS_PREAMBLE,
        "",
        "## MANDATORY EVALUATION PROCEDURE",
        "For EACH dimension you MUST, in this order:",
        "1. Check every boundary heuristic below. If a heuristic fires, "
        "record its ID and apply its score ceiling — no exceptions.",
        "2. Only then apply the holistic rubric anchors, never exceeding "
        "any active ceiling.",
        "3. Quote the specific text evidence that triggered (or cleared) "
        "each heuristic BEFORE you state the score.",
        "",
    ]

    for dim in dims:
        sections.append(f"### {dim.replace('_', ' ').upper()}")
        sections.append("Boundary heuristics (hard ceilings):")
        for h in BOUNDARY_HEURISTICS[dim]:
            sections.append(f"- [{h['id']}] {h['rule']}")
        sections.append(f"Rubric anchors: {RUBRIC_ANCHORS[dim]}")
        sections.append("")

    dim_schema = ",\n".join(
        f'  "{dim}": {{\n'
        f'    "heuristics_checked": ["list of heuristic IDs evaluated"],\n'
        f'    "heuristics_fired": ["IDs whose condition is met, [] if none"],\n'
        f'    "evidence": "verbatim quotes that triggered/cleared heuristics",\n'
        f'    "ceiling": "lowest max_score among fired heuristics, or 4",\n'
        f'    "score": "integer 1-4, MUST NOT exceed ceiling"\n'
        f'  }}'
        for dim in dims
    )

    sections += [
        "## OUTPUT FORMAT",
        "Respond with ONLY this JSON object (no markdown fences):",
        "{",
        dim_schema,
        "}",
        "",
        "Final reminder: " + ANTI_BIAS_PREAMBLE,
    ]
    return "\n".join(sections)


def save_strict_prompt(path, dimensions: list[str] | None = None) -> None:
    """Write the assembled strict prompt to a prompts/ file for versioning."""
    Path(path).write_text(build_strict_prompt(dimensions), encoding="utf-8")
