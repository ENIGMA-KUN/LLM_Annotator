"""
Pairwise Elo: FastChat-style comparison for subjective dimensions.

Rationale:
    Absolute 1-4 ordinal scoring on subjective dimensions (Clarity,
    Semantic_Plausibility, Semantic_Uniqueness) suffers from central
    tendency bias: LLM judges compress everything into 3s. Pairwise
    comparison ("which of these two items has better clause structure?")
    is a strictly easier judgment and removes the anchor problem.

Protocol (adapted from LMSYS FastChat / Chatbot Arena):
    1. Sample item pairs (round-robin or random with a swiss-style budget).
    2. Ask the judge LLM which item is better on ONE dimension, with
       position randomization to cancel order bias.
    3. Update Elo ratings from the win/loss/tie outcomes.
    4. Map final Elo ratings back onto the 1-4 ordinal scale by quartile,
       so downstream alpha computation against the human baseline still works.

Usage:
    judge = PairwiseJudge(engine)        # any AnnotationEngine-like caller
    tournament = EloTournament(items, dimension="Semantic_Plausibility")
    pairs = tournament.schedule_pairs(budget=400)
    for a, b in pairs:
        outcome = judge.compare(a, b, dimension="Semantic_Plausibility")
        tournament.record(a["question_uuid"], b["question_uuid"], outcome)
    ordinal = tournament.to_ordinal_scores()   # {uuid: 1..4}
"""

import itertools
import json
import random
import re
from typing import Optional

import numpy as np

from src.utils import setup_logging

logger = setup_logging()

ELO_INITIAL = 1000.0
ELO_K = 32.0

PAIRWISE_PROMPT_TEMPLATE = """You are a strict educational assessment auditor.
Compare the following two MCQ items on EXACTLY ONE dimension: {dimension_description}

Do NOT consider any other quality aspect. Ties are allowed but should be rare —
actively look for a difference before declaring a tie.

## ITEM 1
Passage: {passage_1}
Question: {question_1}
Correct answer: {answer_1}

## ITEM 2
Passage: {passage_2}
Question: {question_2}
Correct answer: {answer_2}

Respond with ONLY this JSON object:
{{"winner": "1" | "2" | "tie", "reasoning": "one sentence citing specific text"}}
"""

DIMENSION_DESCRIPTIONS = {
    "Clarity": (
        "Writing Clarity — which question stem is less ambiguous and easier "
        "to parse on first reading (clause boundaries, pronoun reference, "
        "sentence length)?"
    ),
    "Semantic_Plausibility": (
        "Semantic Plausibility — which item's distractors are more plausibly "
        "attractive to a student with partial understanding?"
    ),
    "Semantic_Uniqueness": (
        "Semantic Uniqueness — which item's correct answer is more uniquely "
        "defensible against its distractors?"
    ),
}


class PairwiseJudge:
    """Wraps an LLM caller to produce pairwise verdicts with position debiasing."""

    def __init__(self, engine, rng: Optional[random.Random] = None):
        """
        Args:
            engine: object exposing ``model.generate_content`` like
                    AnnotationEngine (reuses its retry/cost plumbing via
                    a single-run annotate call).
            rng: seeded RNG for reproducible position randomization.
        """
        self.engine = engine
        self.rng = rng or random.Random(42)

    def compare(self, item_a: dict, item_b: dict, dimension: str) -> str:
        """
        Compare two items on one dimension.

        Returns:
            "a", "b", or "tie" (already corrected for position swap).
        """
        if dimension not in DIMENSION_DESCRIPTIONS:
            raise ValueError(f"Pairwise not configured for: {dimension}")

        swapped = self.rng.random() < 0.5
        first, second = (item_b, item_a) if swapped else (item_a, item_b)

        prompt = PAIRWISE_PROMPT_TEMPLATE.format(
            dimension_description=DIMENSION_DESCRIPTIONS[dimension],
            passage_1=str(first.get("text", ""))[:2000],
            question_1=first.get("question", ""),
            answer_1=first.get("answer", ""),
            passage_2=str(second.get("text", ""))[:2000],
            question_2=second.get("question", ""),
            answer_2=second.get("answer", ""),
        )

        raw = self._call(prompt)
        verdict = self._parse_verdict(raw)
        if verdict == "tie":
            return "tie"
        # Undo position swap
        if swapped:
            return "a" if verdict == "2" else "b"
        return "a" if verdict == "1" else "b"

    def _call(self, prompt: str) -> str:
        response = self.engine.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        return response.text

    @staticmethod
    def _parse_verdict(raw: str) -> str:
        try:
            parsed = json.loads(raw)
            w = str(parsed.get("winner", "tie")).strip().lower()
            return w if w in ("1", "2", "tie") else "tie"
        except json.JSONDecodeError:
            m = re.search(r'"winner"\s*:\s*"(1|2|tie)"', raw)
            return m.group(1) if m else "tie"


class EloTournament:
    """Maintains Elo ratings for items on one subjective dimension."""

    def __init__(self, items: list[dict], dimension: str, seed: int = 42):
        self.dimension = dimension
        self.items = {it["question_uuid"]: it for it in items}
        self.ratings = {uid: ELO_INITIAL for uid in self.items}
        self.history: list[dict] = []
        self.rng = random.Random(seed)

    def schedule_pairs(self, budget: int) -> list[tuple[dict, dict]]:
        """
        Schedule up to ``budget`` comparisons.

        Strategy: shuffled round-robin truncated to budget. For n=200 items
        a budget of 400-600 (2-3 comparisons per item) is typically enough
        for stable quartile assignment.
        """
        all_pairs = list(itertools.combinations(self.items.keys(), 2))
        self.rng.shuffle(all_pairs)
        chosen = all_pairs[:budget]
        return [(self.items[a], self.items[b]) for a, b in chosen]

    def record(self, uuid_a: str, uuid_b: str, outcome: str) -> None:
        """
        Update Elo ratings.

        Args:
            outcome: "a", "b", or "tie".
        """
        ra, rb = self.ratings[uuid_a], self.ratings[uuid_b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = {"a": 1.0, "b": 0.0, "tie": 0.5}[outcome]
        self.ratings[uuid_a] = ra + ELO_K * (sa - ea)
        self.ratings[uuid_b] = rb + ELO_K * ((1.0 - sa) - (1.0 - ea))
        self.history.append({
            "dimension": self.dimension,
            "a": uuid_a, "b": uuid_b, "outcome": outcome,
            "rating_a": round(self.ratings[uuid_a], 1),
            "rating_b": round(self.ratings[uuid_b], 1),
        })

    def to_ordinal_scores(self) -> dict[str, int]:
        """
        Map final Elo ratings to the 1-4 ordinal scale by quartile, so the
        pairwise pipeline remains comparable to the human 1-4 baseline.
        """
        uuids = list(self.ratings.keys())
        values = np.array([self.ratings[u] for u in uuids])
        quartiles = np.quantile(values, [0.25, 0.5, 0.75])
        scores = {}
        for u, v in zip(uuids, values):
            scores[u] = int(1 + np.searchsorted(quartiles, v, side="right"))
        logger.info(
            f"[{self.dimension}] Elo→ordinal: "
            f"{sum(1 for s in scores.values() if s == 1)}×1, "
            f"{sum(1 for s in scores.values() if s == 2)}×2, "
            f"{sum(1 for s in scores.values() if s == 3)}×3, "
            f"{sum(1 for s in scores.values() if s == 4)}×4"
        )
        return scores

    def leaderboard(self, top: int = 10) -> list[tuple[str, float]]:
        """Top-rated items (useful for sanity-checking judge behavior)."""
        ranked = sorted(self.ratings.items(), key=lambda kv: -kv[1])
        return [(u, round(r, 1)) for u, r in ranked[:top]]
