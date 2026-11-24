// ---------------------------------------------------------------------------
// Mock dataset: 200 MCQ items for the Hybrid Routing Protocol.
//
// Tier A (Automated w/ 3-Run Verification): Linguistic_Feature, Topic_Relevance
// Tier C (Direct to Human):                 Writing_Clarity,
//                                           Semantic_Plausibility,
//                                           Semantic_Uniqueness
// ---------------------------------------------------------------------------

export const TIER_A_METRICS = [
  { key: "Linguistic_Feature", label: "Linguistic Feature" },
  { key: "Topic_Relevance", label: "Topic Relevance" },
];

export const TIER_C_METRICS = [
  { key: "Writing_Clarity", label: "Writing Clarity" },
  { key: "Semantic_Plausibility", label: "Semantic Plausibility" },
  { key: "Semantic_Uniqueness", label: "Semantic Uniqueness" },
];

export const ALL_METRICS = [...TIER_A_METRICS, ...TIER_C_METRICS];

// Gate decisions for a 3-run array
export const GATE = {
  AUTO_ACCEPT: "auto_accept", // 3/3 consensus
  NEEDS_VERIFY: "needs_verification", // 2/3 consensus
  ESCALATED: "escalated", // 0/3 consensus
};

export function gateDecision(runs) {
  const counts = {};
  runs.forEach((r) => (counts[r] = (counts[r] || 0) + 1));
  const max = Math.max(...Object.values(counts));
  if (max === 3) return GATE.AUTO_ACCEPT;
  if (max === 2) return GATE.NEEDS_VERIFY;
  return GATE.ESCALATED;
}

export function consensusScore(runs) {
  const counts = {};
  runs.forEach((r) => (counts[r] = (counts[r] || 0) + 1));
  const [score, count] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return count >= 2 ? Number(score) : null;
}

// Median of the 3 runs, used by the Full Automation (LLM-only) mode
// to force a score even when there is no consensus.
export function medianScore(runs) {
  const sorted = [...runs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// Deterministic PRNG so the queue is stable across reloads
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GRAMMAR_TOPICS = [
  {
    passage:
      "A run-on sentence occurs when two or more independent clauses are joined without proper punctuation or a coordinating conjunction. Writers often produce comma splices, fusing clauses with only a comma, which obscures the logical boundary between complete thoughts.",
    question:
      "Which revision best corrects the run-on sentence: 'The experiment failed the team repeated it'?",
    options: [
      "The experiment failed, the team repeated it.",
      "The experiment failed; the team repeated it.",
      "The experiment failed the team, repeated it.",
      "The experiment, failed the team repeated it.",
    ],
    answer: "B",
  },
  {
    passage:
      "Subordinating conjunctions such as 'although', 'because', and 'while' convert an independent clause into a dependent clause. A dependent clause cannot stand alone as a sentence; doing so creates a sentence fragment, a common error in early academic writing.",
    question:
      "Which of the following is a sentence fragment rather than a complete sentence?",
    options: [
      "Although the results were promising.",
      "The results were promising.",
      "The results, although promising, were preliminary.",
      "Promising results emerged early.",
    ],
    answer: "A",
  },
  {
    passage:
      "Parallel structure requires that items in a series share the same grammatical form. Faulty parallelism, such as mixing gerunds with infinitives in a single list, disrupts rhythm and can mislead readers about the relationship between ideas.",
    question:
      "Which sentence demonstrates correct parallel structure?",
    options: [
      "She enjoys reading, writing, and to hike.",
      "She enjoys reading, writing, and hiking.",
      "She enjoys to read, writing, and hiking.",
      "She enjoys reading, to write, and hiked.",
    ],
    answer: "B",
  },
  {
    passage:
      "A coordinating conjunction (for, and, nor, but, or, yet, so) joined with a comma is one valid way to connect two independent clauses. Omitting the comma before the conjunction in long clauses can produce ambiguity about where one clause ends.",
    question:
      "Where should a comma be placed in: 'The tutor explained the rule but the students still struggled with clause boundaries'?",
    options: [
      "After 'rule'",
      "After 'but'",
      "After 'students'",
      "No comma is needed",
    ],
    answer: "A",
  },
  {
    passage:
      "Logical connectives in argumentative writing signal the relationship between premises and conclusions. Words like 'therefore' and 'consequently' mark inference, while 'however' and 'nevertheless' mark contrast. Misusing connectives weakens an argument's coherence.",
    question:
      "Which connective best completes: 'All observed swans were white; ___, the next swan may still be black'?",
    options: ["therefore", "furthermore", "nevertheless", "similarly"],
    answer: "C",
  },
  {
    passage:
      "A modifier should be placed next to the word it modifies. A dangling modifier appears when the intended subject is missing from the sentence entirely, forcing readers to guess the actor, a frequent issue in passive scientific prose.",
    question:
      "Which sentence contains a dangling modifier?",
    options: [
      "Walking to class, the rain soaked my notes.",
      "Walking to class, I was soaked by the rain.",
      "As I walked to class, the rain soaked my notes.",
      "The rain soaked my notes while I walked to class.",
    ],
    answer: "A",
  },
  {
    passage:
      "Conditional reasoning follows the form 'If P, then Q'. Affirming the consequent (concluding P because Q is true) is a formal fallacy. Distinguishing valid modus ponens from this fallacy is central to logical literacy.",
    question:
      "Given 'If it rains, the ground is wet' and 'The ground is wet', what can be validly concluded?",
    options: [
      "It rained.",
      "It did not rain.",
      "Nothing can be validly concluded.",
      "The ground is always wet.",
    ],
    answer: "C",
  },
  {
    passage:
      "Pronoun-antecedent agreement requires a pronoun to match its antecedent in number and person. Ambiguous pronoun reference, where 'it' or 'they' could point to multiple nouns, is a leading cause of unclear academic writing.",
    question:
      "Why is the sentence 'When the beaker touched the burner, it cracked' ambiguous?",
    options: [
      "'It' could refer to the beaker or the burner.",
      "'Touched' is the wrong tense.",
      "'Cracked' is intransitive.",
      "The sentence is a fragment.",
    ],
    answer: "A",
  },
];

// Status values for the queue
export const STATUS = {
  AUTO_ACCEPTED: "auto_accepted",
  NEEDS_REVIEW: "needs_review",
  PENDING_HUMAN: "pending_human",
  COMPLETED: "completed",
};

/**
 * Generate runs with a controlled agreement pattern.
 * pattern: "3of3" | "2of3" | "split"
 */
function makeRuns(rand, pattern) {
  const base = 1 + Math.floor(rand() * 4); // 1..4
  if (pattern === "3of3") return [base, base, base];
  if (pattern === "2of3") {
    let other = 1 + Math.floor(rand() * 4);
    if (other === base) other = (base % 4) + 1;
    const runs = [base, base, other];
    // shuffle
    return runs.sort(() => rand() - 0.5);
  }
  // split: three distinct values
  const pool = [1, 2, 3, 4].sort(() => rand() - 0.5);
  return pool.slice(0, 3);
}

export function generateMockItems(n = 200) {
  const rand = mulberry32(42);
  const items = [];

  for (let i = 0; i < n; i++) {
    const topic = GRAMMAR_TOPICS[i % GRAMMAR_TOPICS.length];

    // Distribution design:
    //  - ~55% of items: 3/3 agreement on Linguistic_Feature (auto-accept)
    //  - ~20% of items: massive disagreement (split) on Topic_Relevance
    //  - remainder mixed
    const roll = rand();
    let lfPattern, trPattern;
    if (roll < 0.55) {
      lfPattern = "3of3";
      trPattern = rand() < 0.6 ? "3of3" : "2of3";
    } else if (roll < 0.75) {
      lfPattern = rand() < 0.5 ? "2of3" : "3of3";
      trPattern = "split"; // forced human review simulation
    } else {
      lfPattern = rand() < 0.5 ? "2of3" : "split";
      trPattern = rand() < 0.5 ? "2of3" : "3of3";
    }

    const tierARuns = {
      Linguistic_Feature: makeRuns(rand, lfPattern),
      Topic_Relevance: makeRuns(rand, trPattern),
    };

    // Tier C runs exist but are unreliable (alpha < 0.30 vs humans):
    // mostly 2/3 or split agreement. Only used by Full Automation mode.
    const tierCPattern = () => {
      const r = rand();
      if (r < 0.25) return "3of3";
      if (r < 0.65) return "2of3";
      return "split";
    };
    const tierCRuns = {
      Writing_Clarity: makeRuns(rand, tierCPattern()),
      Semantic_Plausibility: makeRuns(rand, tierCPattern()),
      Semantic_Uniqueness: makeRuns(rand, tierCPattern()),
    };

    items.push({
      id: `MCQ-${String(i + 1).padStart(3, "0")}`,
      passage: topic.passage,
      question: topic.question,
      options: topic.options,
      answer: topic.answer,
      tierARuns,
      tierCRuns,
      // Human-entered scores live here after submission
      humanScores: {},
      // Accepted Tier A scores (consensus or human override)
      acceptedScores: {},
      completed: false,
    });
  }
  return items;
}

/**
 * Derive an item's queue status from its gate decisions + completion flag.
 */
export function deriveStatus(item) {
  if (item.completed) return STATUS.COMPLETED;
  const decisions = Object.values(item.tierARuns).map(gateDecision);
  const allAuto = decisions.every((d) => d === GATE.AUTO_ACCEPT);
  if (decisions.some((d) => d === GATE.ESCALATED)) return STATUS.NEEDS_REVIEW;
  if (decisions.some((d) => d === GATE.NEEDS_VERIFY)) return STATUS.NEEDS_REVIEW;
  // Even with full Tier A consensus, Tier C always needs direct human input
  return allAuto ? STATUS.PENDING_HUMAN : STATUS.NEEDS_REVIEW;
}
