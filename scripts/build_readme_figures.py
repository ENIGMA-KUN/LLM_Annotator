"""Build the README figures in docs/images/.

reliability.svg and workload.svg are computed from the experiment result files,
so every number in them traces to data. routing.svg and roadmap.svg are
diagrams; edit ROADMAP below when the project status changes. Inputs live under
Data/, which is not distributed; maintainers rerun this after any change.

    python scripts/build_readme_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Data" / "results"
LLM_CSV = RESULTS / "neurips_experiments" / "results_with_cis.csv"
ROUTING_CSV = RESULTS / "neurips_experiments" / "item_level_routing.csv"
HUMAN_CSV = RESULTS / "human_baseline.csv"
CALL_LOGS = RESULTS / "neurips_experiments"
OUT_DIR = ROOT / "docs" / "images"

CRITERIA = [
    ("Linguistic_Feature", "Linguistic Feature"),
    ("Relevance", "Topic Relevance"),
    ("Semantic_Uniqueness", "Semantic Uniqueness"),
    ("Semantic_Plausibility", "Semantic Plausibility"),
    ("Clarity", "Writing Clarity"),
]
GATE = 0.50
RUNS_PER_ITEM = 3

# Share of the 1,000 decisions (200 questions x 5 criteria) auto-accepted when
# the cached experiment scores are replayed through the API pipeline. The replay
# writes no result file, so the figure is recorded here from the run log.
REPLAY_AUTO_PCT = 35.4

MODEL_NAMES = {"gemini_flash": "Gemini 2.5 Flash", "gemini_lite": "Gemini 2.5 Flash-Lite", "gpt54": "GPT-5.4"}
PROMPT_NAMES = {"P0_baseline": "zero-shot", "P1_manual": "rubric prompt", "P2_evidently": "optimised prompt"}

ROADMAP = {
    "Done": [
        "45 configurations: 3 models, 3 prompts, 5 criteria, 3 runs each",
        "Expert ceiling on all 200 questions, with intervals",
        "Routing protocol with blind expert review and label provenance",
        "Targeted review beats random: alpha 0.762 vs 0.659 at a 30% budget (Relevance)",
        "Fairness audit by subject, difficulty and passage length",
        "Negative result: automated prompt optimisation collapses to constant scores",
    ],
    "In progress": [
        "Retrieval few-shot: an offline probe finds label signal for Relevance and Plausibility",
        "Rubric rewrite from holistic 1 to 4 levels to atomic yes/no checks",
        "Checking that each prompt shows the model the same object the experts judged",
    ],
    "Next": [
        "Live A/B test on Semantic Plausibility, about 40 questions × 3 runs",
        "Live pairwise judge on the three criteria that stay with experts",
        "Grow the benchmark to about 300 questions to tighten the intervals",
        "Link quality scores to student performance data",
    ],
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W = 860

STYLE = """<style>
  .bg{fill:#ffffff;stroke:#d0d7de}.ink{fill:#1f2328}.ink2{fill:#59636e}.muted{fill:#6e7781}
  .box{fill:#f6f8fa;stroke:#d0d7de;stroke-width:1}
  .grid{stroke:#eaeef2;stroke-width:1}.zero{stroke:#8c959f;stroke-width:1}
  .gate{stroke:#1f2328;stroke-width:1.5;stroke-dasharray:4 4}
  .llm{fill:#a12f55}.exp{fill:#4f82c4}.rule{fill:#ffc627}
  .llmT{fill:#a12f55;fill-opacity:.10;stroke:#a12f55;stroke-width:1.5}
  .llmF{fill:#a12f55;fill-opacity:.06;stroke:#a12f55;stroke-width:1.5;stroke-dasharray:5 3}
  .expT{fill:#4f82c4;fill-opacity:.10;stroke:#4f82c4;stroke-width:1.5}
  .arrow{stroke:#8c959f;stroke-width:1.5;fill:none}.head{fill:#8c959f}
  .tick{stroke:#1f2328;stroke-width:1.5}
  .dotFull{fill:#1f2328}.dotRing{fill:none;stroke:#1f2328;stroke-width:1.5}
  .ci line{stroke:#1f2328;stroke-width:1.5;stroke-opacity:.7}
  .tnum{font-variant-numeric:tabular-nums}
  @media (prefers-color-scheme: dark){
    .bg{fill:#161b22;stroke:#30363d}.ink{fill:#e6edf3}.ink2{fill:#9da7b3}.muted{fill:#848d97}
    .box{fill:#0d1117;stroke:#30363d}
    .grid{stroke:#21262d}.zero{stroke:#6e7681}.gate{stroke:#e6edf3}
    .llm{fill:#d9577a}.exp{fill:#5f97d8}
    .llmT{fill:#d9577a;fill-opacity:.16;stroke:#d9577a}
    .llmF{fill:#d9577a;fill-opacity:.08;stroke:#d9577a}
    .expT{fill:#5f97d8;fill-opacity:.16;stroke:#5f97d8}
    .arrow{stroke:#6e7681}.head{fill:#6e7681}.tick{stroke:#e6edf3}
    .dotFull{fill:#e6edf3}.dotRing{stroke:#e6edf3}.ci line{stroke:#e6edf3}
  }
</style>"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v: float) -> str:
    return f"{v:.3f}".replace("-", "−")


def open_svg(h: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">',
        f'<title id="t">{esc(title)}</title>',
        f'<desc id="d">{esc(desc)}</desc>',
        STYLE,
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>',
        f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="12"/>',
        '<rect class="rule" x="32" y="28" width="44" height="4" rx="2"/>',
        f'<text class="ink" x="32" y="62" font-size="22" font-weight="600">{esc(title)}</text>',
    ]


def text(x: float, y: float, s: str, cls: str = "ink", size: float = 13, weight: int = 400, anchor: str = "start") -> str:
    w = f' font-weight="{weight}"' if weight != 400 else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}"{w}{a}>{esc(s)}</text>'


# ---------------------------------------------------------------- data -----

def require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit("Missing result files (Data/ is not distributed): " + ", ".join(missing))


def load_reliability() -> list[dict]:
    require(LLM_CSV, HUMAN_CSV)
    best: dict[str, dict] = {}
    with LLM_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            a = float(r["alpha"])
            if r["dimension"] not in best or a > best[r["dimension"]]["alpha"]:
                best[r["dimension"]] = {
                    "alpha": a,
                    "lo": float(r["ci_lower"]),
                    "hi": float(r["ci_upper"]),
                    "model": r["model"],
                    "prompt": r["prompt"],
                    "setup": f'{MODEL_NAMES.get(r["model"], r["model"])}, {PROMPT_NAMES.get(r["prompt"], r["prompt"])}',
                }
    human: dict[str, dict] = {}
    with HUMAN_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            human[r["dimension"]] = {"alpha": float(r["alpha_ordinal"]), "lo": float(r["ci_lower"]), "hi": float(r["ci_upper"])}
    return [{"key": k, "label": label, "llm": best[k], "human": human[k]} for k, label in CRITERIA]


def load_workload(rows: list[dict]) -> dict:
    require(ROUTING_CSV)
    routing = {}
    with ROUTING_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            routing[(r["model"], r["prompt"], r["dimension"])] = r

    auto_total = unanimous_total = 0.0
    for row in rows:
        if row["llm"]["alpha"] < GATE:
            continue
        r = routing[(row["llm"]["model"], row["llm"]["prompt"], row["key"])]
        auto_total += float(r["pct_auto_accept"])
        unanimous_total += float(r["pct_3of3"])

    costs = []
    for log in sorted(CALL_LOGS.glob("*/call_log.jsonl")):
        summary = None
        for line in log.read_text(encoding="utf-8").splitlines():
            if '"total_cost_usd"' in line:
                summary = json.loads(line)
        if summary and summary.get("total_calls"):
            questions = summary["total_calls"] / RUNS_PER_ITEM
            costs.append((summary["total_cost_usd"] / questions * 100, log.parent.name))

    return {
        "verified_pct": auto_total / len(CRITERIA),
        "unanimous_pct": unanimous_total / len(CRITERIA),
        "ceiling_pct": 100 * sum(r["llm"]["alpha"] >= GATE for r in rows) / len(CRITERIA),
        "cost_min": min(costs),
        "cost_max": max(costs),
        "n_cost_cells": len(costs),
    }


# --------------------------------------------------------- reliability -----

X_MIN, X_MAX = -0.30, 1.00
PLOT_L, PLOT_R = 262.0, 786.0
ROW_TOP, ROW_H = 152.0, 60.0
BAR_H, BAR_GAP = 12.0, 2.0


def px(v: float) -> float:
    return PLOT_L + (v - X_MIN) / (X_MAX - X_MIN) * (PLOT_R - PLOT_L)


def bar(v: float, y: float, cls: str) -> str:
    """Bar from zero to v, rounded only at the data end."""
    x0, x1 = px(0.0), px(v)
    r = min(4.0, abs(x1 - x0) / 2)
    h = BAR_H
    s = 1 if v >= 0 else -1
    d = (f"M{x0:.1f},{y:.1f} H{x1 - s * r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
         f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - s * r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return f'<path class="{cls}" d="{d}"/>'


def whisker(lo: float, hi: float, cy: float) -> str:
    a, b = px(lo), px(hi)
    return (f'<g class="ci"><line x1="{a:.1f}" y1="{cy:.1f}" x2="{b:.1f}" y2="{cy:.1f}"/>'
            f'<line x1="{a:.1f}" y1="{cy - 4:.1f}" x2="{a:.1f}" y2="{cy + 4:.1f}"/>'
            f'<line x1="{b:.1f}" y1="{cy - 4:.1f}" x2="{b:.1f}" y2="{cy + 4:.1f}"/></g>')


def build_reliability(rows: list[dict]) -> str:
    h = 548
    plot_bottom = ROW_TOP + ROW_H * len(rows)
    desc = "Krippendorff alpha per criterion. " + "; ".join(
        f'{r["label"]}: experts {r["human"]["alpha"]:.3f}, best LLM {r["llm"]["alpha"]:.3f} ({r["llm"]["setup"]})'
        for r in rows
    ) + ". Automation gate at 0.50."
    out = open_svg(h, "Where an LLM can stand in for an expert", desc)
    out.append(text(32, 86, "Agreement on five quality criteria for multiple-choice questions, measured as Krippendorff's alpha (ordinal).", "ink2"))

    ly = 118
    out.append(f'<rect class="llm" x="32" y="{ly - 10}" width="12" height="12" rx="3"/>')
    out.append(text(50, ly, "LLM, best of 3 models × 3 prompts", "ink2", 12.5))
    out.append(f'<rect class="exp" x="282" y="{ly - 10}" width="12" height="12" rx="3"/>')
    out.append(text(300, ly, "Expert vs. expert", "ink2", 12.5))
    out.append(f'<g class="ci"><line x1="430" y1="{ly - 4}" x2="452" y2="{ly - 4}"/>'
               f'<line x1="430" y1="{ly - 8}" x2="430" y2="{ly}"/><line x1="452" y1="{ly - 8}" x2="452" y2="{ly}"/></g>')
    out.append(text(460, ly, "95% interval", "ink2", 12.5))

    for t in (0.25, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{px(t):.1f}" y1="{ROW_TOP - 6}" x2="{px(t):.1f}" y2="{plot_bottom}"/>')
    out.append(f'<line class="zero" x1="{px(0):.1f}" y1="{ROW_TOP - 6}" x2="{px(0):.1f}" y2="{plot_bottom}"/>')

    for i, r in enumerate(rows):
        top = ROW_TOP + i * ROW_H
        verdict = "LLM scores, then an agreement check" if r["llm"]["alpha"] >= GATE else "Experts decide"
        out.append(text(32, top + 20, r["label"], "ink", 14, 600))
        out.append(text(32, top + 38, verdict, "ink2", 12))

        ly2 = top + 8
        out.append(bar(r["llm"]["alpha"], ly2, "llm"))
        out.append(whisker(r["llm"]["lo"], r["llm"]["hi"], ly2 + BAR_H / 2))
        lx = max(px(r["llm"]["alpha"]), px(r["llm"]["hi"]), px(0)) + 8
        out.append(text(lx, ly2 + 10, fmt(r["llm"]["alpha"]), "ink tnum", 12, 600))

        ey = ly2 + BAR_H + BAR_GAP
        out.append(bar(r["human"]["alpha"], ey, "exp"))
        out.append(whisker(r["human"]["lo"], r["human"]["hi"], ey + BAR_H / 2))
        ex = max(px(r["human"]["alpha"]), px(r["human"]["hi"])) + 8
        out.append(text(ex, ey + 10, fmt(r["human"]["alpha"]), "ink2 tnum", 12))

    gx = px(GATE)
    out.append(f'<line class="gate" x1="{gx:.1f}" y1="{ROW_TOP - 14}" x2="{gx:.1f}" y2="{plot_bottom + 4}"/>')
    out.append(text(gx, ROW_TOP - 20, "0.50 automation gate", "ink", 12, 600, "middle"))
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(text(px(t), plot_bottom + 22, f"{t:.2f}", "muted tnum", 11.5, anchor="middle"))
    out.append(text(px(0), plot_bottom + 40, "0 is chance agreement, 1 is perfect", "muted", 11.5))

    fy = h - 34
    out.append(text(32, fy, "Experts: 200 questions from four OpenStax textbooks, two trained raters, 1 to 4 scale. LLM: 100-question test split, 3 runs", "muted", 11.5))
    out.append(text(32, fy + 16, "per question; each criterion shows its best of 9 setups, so the bars are an upper estimate. Intervals are item-level bootstrap.", "muted", 11.5))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------- routing -----

def arrow(d: str) -> str:
    return f'<path class="arrow" d="{d}" marker-end="url(#ah)"/>'


def pill(x: float, cy: float, w: float, label: str, cls: str) -> list[str]:
    return [
        f'<rect class="{cls}" x="{x:.1f}" y="{cy - 16:.1f}" width="{w:.1f}" height="32" rx="16"/>',
        text(x + w / 2, cy + 4.5, label, "ink", 12.5, 600, "middle"),
    ]


def build_routing(rows: list[dict]) -> str:
    h = 500
    passing = [r["label"] for r in rows if r["llm"]["alpha"] >= GATE]
    failing = [r["label"] for r in rows if r["llm"]["alpha"] < GATE]
    out = open_svg(h, "How a score gets accepted",
                   "Each question is scored three times by the LLM. For criteria above the gate, unanimous runs are kept, "
                   "a 2 of 3 majority is kept and flagged for a spot check, and a split goes to blind expert review. "
                   "Criteria below the gate always go to blind expert review.")
    out.append(text(32, 86, "Each question is scored three times. The criterion decides what happens to those scores.", "ink2"))

    cy_mid = 258
    out.append(f'<rect class="box" x="32" y="{cy_mid - 34}" width="112" height="68" rx="10"/>')
    out.append(text(88, cy_mid - 6, "Question", "ink", 14, 600, "middle"))
    out.append(text(88, cy_mid + 14, "and its options", "ink2", 12, anchor="middle"))
    out.append(arrow(f"M144,{cy_mid} H166"))

    out.append(f'<rect class="box" x="170" y="{cy_mid - 52}" width="146" height="104" rx="10"/>')
    out.append(text(243, cy_mid - 24, "LLM scores it", "ink", 14, 600, "middle"))
    out.append(text(243, cy_mid - 6, "3 separate times", "ink2", 12, anchor="middle"))
    for j, s in enumerate(("3", "3", "2")):
        x0 = 203 + j * 28
        out.append(f'<rect class="llmT" x="{x0}" y="{cy_mid + 10}" width="22" height="24" rx="6"/>')
        out.append(text(x0 + 11, cy_mid + 27, s, "ink tnum", 12.5, 600, "middle"))

    top_cy, bot_cy = 172, 368
    out.append(arrow(f"M316,{cy_mid} H330 V{top_cy} H344"))
    out.append(arrow(f"M330,{cy_mid} V{bot_cy} H344"))

    out.append(f'<rect class="box" x="348" y="{top_cy - 54}" width="168" height="108" rx="10"/>')
    out.append(text(364, top_cy - 30, "Above the gate", "muted", 11.5))
    for k, name in enumerate(passing):
        out.append(text(364, top_cy - 8 + k * 20, name, "ink", 13.5, 600))
    out.append(text(364, top_cy + 40, "agreement check", "ink2", 12))

    out.append(f'<rect class="box" x="348" y="{bot_cy - 54}" width="168" height="108" rx="10"/>')
    out.append(text(364, bot_cy - 30, "Below the gate", "muted", 11.5))
    for k, name in enumerate(failing):
        out.append(text(364, bot_cy - 8 + k * 20, name, "ink", 13.5, 600))

    outcomes = [
        (top_cy - 46, "3 of 3 agree", "Score kept", "llmT"),
        (top_cy, "2 of 3 agree", "Majority kept, spot check", "llmF"),
        (top_cy + 46, "All 3 differ", "Blind expert review", "expT"),
    ]
    for cy, cond, label, cls in outcomes:
        out.append(arrow(f"M516,{top_cy} H528 V{cy} H540"))
        out.append(text(548, cy + 4.5, cond, "ink2", 12.5))
        out.extend(pill(640, cy, 188, label, cls))
    out.append(arrow(f"M516,{bot_cy} H632"))
    out.append(text(548, bot_cy - 12, "Always", "ink2", 12.5))
    out.extend(pill(640, bot_cy, 188, "Blind expert review", "expT"))
    out.append(text(640, bot_cy + 34, "LLM score is logged, never shown", "muted", 11.5))

    out.append(text(32, h - 50, "Experts never see the LLM score before entering their own, so it cannot anchor their rating. Every final label records", "muted", 11.5))
    out.append(text(32, h - 34, "where it came from, the three runs and the routing reason. A criterion changes lanes only when its measured agreement clears 0.50.", "muted", 11.5))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------ workload -----

def build_workload(wl: dict) -> str:
    h = 410
    replay = REPLAY_AUTO_PCT
    out = open_svg(h, "What the routing saves",
                   f"Replaying real scores through the routing hands {replay}% of 1,000 decisions to the LLM. "
                   f"Measured on the test split with the best setups: {wl['verified_pct']:.1f}%. "
                   f"Ceiling with 2 of 5 criteria automated: {wl['ceiling_pct']:.0f}%.")
    out.append(text(32, 86, "200 questions × 5 criteria = 1,000 scoring decisions that would otherwise all go to experts.", "ink2"))

    x0, x1 = 32.0, 828.0
    span = x1 - x0
    by, bh = 142.0, 34.0
    split = x0 + span * replay / 100
    out.append(f'<path class="llm" d="M{x0 + 6},{by} H{split - 1} V{by + bh} H{x0 + 6} Q{x0},{by + bh} {x0},{by + bh - 6} V{by + 6} Q{x0},{by} {x0 + 6},{by} Z"/>')
    out.append(f'<path class="exp" d="M{split + 1},{by} H{x1 - 6} Q{x1},{by} {x1},{by + 6} V{by + bh - 6} Q{x1},{by + bh} {x1 - 6},{by + bh} H{split + 1} Z"/>')
    out.append(text(x0, by - 12, f"{replay:.1f}%  LLM label kept", "ink", 13, 600))
    out.append(text(x1, by - 12, f"{100 - replay:.1f}%  expert decides", "ink", 13, 600, "end"))

    tv = x0 + span * wl["verified_pct"] / 100
    tc = x0 + span * wl["ceiling_pct"] / 100
    out.append(f'<line class="tick" x1="{tv:.1f}" y1="{by + bh + 4}" x2="{tv:.1f}" y2="{by + bh + 18}"/>')
    out.append(f'<line class="tick" x1="{tc:.1f}" y1="{by + bh + 4}" x2="{tc:.1f}" y2="{by + bh + 32}"/>')
    out.append(text(tv - 6, by + bh + 17, f"{wl['verified_pct']:.0f}% on the test split", "ink2", 12, anchor="end"))
    out.append(text(tc + 6, by + bh + 31, f"{wl['ceiling_pct']:.0f}% ceiling: 2 of 5 criteria fully automated", "ink2", 12))

    tiles = [
        (f"{replay:.0f}%", ["fewer expert decisions when real", "scores are replayed through the routing"]),
        ("2 of 5", ["criteria go to the LLM;", "the other three stay with experts"]),
        (f"under ${math.ceil(wl['cost_max'][0])}",
         [f"per 100 questions, 3 runs each", f"(${wl['cost_min'][0]:.2f} to ${wl['cost_max'][0]:.2f} across logged runs)"]),
    ]
    tw, gap, ty = 252.0, 20.0, 232.0
    for i, (big, lines) in enumerate(tiles):
        tx = 32 + i * (tw + gap)
        out.append(f'<rect class="box" x="{tx}" y="{ty}" width="{tw}" height="112" rx="10"/>')
        out.append(text(tx + 18, ty + 46, big, "ink", 32, 600))
        for k, ln in enumerate(lines):
            out.append(text(tx + 18, ty + 72 + k * 17, ln, "ink2", 12.5))

    out.append(text(32, h - 34, f"Kept labels include 2 of 3 majorities, which are flagged for a spot check. Counting unanimous runs only gives "
                    f"{wl['unanimous_pct']:.1f}% on the test split.", "muted", 11.5))
    out.append(text(32, h - 18, f"Cost from {wl['n_cost_cells']} logged experiment cells; cheapest {MODEL_NAMES.get(wl['cost_min'][1].split('_P')[0], wl['cost_min'][1])}, "
                    f"most expensive {MODEL_NAMES.get(wl['cost_max'][1].split('_P')[0], wl['cost_max'][1])}.", "muted", 11.5))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------- roadmap -----

def build_roadmap() -> str:
    col_w, gap, top = 252.0, 20.0, 112.0
    wrapped = {k: [textwrap.wrap(item, 34) for item in v] for k, v in ROADMAP.items()}
    col_h = max(sum(len(lines) * 17 + 14 for lines in items) for items in wrapped.values()) + 70
    h = int(top + col_h + 56)
    out = open_svg(h, "Where the project is", "Research status. " + " ".join(
        f"{k}: " + "; ".join(v) + "." for k, v in ROADMAP.items()))
    out.append(text(32, 86, "Two criteria are automated behind a check. The open work is on the three that are not.", "ink2"))

    markers = {"Done": "full", "In progress": "half", "Next": "ring"}
    for i, (head, items) in enumerate(wrapped.items()):
        cx = 32 + i * (col_w + gap)
        out.append(f'<rect class="box" x="{cx}" y="{top}" width="{col_w}" height="{col_h}" rx="10"/>')
        out.append(text(cx + 18, top + 32, head, "ink", 15, 600))
        out.append(text(cx + col_w - 18, top + 32, str(len(items)), "muted tnum", 13, anchor="end"))
        y = top + 62
        for lines in items:
            mx, my = cx + 24, y - 4.5
            kind = markers[head]
            if kind == "full":
                out.append(f'<circle class="dotFull" cx="{mx}" cy="{my}" r="4.5"/>')
            elif kind == "half":
                out.append(f'<circle class="dotRing" cx="{mx}" cy="{my}" r="4"/>')
                out.append(f'<path class="dotFull" d="M{mx},{my - 4} A4,4 0 0 0 {mx},{my + 4} Z"/>')
            else:
                out.append(f'<circle class="dotRing" cx="{mx}" cy="{my}" r="4"/>')
            for k, ln in enumerate(lines):
                out.append(text(cx + 38, y + k * 17, ln, "ink2", 12.5))
            y += len(lines) * 17 + 14

    out.append(text(32, h - 26, "Filled: done. Half: in progress. Ring: next. Details in docs/RESEARCH_STATUS.md.", "muted", 11.5))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    rows = load_reliability()
    wl = load_workload(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "reliability.svg": build_reliability(rows),
        "routing.svg": build_routing(rows),
        "workload.svg": build_workload(wl),
        "roadmap.svg": build_roadmap(),
    }
    for name, svg in figures.items():
        (OUT_DIR / name).write_text(svg, encoding="utf-8")
        print(f"wrote docs/images/{name}")
    for r in rows:
        print(f'  {r["label"]:<22} experts {r["human"]["alpha"]:.3f}  LLM {r["llm"]["alpha"]:+.3f}  ({r["llm"]["setup"]})')
    print(f"  auto-accept on test split {wl['verified_pct']:.1f}%, unanimous only {wl['unanimous_pct']:.1f}%, "
          f"ceiling {wl['ceiling_pct']:.0f}%")
    print(f"  cost per 100 questions: ${wl['cost_min'][0]:.2f} ({wl['cost_min'][1]}) to ${wl['cost_max'][0]:.2f} ({wl['cost_max'][1]})")


if __name__ == "__main__":
    main()
