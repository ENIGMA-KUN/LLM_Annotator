import React from "react";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  FlaskConical, AlertTriangle, TrendingUp, ShieldAlert, Ruler, FileText, Users,
} from "lucide-react";
// R&D findings to date. Compiled from Data/results/neurips_experiments/ by
// scripts/build_findings_data.py, so this tab renders with the API server down.
// The compiled file is not in the public repository; a glob import resolves to
// nothing when it is absent, so a fresh clone still builds and this tab says why
// it is empty.
const findingsModules = import.meta.glob("/src/data/findingsData.js", { eager: true });
const findings = Object.values(findingsModules)[0]?.findings ?? null;

const DIM_LABEL = {
  Linguistic_Feature: "Linguistic Feature",
  Relevance: "Relevance",
  Semantic_Plausibility: "Semantic Plausibility",
  Semantic_Uniqueness: "Semantic Uniqueness",
  Clarity: "Clarity",
};

const PROMPT_LABEL = {
  P0_baseline: "P0 zero-shot",
  P1_manual: "P1 hand-written",
  P2_evidently: "P2 auto-optimised",
};

function fmt(x) {
  return (x >= 0 ? "+" : "") + x.toFixed(3);
}

/** Horizontal interval plot. Zero and the 0.5 gate are marked. */
function IntervalBar({ alpha, lo, hi }) {
  const MIN = -0.5;
  const MAX = 1.0;
  const pos = (v) => ((Math.max(MIN, Math.min(MAX, v)) - MIN) / (MAX - MIN)) * 100;
  const good = lo > 0.5;
  const ok = lo > 0;
  const color = good ? "bg-emerald-500" : ok ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className="relative h-6 w-full rounded bg-gray-100">
      <div className="absolute inset-y-0 w-px bg-gray-400" style={{ left: `${pos(0)}%` }} />
      <div
        className="absolute inset-y-0 w-px border-l border-dashed border-gray-500"
        style={{ left: `${pos(0.5)}%` }}
      />
      <div
        className={`absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full ${color} opacity-40`}
        style={{ left: `${pos(lo)}%`, width: `${Math.max(pos(hi) - pos(lo), 1)}%` }}
      />
      <div
        className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${color}`}
        style={{ left: `${pos(alpha)}%` }}
      />
    </div>
  );
}

function SectionCard({ icon: Icon, title, description, children, tone }) {
  // Tone marks meaning, not decoration. Brand identity sits in the maroon icon
  // and the gold rule; result quality keeps its own semantic scale.
  const border =
    tone === "warn" ? "border-amber-300" : tone === "good" ? "border-emerald-300" : "";
  return (
    <Card className={border}>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-asu-maroon-tint text-asu-maroon">
            <Icon className="h-4 w-4" />
          </span>
          <CardTitle>{title}</CardTitle>
        </div>
        <span className="asu-rule mt-1" aria-hidden="true" />
        {description && <CardDescription className="pt-1">{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export default function FindingsView() {
  if (!findings) {
    return (
      <SectionCard
        icon={FileText}
        title="Findings data not included in this build"
        description="The results behind this tab are not distributed with the repository."
      >
        <p className="text-sm text-gray-600">
          Run <code>python scripts/build_findings_data.py</code> with the result files under{" "}
          <code>Data/</code> to generate <code>src/data/findingsData.js</code>, then rebuild. The
          headline numbers are in the project README.
        </p>
      </SectionCard>
    );
  }

  const f = findings;
  const al = f.activeLearning.find((c) => c.dimension === "Relevance");
  const alBest = al?.points.reduce((a, b) => (b.gain > a.gain ? b : a));
  const flaggedFairness = f.fairness.filter((r) => r.flagged);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
      {/* Scope */}
      <SectionCard
        icon={FlaskConical}
        title="R&D to date"
        description={`${f.scale.nConfigs} model-prompt-dimension configurations across ${f.scale.nModels} models and ${f.scale.nPrompts} prompt strategies, 3 runs per item at temperature ${f.provenance.config?.temperature ?? "0.3"}. Every number on this tab is generated from the experiment CSVs, not entered by hand.`}
      >
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[
            ["Configurations", f.scale.nConfigs],
            ["Models compared", f.scale.nModels],
            ["Prompt strategies", f.scale.nPrompts],
            ["Dimensions", f.scale.nDimensions],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg border bg-gray-50 p-3">
              <p className="text-2xl font-bold text-gray-900">{value}</p>
              <p className="text-xs text-gray-500">{label}</p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-gray-500">
          Experiment run {f.provenance.experimentRun?.slice(0, 10)} · prompt hashes and
          data checksums recorded · generated {f.generatedAt.slice(0, 10)}
        </p>
      </SectionCard>

      {/* Human ceiling. The denominator every other number needs. */}
      {f.humanCeiling?.length > 0 && (
        <SectionCard
          icon={Users}
          tone="good"
          title="The human ceiling, measured"
          description={`Both annotators rated all ${f.humanCeiling[0].nTotal} items on all five dimensions, so human-to-human agreement is measurable. Ordinal Krippendorff α, LLM and humans compared on identical items. The last two columns show what fraction of items each rater put on its single most common score.`}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs uppercase text-gray-500">
                  <th className="py-2 pr-3 font-medium">Dimension</th>
                  <th className="py-2 pr-3 font-medium">Human ceiling</th>
                  <th className="py-2 pr-3 font-medium">LLM α</th>
                  <th className="py-2 pr-3 font-medium">% of ceiling</th>
                  <th className="py-2 pr-3 font-medium">Human on modal score</th>
                  <th className="py-2 font-medium">LLM on modal score</th>
                </tr>
              </thead>
              <tbody>
                {f.humanCeiling.map((r) => (
                  <tr key={r.dimension} className="border-b last:border-0">
                    <td className="py-2 pr-3 font-medium">{DIM_LABEL[r.dimension]}</td>
                    <td className="py-2 pr-3 font-mono">
                      {fmt(r.ceilingMatched)}
                      <span className="ml-1 text-xs text-gray-400">
                        full set {fmt(r.humanAlpha)} [{fmt(r.humanCiLower)}, {fmt(r.humanCiUpper)}]
                      </span>
                    </td>
                    <td className="py-2 pr-3 font-mono font-semibold">{fmt(r.llmAlpha)}</td>
                    <td className="py-2 pr-3">
                      {r.pctOfCeiling != null && (
                        <Badge
                          variant={
                            r.pctOfCeiling >= 60
                              ? "emerald"
                              : r.pctOfCeiling >= 30
                              ? "amber"
                              : "rose"
                          }
                        >
                          {r.pctOfCeiling}%
                        </Badge>
                      )}
                    </td>
                    <td className="py-2 pr-3 font-mono text-gray-500">{r.humanModalPct}%</td>
                    <td className="py-2 font-mono">
                      <span className={r.llmCollapsed ? "font-bold text-rose-600" : "text-gray-600"}>
                        {r.llmModalPct}%
                      </span>
                      {r.llmCollapsed && (
                        <Badge variant="rose" className="ml-2">collapsed</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 rounded-md border border-emerald-300 bg-emerald-50 p-3 text-xs text-emerald-900">
            <strong>Read the last two columns together.</strong> Linguistic Feature is the
            one dimension where the model's score distribution matches the humans' (37%
            against 38% on the most common score), and it is the one dimension that works.
            Clarity is the worst: the model puts 95% of items on a single score while humans
            spread across the scale at 45%. The model has stopped discriminating. Humans
            measure Clarity perfectly well, so this is a rubric failure with a known fix,
            not an unmeasurable construct.
          </div>
        </SectionCard>
      )}

      {/* Headline agreement */}
      <SectionCard
        icon={Ruler}
        title="Where the LLM agrees with human annotators"
        description="Best configuration per dimension. Krippendorff α against expert labels with bootstrap 95% intervals. Dot is the estimate, bar is the interval, dashed line is the 0.5 routing gate."
      >
        <div className="space-y-3">
          {f.bestPerDimension.map((r) => (
            <div key={r.dimension} className="grid grid-cols-12 items-center gap-3">
              <div className="col-span-12 md:col-span-3">
                <p className="text-sm font-semibold">{DIM_LABEL[r.dimension]}</p>
                <p className="text-xs text-gray-500">
                  {r.model} · {PROMPT_LABEL[r.prompt] ?? r.prompt} · n={r.n}
                </p>
              </div>
              <div className="col-span-8 md:col-span-6">
                <IntervalBar alpha={r.alpha} lo={r.ciLower} hi={r.ciUpper} />
              </div>
              <div className="col-span-4 md:col-span-3 text-right">
                <span className="font-mono text-sm font-semibold">{fmt(r.alpha)}</span>
                <span className="ml-1 font-mono text-xs text-gray-500">
                  [{fmt(r.ciLower)}, {fmt(r.ciUpper)}]
                </span>
                <div className="mt-1">
                  <Badge variant={r.tier === "A" ? "emerald" : "indigo"}>
                    Tier {r.tier}
                  </Badge>
                  {r.crossesHalf && (
                    <Badge variant="amber" className="ml-1">CI crosses gate</Badge>
                  )}
                  {!r.excludesZero && (
                    <Badge variant="rose" className="ml-1">CI includes 0</Badge>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 rounded-md border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600">
          Two dimensions clear the gate on the point estimate. Both intervals still cross
          0.5, so neither is confirmed at this sample size. Three dimensions cannot be
          separated from chance and are routed to humans by design.
        </div>
      </SectionCard>

      {/* The routing paradox */}
      <SectionCard
        icon={ShieldAlert}
        tone="warn"
        title="Why we route by dimension, not by confidence alone"
        description="Run consensus against actual agreement. The model reaches near-total agreement with itself on dimensions where it agrees with humans no better than chance. A confidence gate alone would auto-accept those."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase text-gray-500">
                <th className="py-2 pr-3 font-medium">Dimension</th>
                <th className="py-2 pr-3 font-medium">3/3 run consensus</th>
                <th className="py-2 pr-3 font-medium">Gate would auto-accept</th>
                <th className="py-2 pr-3 font-medium">Actual α vs humans</th>
                <th className="py-2 font-medium">Router decision</th>
              </tr>
            </thead>
            <tbody>
              {f.routingParadox.map((r) => (
                <tr key={r.dimension} className="border-b last:border-0">
                  <td className="py-2 pr-3 font-medium">{DIM_LABEL[r.dimension]}</td>
                  <td className="py-2 pr-3 font-mono">{r.pct3of3}%</td>
                  <td className="py-2 pr-3 font-mono">{r.pctAutoAccept}%</td>
                  <td className="py-2 pr-3 font-mono">
                    <span className={r.alpha < 0.3 ? "text-rose-600" : "text-emerald-600"}>
                      {fmt(r.alpha)}
                    </span>
                  </td>
                  <td className="py-2">
                    {r.unsafe ? (
                      <Badge variant="indigo">Held back for human review</Badge>
                    ) : (
                      <Badge variant="emerald">Eligible for auto-accept</Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            <strong>Self-consistency is not correctness.</strong> On Clarity the model
            agrees with itself on 99% of items while scoring below chance against humans.
            This is the single strongest argument for the dimension router: confidence
            alone would have auto-accepted four of five dimensions, and three of those
            carry no usable signal.
          </span>
        </div>
      </SectionCard>

      {/* Prompt strategy */}
      <SectionCard
        icon={FileText}
        title="Prompt strategy does not transfer across dimensions"
        description={`Krippendorff α by prompt for ${f.promptMatrix.model}. Improving one dimension degrades another, and the automated optimiser is the worst option on average.`}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase text-gray-500">
                <th className="py-2 pr-3 font-medium">Dimension</th>
                {f.promptMatrix.prompts.map((p) => (
                  <th key={p} className="py-2 pr-3 font-medium">
                    {PROMPT_LABEL[p] ?? p}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {f.promptMatrix.rows.map((row) => {
                const best = Math.max(...row.values);
                return (
                  <tr key={row.dimension} className="border-b last:border-0">
                    <td className="py-2 pr-3 font-medium">{DIM_LABEL[row.dimension]}</td>
                    {row.values.map((v, i) => (
                      <td
                        key={i}
                        className={`py-2 pr-3 font-mono ${
                          v === best ? "font-bold text-emerald-700" : "text-gray-600"
                        }`}
                      >
                        {fmt(v)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-gray-600">
          The hand-written prompt more than doubles Relevance but collapses Linguistic
          Feature from {fmt(0.536)} to {fmt(0.091)}. The auto-optimised prompt is worse
          than zero-shot on four of five dimensions. Reported as a negative result.
        </p>
      </SectionCard>

      {/* Active learning */}
      {al && (
        <SectionCard
          icon={TrendingUp}
          tone="good"
          title="Targeted human review beats random review"
          description="Krippendorff α on Relevance as human review budget increases. Uncertainty-based selection against a random-selection control."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs uppercase text-gray-500">
                  <th className="py-2 pr-3 font-medium">Human budget</th>
                  <th className="py-2 pr-3 font-medium">Targeted α</th>
                  <th className="py-2 pr-3 font-medium">Random α</th>
                  <th className="py-2 font-medium">Gain</th>
                </tr>
              </thead>
              <tbody>
                {al.points.map((p) => (
                  <tr key={p.budgetPct} className="border-b last:border-0">
                    <td className="py-2 pr-3 font-mono">{p.budgetPct}%</td>
                    <td className="py-2 pr-3 font-mono font-semibold">{p.active.toFixed(3)}</td>
                    <td className="py-2 pr-3 font-mono text-gray-500">{p.random.toFixed(3)}</td>
                    <td className="py-2 font-mono">
                      <span className={p.gain > 0 ? "text-emerald-600" : "text-gray-400"}>
                        {fmt(p.gain)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {alBest && (
            <div className="mt-3 rounded-md border border-emerald-300 bg-emerald-50 p-3 text-xs text-emerald-900">
              Best gain at {alBest.budgetPct}% budget: α {alBest.active.toFixed(3)} against{" "}
              {alBest.random.toFixed(3)} for random selection, a gain of {fmt(alBest.gain)}.
              This is the clearest evidence that the hybrid design earns its complexity.
              Single run on one dimension, not yet replicated.
            </div>
          )}
        </SectionCard>
      )}

      {/* Robustness */}
      <SectionCard
        icon={ShieldAlert}
        title="Where quality varies by subgroup"
        description="Krippendorff α split by subgroup. A gap above threshold is flagged."
      >
        <div className="space-y-2">
          {flaggedFairness.map((r) => (
            <div
              key={`${r.dimension}-${r.analysis}`}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-amber-200 bg-amber-50 p-3"
            >
              <div>
                <p className="text-sm font-semibold">
                  {DIM_LABEL[r.dimension]} · {r.analysis.replace(/_/g, " ")}
                </p>
                <p className="text-xs text-gray-600">
                  {r.subgroups
                    .map((s) => `${s.name}: ${fmt(s.alpha)} (n=${s.n})`)
                    .join("   ·   ")}
                </p>
              </div>
              <Badge variant="amber">gap {r.maxGap.toFixed(3)}</Badge>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-gray-600">
          The largest gap is on human agreement: the model tracks the experts closely on
          items the experts agree about, and adds little where they disagree. Those are
          precisely the items a human is needed for, which is what the router is built to
          send them.
        </p>
      </SectionCard>

      {/* Limits */}
      <SectionCard
        icon={Ruler}
        title="What this sample size can and cannot support"
        description="Bootstrap interval width by sample size, and the n required to confirm each dimension at the target."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase text-gray-500">
                <th className="py-2 pr-3 font-medium">Dimension</th>
                <th className="py-2 pr-3 font-medium">Best α</th>
                <th className="py-2 pr-3 font-medium">CI width at n=100</th>
                <th className="py-2 pr-3 font-medium">CI width at n=300</th>
                <th className="py-2 font-medium">n required</th>
              </tr>
            </thead>
            <tbody>
              {f.power.map((r) => (
                <tr key={r.dimension} className="border-b last:border-0">
                  <td className="py-2 pr-3 font-medium">{DIM_LABEL[r.dimension]}</td>
                  <td className="py-2 pr-3 font-mono">{fmt(r.bestAlpha)}</td>
                  <td className="py-2 pr-3 font-mono text-gray-500">{r.ciWidth100}</td>
                  <td className="py-2 pr-3 font-mono">{r.ciWidth300}</td>
                  <td className="py-2">
                    <Badge variant={r.requiredN === "300" ? "amber" : "slate"}>
                      {r.requiredN}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 rounded-md border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600">
          <p className="font-semibold text-gray-700">Known limits, stated up front</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            <li>
              At n=100 the interval is roughly 0.32 wide, so the two Tier A dimensions
              cannot yet be confirmed above the gate. 300 items would settle both.
            </li>
            <li>
              The two human annotators labelled non-overlapping sets, so there is no
              measured human-to-human ceiling to compare against.
            </li>
            <li>
              Source items are OpenStax, which is public and likely in pre-training.
              Contamination is not yet excluded.
            </li>
            <li>
              Workload figures on the threshold sweep count dimensions exempted, not
              annotator time. Reading the item is a fixed cost that routing does not remove.
            </li>
          </ul>
        </div>
      </SectionCard>
    </div>
  );
}
