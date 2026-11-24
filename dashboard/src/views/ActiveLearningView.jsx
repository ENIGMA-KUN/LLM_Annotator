import React, { useEffect, useState } from "react";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { api, ALL_DIMENSIONS, DIM_LABELS } from "@/lib/api";
import { Target, AlertTriangle } from "lucide-react";

// Horizontal bar pair for one budget level: active vs random alpha.
function CurveRow({ row }) {
  const pct = (a) => `${Math.max(0, Math.min(1, (a ?? 0))) * 100}%`;
  return (
    <div className="space-y-1 border-t py-2 first:border-t-0">
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>
          Budget {row.budget_pct}% ({row.n_human_reviewed} items to humans)
        </span>
        <span>
          gain{" "}
          <span className={row.alpha_gain > 0 ? "font-semibold text-emerald-600" : "text-gray-400"}>
            {row.alpha_gain != null ? (row.alpha_gain >= 0 ? "+" : "") + row.alpha_gain.toFixed(3) : "—"}
          </span>
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-14 text-xs font-medium text-indigo-700">active</span>
        <div className="h-3 flex-1 rounded bg-gray-100">
          <div className="h-3 rounded bg-indigo-500" style={{ width: pct(row.alpha_active) }} />
        </div>
        <span className="w-14 text-right font-mono text-xs">{row.alpha_active?.toFixed(3) ?? "—"}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-14 text-xs font-medium text-gray-500">random</span>
        <div className="h-3 flex-1 rounded bg-gray-100">
          <div className="h-3 rounded bg-gray-400" style={{ width: pct(row.alpha_random) }} />
        </div>
        <span className="w-14 text-right font-mono text-xs">{row.alpha_random?.toFixed(3) ?? "—"}</span>
      </div>
    </div>
  );
}

// Barrier-breaking framework #1: uncertainty-based active learning.
export default function ActiveLearningView() {
  const [dimension, setDimension] = useState("Linguistic_Feature");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setData(null); setError(null);
    api.activeLearning(dimension).then(setData).catch((e) => setError(e.message));
  }, [dimension]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Target className="h-4 w-4" /> Active Learning: uncertainty-based routing
          </CardTitle>
          <CardDescription>
            Inter-run disagreement ranks items by uncertainty; the most uncertain go to
            humans first. The curve compares agreement (Krippendorff α vs. expert
            consensus) when spending the same human budget on uncertain items
            (active) vs. random items.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="w-64">
              <Select value={dimension} onValueChange={setDimension}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {ALL_DIMENSIONS.map((d) => (
                    <SelectItem key={d} value={d}>{DIM_LABELS[d]}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {data?.curve_source && (
              <Badge variant={data.curve_source === "real_experiment" ? "emerald" : "secondary"}>
                {data.curve_source === "real_experiment"
                  ? "real experiment data"
                  : "computed from current scores"}
              </Badge>
            )}
            {data?.optimal_budget && (
              <Badge variant="indigo">
                α ≥ {data.optimal_budget.target_alpha} reached at{" "}
                {data.optimal_budget.min_budget_pct}% human budget
                (α = {data.optimal_budget.alpha_achieved?.toFixed(3)})
              </Badge>
            )}
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          )}
          {!data && !error && <p className="text-sm text-gray-400">Loading…</p>}
          {data && data.curve.length === 0 && (
            <p className="text-sm text-gray-500">
              No curve available for this dimension: no cached experiment file and
              no human consensus columns to compute one now.
            </p>
          )}
          {data && data.curve.length > 0 && (
            <div>{data.curve.map((row) => <CurveRow key={row.budget_pct} row={row} />)}</div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Most uncertain items (route these first)</CardTitle>
          <CardDescription>
            Live ranking from the current batch scores (or the cached experiment when
            no batch is annotated yet).
          </CardDescription>
        </CardHeader>
        <CardContent>
          {data && data.most_uncertain.length > 0 ? (
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-gray-500">
                <tr>
                  <th className="py-2 pr-3">Item</th>
                  <th className="py-2 pr-3">3 runs</th>
                  <th className="py-2 pr-3">Agreement</th>
                  <th className="py-2 pr-3">Median</th>
                  <th className="py-2">Uncertainty</th>
                </tr>
              </thead>
              <tbody>
                {data.most_uncertain.map((r) => (
                  <tr key={r.uuid} className="border-t">
                    <td className="py-1.5 pr-3 font-mono text-xs">{String(r.uuid).slice(0, 8)}</td>
                    <td className="py-1.5 pr-3 font-mono text-xs">[{(r.scores ?? []).join(", ")}]</td>
                    <td className="py-1.5 pr-3">
                      <Badge variant={r.agreement_level === "split" ? "rose" : r.agreement_level === "2of3" ? "amber" : "emerald"}>
                        {r.agreement_level}
                      </Badge>
                    </td>
                    <td className="py-1.5 pr-3">{r.median_score ?? "—"}</td>
                    <td className="py-1.5">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-32 rounded bg-gray-100">
                          <div className="h-2 rounded bg-rose-500" style={{ width: `${(r.uncertainty_score ?? 0) * 100}%` }} />
                        </div>
                        <span className="font-mono text-xs">{r.uncertainty_score?.toFixed(2)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-gray-400">No scores yet.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
