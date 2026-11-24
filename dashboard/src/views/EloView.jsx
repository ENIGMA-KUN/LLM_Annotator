import React, { useState } from "react";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { api, ALL_DIMENSIONS, DIM_LABELS } from "@/lib/api";
import { Swords, RefreshCw, AlertTriangle } from "lucide-react";

const BUDGETS = [50, 150, 300, 600];

// Barrier-breaking framework #3: pairwise comparison + Elo for subjective dims.
export default function EloView() {
  const [dimension, setDimension] = useState("Semantic_Plausibility");
  const [budget, setBudget] = useState(150);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setBusy(true); setError(null);
    try {
      setResult(await api.eloRun(dimension, budget));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const maxHist = result
    ? Math.max(1, ...Object.values(result.ordinal_histogram))
    : 1;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Swords className="h-4 w-4" /> Pairwise Elo: our attack on the Tier C wall
          </CardTitle>
          <CardDescription>
            Absolute 1–4 scoring fails on subjective dimensions (central-tendency
            bias: the judge compresses everything to 3). Pairwise comparison
            (“which of these two items is better on this dimension?”) is a strictly
            easier judgment. Wins/losses update Elo ratings (FastChat / Chatbot
            Arena protocol, position-randomized), and final ratings map back to the
            1–4 scale by quartile so α against the human baseline stays computable.
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
            <div className="w-44">
              <Select value={String(budget)} onValueChange={(v) => setBudget(Number(v))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {BUDGETS.map((b) => (
                    <SelectItem key={b} value={String(b)}>{b} comparisons</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={run} disabled={busy}>
              {busy ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Swords className="h-4 w-4" />}
              Run tournament
            </Button>
          </div>

          <div className="rounded-md border border-indigo-200 bg-indigo-50 p-3 text-xs text-indigo-700">
            The judge here is <strong>simulated</strong> (seeded noise around the
            cached LLM medians) so the framework runs without API spend. Plugging in
            a live LLM judge is one line: <code className="font-mono">PairwiseJudge(engine)</code>.
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          )}
        </CardContent>
      </Card>

      {result && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Tournament summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{result.n_items} items</Badge>
                <Badge variant="secondary">{result.n_comparisons} comparisons</Badge>
                <Badge variant="indigo">{DIM_LABELS[result.dimension]}</Badge>
                <Badge variant="amber">{result.judge}</Badge>
              </div>
              <div>
                <p className="mb-2 text-sm font-semibold text-gray-700">
                  Elo → ordinal score distribution (by quartile)
                </p>
                {Object.entries(result.ordinal_histogram).map(([score, n]) => (
                  <div key={score} className="mb-1 flex items-center gap-2">
                    <span className="w-8 text-right font-mono text-xs">{score}</span>
                    <div className="h-4 flex-1 rounded bg-gray-100">
                      <div
                        className="h-4 rounded bg-indigo-500"
                        style={{ width: `${(n / maxHist) * 100}%` }}
                      />
                    </div>
                    <span className="w-10 font-mono text-xs text-gray-500">{n}</span>
                  </div>
                ))}
                <p className="mt-2 text-xs text-gray-500">
                  A spread across 1–4 (rather than a spike at 3) is exactly what
                  absolute scoring fails to produce on subjective dimensions.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Leaderboard (top 10 by Elo)</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-gray-500">
                  <tr>
                    <th className="py-2 pr-3">#</th>
                    <th className="py-2 pr-3">Item</th>
                    <th className="py-2 pr-3">Elo</th>
                    <th className="py-2">Mapped score</th>
                  </tr>
                </thead>
                <tbody>
                  {result.leaderboard.map((r, idx) => (
                    <tr key={r.question_uuid} className="border-t">
                      <td className="py-1.5 pr-3 text-gray-400">{idx + 1}</td>
                      <td className="py-1.5 pr-3 font-mono text-xs">{r.question_uuid.slice(0, 8)}</td>
                      <td className="py-1.5 pr-3 font-mono">{r.elo}</td>
                      <td className="py-1.5">
                        <Badge variant="emerald">{r.ordinal}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
