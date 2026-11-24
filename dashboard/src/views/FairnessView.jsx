import React, { useEffect, useState } from "react";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api, DIM_LABELS } from "@/lib/api";
import { Scale, AlertTriangle } from "lucide-react";

const ANALYSIS_LABELS = {
  by_subject: "Textbook subject",
  by_difficulty: "Item difficulty",
  by_human_agreement: "Human agreement",
  by_text_length: "Passage length",
};

// Barrier-breaking framework #2: subgroup fairness audit of LLM annotation quality.
export default function FairnessView() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.fairness().then(setData).catch((e) => setError(e.message));
  }, []);

  // group rows: dimension -> analysis -> [rows]
  const grouped = {};
  (data?.rows ?? []).forEach((r) => {
    grouped[r.dimension] = grouped[r.dimension] || {};
    grouped[r.dimension][r.analysis] = grouped[r.dimension][r.analysis] || [];
    grouped[r.dimension][r.analysis].push(r);
  });

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Scale className="h-4 w-4" /> Fairness Audit: does annotation quality vary by subgroup?
          </CardTitle>
          <CardDescription>
            Krippendorff α between the LLM median and expert consensus, split by
            textbook subject, item difficulty, human-agreement level, and passage
            length. A gap &gt; {data?.threshold ?? 0.15} between subgroups is flagged
            as a fairness concern. Results computed from the real experiment run.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          )}
          {!data && !error && <p className="text-sm text-gray-400">Loading…</p>}
          {data && (
            <p className="text-sm text-gray-600">
              {data.n_flags === 0
                ? "No subgroup gap exceeds the flag threshold in the audited dimensions."
                : `${data.n_flags} dimension–analysis pairs exceed the gap threshold and warrant attention.`}
            </p>
          )}
        </CardContent>
      </Card>

      {Object.entries(grouped).map(([dim, analyses]) => (
        <Card key={dim}>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">{DIM_LABELS[dim] ?? dim}</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            {Object.entries(analyses).map(([analysis, rows]) => {
              const gap = rows[0]?.max_gap ?? 0;
              const fair = rows.every((r) => r.fair !== false) && gap <= (data?.threshold ?? 0.15);
              return (
                <div key={analysis} className="rounded-lg border p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-gray-700">
                      {ANALYSIS_LABELS[analysis] ?? analysis}
                    </h4>
                    <Badge variant={fair ? "emerald" : "rose"}>
                      {fair ? "fair" : "flagged"} · gap {Number(gap).toFixed(3)}
                    </Badge>
                  </div>
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase text-gray-400">
                      <tr>
                        <th className="py-1 pr-2">Subgroup</th>
                        <th className="py-1 pr-2">n</th>
                        <th className="py-1 pr-2">α</th>
                        <th className="py-1">Bias</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.subgroup} className="border-t">
                          <td className="py-1 pr-2">{r.subgroup}</td>
                          <td className="py-1 pr-2 text-gray-500">{r.n}</td>
                          <td className="py-1 pr-2 font-mono">
                            {r.alpha == null ? "—" : Number(r.alpha).toFixed(3)}
                          </td>
                          <td className={`py-1 font-mono ${r.bias > 0 ? "text-rose-600" : "text-gray-600"}`}>
                            {r.bias == null ? "—" : (r.bias >= 0 ? "+" : "") + Number(r.bias).toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
