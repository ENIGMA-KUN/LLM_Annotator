import React, { useEffect, useRef, useState } from "react";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import OrdinalSelector from "@/components/dashboard/OrdinalSelector";
import { api, DIM_LABELS } from "@/lib/api";
import {
  Upload, PlayCircle, Users, PackageCheck, Download,
  CheckCircle2, AlertTriangle, RefreshCw, EyeOff, FlaskConical,
} from "lucide-react";

function StepBadge({ n, done, active }) {
  const cls = done
    ? "bg-emerald-600 text-white"
    : active
      ? "bg-gray-900 text-white"
      : "bg-gray-200 text-gray-500";
  return (
    <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm font-bold ${cls}`}>
      {done ? "✓" : n}
    </span>
  );
}

const STAT_TONES = {
  slate: "text-gray-700",
  emerald: "text-emerald-700",
  amber: "text-amber-700",
  indigo: "text-indigo-700",
  rose: "text-rose-700",
};

function Stat({ label, value, tone = "slate" }) {
  return (
    <div className="rounded-lg border bg-white p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-xl font-bold ${STAT_TONES[tone]}`}>{value}</p>
    </div>
  );
}

// Real pipeline UI backed by the FastAPI server (replaces the Streamlit console).
export default function PipelineView() {
  const [health, setHealth] = useState(null);
  const [batch, setBatch] = useState(null);        // upload result
  const [gateStats, setGateStats] = useState(null); // annotate result
  const [annotateMode, setAnnotateMode] = useState(null);
  const [queueRows, setQueueRows] = useState([]);
  const [activeReview, setActiveReview] = useState(null); // queue row being scored
  const [draftScore, setDraftScore] = useState(null);
  const [finalized, setFinalized] = useState(null);
  const [busy, setBusy] = useState(null);          // string label of running op
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ ok: false }));
    // Restore server-side session (e.g., after a page refresh)
    api.state().then((s) => {
      if (s.n_items > 0) setBatch({ n_items: s.n_items, source: "(restored session)" });
      if (s.gate_stats) { setGateStats(s.gate_stats); setAnnotateMode(s.annotate_mode); }
      if (s.gate_stats) refreshQueue();
    }).catch(() => {});
  }, []);

  const run = (label, fn) => async (...args) => {
    setBusy(label); setError(null);
    try { return await fn(...args); }
    catch (e) { setError(`${label} failed: ${e.message}`); }
    finally { setBusy(null); }
  };

  const refreshQueue = async () => {
    const q = await api.queue();
    setQueueRows(q.rows);
  };

  const handleUploadSample = run("Load sample", async () => {
    const res = await api.uploadSample();
    setBatch(res); setGateStats(null); setQueueRows([]); setFinalized(null);
  });

  const handleUploadFile = run("Upload", async (file) => {
    const res = await api.uploadFile(file);
    setBatch(res); setGateStats(null); setQueueRows([]); setFinalized(null);
  });

  const handleAnnotate = run("Annotate", async (mode) => {
    const res = await api.annotate(mode);
    setGateStats(res.gate_stats); setAnnotateMode(res.mode);
    setFinalized(null);
    await refreshQueue();
  });

  const handleSubmitReview = run("Submit review", async () => {
    await api.review({
      question_uuid: activeReview.question_uuid,
      dimension: activeReview.dimension,
      score: draftScore,
      annotator_id: "reviewer_1",
    });
    setActiveReview(null); setDraftScore(null);
    await refreshQueue();
  });

  const handleFinalize = run("Finalize", async (fullAuto) => {
    const res = await api.finalize(fullAuto);
    setFinalized(res);
  });

  const apiDown = health && !health.ok;
  const pending = queueRows.filter((r) => !r.reviewed);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
      {apiDown && (
        <div className="flex items-center gap-2 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
          <AlertTriangle className="h-4 w-4" />
          Pipeline API is not reachable. Start it with:&nbsp;
          <code className="rounded bg-rose-100 px-1 font-mono text-xs">
            uvicorn src.api.server:app --port 8000
          </code>
        </div>
      )}
      {error && (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          {error}
        </div>
      )}

      {/* ---- Step 1: Upload ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <StepBadge n={1} done={!!batch} active={!batch} />
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Upload className="h-4 w-4" /> Upload items
              </CardTitle>
              <CardDescription>
                CSV with columns: question_uuid, text, question, answer
                (optional: option_a–d, subject, source)
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-3">
          <input
            ref={fileRef} type="file" accept=".csv" className="hidden"
            onChange={(e) => e.target.files[0] && handleUploadFile(e.target.files[0])}
          />
          <Button variant="outline" disabled={!!busy}
            onClick={() => fileRef.current?.click()}>
            Choose CSV…
          </Button>
          <Button disabled={!!busy || !health?.sample_available}
            onClick={handleUploadSample}>
            Load benchmark sample (200 items)
          </Button>
          {batch && (
            <Badge variant="emerald">
              {batch.n_items} items loaded {batch.source ? `· ${batch.source}` : ""}
            </Badge>
          )}
        </CardContent>
      </Card>

      {/* ---- Step 2: Annotate ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <StepBadge n={2} done={!!gateStats} active={!!batch && !gateStats} />
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <PlayCircle className="h-4 w-4" /> Annotate (3 runs per item) &amp; gate
              </CardTitle>
              <CardDescription>
                Replay mode reuses the cached Gemini Flash 3-run scores from the real
                experiment (no API cost). Live mode calls the LLM now.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <Button disabled={!batch || !!busy} onClick={() => handleAnnotate("replay")}>
              {busy === "Annotate" ? <RefreshCw className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
              Run with cached experiment scores
            </Button>
            <Button variant="outline" disabled={!batch || !!busy || !health?.live_key_present}
              onClick={() => handleAnnotate("live")}
              title={health?.live_key_present ? "" : "Set GEMINI_API_KEY to enable live calls"}>
              Run live (requires API key)
            </Button>
            {annotateMode && (
              <Badge variant="secondary">mode: {annotateMode}</Badge>
            )}
          </div>
          {gateStats && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="Total decisions" value={gateStats.total_decisions} />
              <Stat label="Auto-accepted (3/3 + 2/3)" value={gateStats.total_auto_accepted} tone="emerald" />
              <Stat label="Workload reduction" value={`${gateStats.workload_reduction_pct}%`} tone="emerald" />
              <Stat label="Routed to human" value={gateStats.total_decisions - gateStats.total_auto_accepted} tone="amber" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* ---- Step 3: Human review queue ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <StepBadge n={3} done={gateStats && pending.length === 0 && queueRows.length > 0}
              active={!!gateStats && pending.length > 0} />
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Users className="h-4 w-4" /> Human review queue
                {gateStats && (
                  <Badge variant={pending.length ? "amber" : "emerald"}>
                    {pending.length} pending
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                Tier C reviews are blind: LLM runs are withheld. Tier A escalations
                show the disagreeing runs.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        {gateStats && (
          <CardContent className="space-y-3">
            <div className="max-h-72 overflow-y-auto rounded-md border">
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 bg-gray-50 text-xs uppercase text-gray-500">
                  <tr>
                    <th className="px-3 py-2">Item</th>
                    <th className="px-3 py-2">Dimension</th>
                    <th className="px-3 py-2">Tier</th>
                    <th className="px-3 py-2">Reason</th>
                    <th className="px-3 py-2">LLM runs</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {queueRows.slice(0, 200).map((r) => (
                    <tr key={`${r.question_uuid}-${r.dimension}`}
                      className={r.reviewed ? "bg-emerald-50/50 text-gray-400" : "border-t"}>
                      <td className="px-3 py-1.5 font-mono text-xs">{r.question_uuid.slice(0, 8)}</td>
                      <td className="px-3 py-1.5">{DIM_LABELS[r.dimension] ?? r.dimension}</td>
                      <td className="px-3 py-1.5">
                        <Badge variant={r.tier === "C" ? "indigo" : "amber"}>{r.tier}</Badge>
                      </td>
                      <td className="px-3 py-1.5 text-xs">{r.queue_reason}</td>
                      <td className="px-3 py-1.5 font-mono text-xs">
                        {r.tier === "C"
                          ? <span className="flex items-center gap-1 text-gray-400"><EyeOff className="h-3 w-3" /> blind</span>
                          : (r.llm_runs_visible ?? "—")}
                      </td>
                      <td className="px-3 py-1.5">
                        {r.reviewed
                          ? <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                          : <Button size="sm" variant="outline"
                              onClick={() => { setActiveReview(r); setDraftScore(null); }}>
                              Score
                            </Button>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {activeReview && (
              <div className="space-y-3 rounded-lg border border-indigo-200 bg-indigo-50/50 p-4">
                <p className="text-sm font-semibold text-gray-800">
                  {DIM_LABELS[activeReview.dimension] ?? activeReview.dimension}
                  <span className="ml-2 font-mono text-xs text-gray-500">
                    {activeReview.question_uuid.slice(0, 8)}
                  </span>
                </p>
                {activeReview.question && (
                  <p className="text-sm text-gray-600">{activeReview.question}</p>
                )}
                <OrdinalSelector value={draftScore} onChange={setDraftScore} accent="indigo" />
                <div className="flex gap-2">
                  <Button disabled={draftScore == null || !!busy} onClick={handleSubmitReview}>
                    Submit review
                  </Button>
                  <Button variant="outline" onClick={() => setActiveReview(null)}>Cancel</Button>
                </div>
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* ---- Step 4: Finalize & export ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <StepBadge n={4} done={!!finalized} active={!!gateStats} />
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <PackageCheck className="h-4 w-4" /> Finalize &amp; export
              </CardTitle>
              <CardDescription>
                Hybrid keeps unreviewed decisions pending; Full Automation forces
                3-run medians flagged auto_lowconf for audit.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        {gateStats && (
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-3">
              <Button disabled={!!busy} onClick={() => handleFinalize(false)}>
                Finalize (hybrid)
              </Button>
              <Button variant="destructive" disabled={!!busy} onClick={() => handleFinalize(true)}>
                Finalize (full automation, LLM only)
              </Button>
              {finalized && (
                <a href={api.exportUrl} download>
                  <Button variant="emerald">
                    <Download className="h-4 w-4" /> Download CSV
                  </Button>
                </a>
              )}
            </div>
            {finalized && (
              <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
                <Stat label="Auto-accepted" value={`${finalized.workload_report.pct_auto_accepted}%`} tone="emerald" />
                <Stat label="Auto-flagged (2/3)" value={`${finalized.workload_report.pct_auto_flagged}%`} tone="emerald" />
                <Stat label="Low-conf forced" value={`${finalized.workload_report.pct_auto_lowconf ?? 0}%`} tone="rose" />
                <Stat label="Human-reviewed" value={`${finalized.workload_report.pct_routed_human}%`} tone="indigo" />
                <Stat label="Still pending" value={`${finalized.workload_report.pct_pending}%`} tone="amber" />
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  );
}
