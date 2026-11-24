import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import RunPills from "./RunPills";
import OrdinalSelector from "./OrdinalSelector";
import StatusBadge from "./StatusBadge";
import {
  TIER_A_METRICS,
  TIER_C_METRICS,
  GATE,
  gateDecision,
  consensusScore,
  medianScore,
  deriveStatus,
} from "@/data/mockData";
import {
  Bot,
  User,
  CheckCircle2,
  Send,
  Zap,
  AlertTriangle,
} from "lucide-react";

const OPTION_KEYS = ["A", "B", "C", "D"];

export default function ItemWorkspace({ item, onSubmit }) {
  // Local draft of scores being edited
  const [draft, setDraft] = useState({});
  const [confirmOpen, setConfirmOpen] = useState(false);

  // Re-initialize draft when the selected item changes:
  // pre-fill Tier A consensus values and any previously saved scores.
  useEffect(() => {
    if (!item) return;
    const initial = {};
    TIER_A_METRICS.forEach(({ key }) => {
      const runs = item.tierARuns[key];
      const consensus = consensusScore(runs);
      initial[key] =
        item.acceptedScores[key] ?? (consensus !== null ? consensus : null);
    });
    TIER_C_METRICS.forEach(({ key }) => {
      initial[key] = item.humanScores[key] ?? null;
    });
    setDraft(initial);
  }, [item?.id]);

  if (!item) {
    return (
      <Card className="flex h-full items-center justify-center">
        <p className="text-sm text-gray-400">
          Select an item from the queue to begin reviewing.
        </p>
      </Card>
    );
  }

  const status = deriveStatus(item);
  const allScored = [...TIER_A_METRICS, ...TIER_C_METRICS].every(
    ({ key }) => draft[key] != null
  );
  const missing = [...TIER_A_METRICS, ...TIER_C_METRICS]
    .filter(({ key }) => draft[key] == null)
    .map(({ label }) => label);

  const handleSubmit = () => {
    if (!allScored) {
      setConfirmOpen(true);
      return;
    }
    onSubmit(item.id, draft);
  };

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto pr-1">
      {/* Header */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 py-4">
          <div className="flex items-center gap-3">
            <CardTitle className="font-mono text-lg">{item.id}</CardTitle>
            <StatusBadge status={status} />
          </div>
          <Button
            onClick={handleSubmit}
            variant={allScored ? "emerald" : "secondary"}
            disabled={item.completed}
          >
            {item.completed ? (
              <>
                <CheckCircle2 className="h-4 w-4" /> Submitted
              </>
            ) : (
              <>
                <Send className="h-4 w-4" /> Submit Evaluation
              </>
            )}
          </Button>
        </CardHeader>
      </Card>

      {/* Content Panel */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Item Content</CardTitle>
          <CardDescription>Source passage, question stem, and options</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-md border bg-gray-50 p-4 text-sm leading-relaxed text-gray-600">
            {item.passage}
          </div>
          <p className="text-sm font-semibold text-gray-800">{item.question}</p>
          <ul className="grid gap-2 sm:grid-cols-2">
            {item.options.map((opt, i) => {
              const key = OPTION_KEYS[i];
              const isCorrect = key === item.answer;
              return (
                <li
                  key={key}
                  className={
                    isCorrect
                      ? "flex items-start gap-2 rounded-md border border-emerald-300 bg-emerald-50 p-3 text-sm"
                      : "flex items-start gap-2 rounded-md border p-3 text-sm"
                  }
                >
                  <span
                    className={
                      isCorrect
                        ? "flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-xs font-bold text-white"
                        : "flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-gray-200 text-xs font-bold text-gray-600"
                    }
                  >
                    {key}
                  </span>
                  <span className="text-gray-700">{opt}</span>
                  {isCorrect && (
                    <Badge variant="emerald" className="ml-auto shrink-0">
                      Correct
                    </Badge>
                  )}
                </li>
              );
            })}
          </ul>
        </CardContent>
      </Card>

      {/* Evaluation & Rubric Panel */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Evaluation &amp; Rubric</CardTitle>
          <CardDescription>
            Tier A metrics carry 3-run LLM verification; Tier C metrics require
            direct human scoring.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="tier_a">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="tier_a" className="gap-2">
                <Bot className="h-4 w-4" /> Tier A: Automated (3-Run Gate)
              </TabsTrigger>
              <TabsTrigger value="tier_c" className="gap-2">
                <User className="h-4 w-4" /> Tier C: Direct Human Review
              </TabsTrigger>
              <TabsTrigger value="full_auto" className="gap-2">
                <Zap className="h-4 w-4" /> Full Automation (LLM-only)
              </TabsTrigger>
            </TabsList>

            {/* Tier A */}
            <TabsContent value="tier_a" className="space-y-5 pt-3">
              {TIER_A_METRICS.map(({ key, label }) => {
                const runs = item.tierARuns[key];
                const decision = gateDecision(runs);
                const needsHuman = decision !== GATE.AUTO_ACCEPT;
                return (
                  <div
                    key={key}
                    className="space-y-3 rounded-lg border p-4"
                  >
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-gray-800">
                        {label}
                      </h4>
                      <span className="text-xs text-gray-400">
                        3 independent LLM runs · temp 0.3
                      </span>
                    </div>
                    <RunPills runs={runs} />
                    {needsHuman ? (
                      <div className="space-y-2 border-t pt-3">
                        <p className="text-xs font-medium text-amber-700">
                          {decision === GATE.NEEDS_VERIFY
                            ? "Majority pre-filled. Verify or override:"
                            : "No consensus. Human score required:"}
                        </p>
                        <OrdinalSelector
                          value={draft[key]}
                          onChange={(v) => setDraft((d) => ({ ...d, [key]: v }))}
                          accent="amber"
                        />
                      </div>
                    ) : (
                      <p className="border-t pt-3 text-xs font-medium text-emerald-700">
                        Score {draft[key]} auto-accepted, no human action
                        required.
                      </p>
                    )}
                  </div>
                );
              })}
            </TabsContent>

            {/* Tier C */}
            <TabsContent value="tier_c" className="space-y-5 pt-3">
              <div className="rounded-md border border-indigo-200 bg-indigo-50 p-3 text-xs text-indigo-700">
                These dimensions fail under LLM-only evaluation and bypass
                automation entirely. LLM scores are never shown here to keep the
                review blind.
              </div>
              {TIER_C_METRICS.map(({ key, label }) => (
                <div key={key} className="space-y-3 rounded-lg border p-4">
                  <h4 className="text-sm font-semibold text-gray-800">
                    {label}
                  </h4>
                  <OrdinalSelector
                    value={draft[key]}
                    onChange={(v) => setDraft((d) => ({ ...d, [key]: v }))}
                    accent="indigo"
                  />
                </div>
              ))}
            </TabsContent>

            {/* Full Automation (LLM-only) */}
            <TabsContent value="full_auto" className="space-y-5 pt-3">
              <div className="flex items-start gap-2 rounded-md border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <p className="font-semibold">
                    Low-reliability mode. Use only when human capacity is
                    unavailable.
                  </p>
                  <p>
                    Tier C dimensions fail LLM-only evaluation (Krippendorff
                    α &lt; 0.30 vs human baseline). Scores below are 3-run
                    medians: a forced best guess, not a verified label. All
                    accepted values are flagged as <em>auto_lowconf</em> for
                    later human audit.
                  </p>
                </div>
              </div>

              {[...TIER_A_METRICS, ...TIER_C_METRICS].map(({ key, label }) => {
                const runs = item.tierARuns[key] ?? item.tierCRuns[key];
                const isTierC = !item.tierARuns[key];
                const forced = consensusScore(runs) ?? medianScore(runs);
                return (
                  <div key={key} className="space-y-3 rounded-lg border p-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-gray-800">
                        {label}
                      </h4>
                      <Badge variant={isTierC ? "rose" : "emerald"}>
                        {isTierC ? "Tier C: weak (α < 0.30)" : "Tier A: viable"}
                      </Badge>
                    </div>
                    <RunPills runs={runs} />
                    <p className="border-t pt-3 text-xs text-gray-500">
                      LLM-forced score:{" "}
                      <span className="font-bold text-gray-800">{forced}</span>{" "}
                      ({consensusScore(runs) !== null ? "consensus" : "median fallback"})
                    </p>
                  </div>
                );
              })}

              <Button
                className="w-full"
                variant="destructive"
                disabled={item.completed}
                onClick={() => {
                  const auto = {};
                  TIER_A_METRICS.forEach(({ key }) => {
                    const runs = item.tierARuns[key];
                    auto[key] = consensusScore(runs) ?? medianScore(runs);
                  });
                  TIER_C_METRICS.forEach(({ key }) => {
                    const runs = item.tierCRuns[key];
                    auto[key] = consensusScore(runs) ?? medianScore(runs);
                  });
                  setDraft(auto);
                }}
              >
                <Zap className="h-4 w-4" /> Accept All LLM Scores (Weak, Flag
                for Audit)
              </Button>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Incomplete-submission dialog */}
      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Evaluation incomplete</DialogTitle>
            <DialogDescription>
              The following dimensions still need a score before this item can
              be submitted:
            </DialogDescription>
          </DialogHeader>
          <ul className="list-inside list-disc text-sm text-gray-600">
            {missing.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              Continue Editing
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
