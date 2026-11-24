import React from "react";
import { Badge } from "@/components/ui/badge";
import { GATE, gateDecision } from "@/data/mockData";
import { cn } from "@/lib/utils";

const GATE_BADGE = {
  [GATE.AUTO_ACCEPT]: { variant: "emerald", label: "3/3 Consensus: Auto-Accepted" },
  [GATE.NEEDS_VERIFY]: { variant: "amber", label: "2/3 Consensus: Needs Verification" },
  [GATE.ESCALATED]: { variant: "rose", label: "0/3 Consensus: Escalated to Human" },
};

const PILL_COLOR = {
  [GATE.AUTO_ACCEPT]: "bg-emerald-100 text-emerald-800 border-emerald-300",
  [GATE.NEEDS_VERIFY]: "bg-amber-100 text-amber-800 border-amber-300",
  [GATE.ESCALATED]: "bg-rose-100 text-rose-800 border-rose-300",
};

/**
 * Visual breakdown of the 3 independent LLM runs for a Tier A metric.
 */
export default function RunPills({ runs }) {
  const decision = gateDecision(runs);
  const badge = GATE_BADGE[decision];

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex gap-1.5">
        {runs.map((score, i) => (
          <div
            key={i}
            title={`Run ${i + 1}: ${score}`}
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-md border text-sm font-bold tabular-nums",
              PILL_COLOR[decision]
            )}
          >
            {score}
          </div>
        ))}
      </div>
      <Badge variant={badge.variant}>{badge.label}</Badge>
    </div>
  );
}
