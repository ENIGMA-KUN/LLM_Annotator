import React, { useMemo, useState } from "react";
import MetricCards from "@/components/dashboard/MetricCards";
import QueueSidebar from "@/components/dashboard/QueueSidebar";
import ItemWorkspace from "@/components/dashboard/ItemWorkspace";
import {
  generateMockItems,
  deriveStatus,
  gateDecision,
  GATE,
  STATUS,
  TIER_A_METRICS,
  TIER_C_METRICS,
} from "@/data/mockData";

// The original annotator workspace (mock-data demo of the routing protocol).
export default function WorkspaceView() {
  const [items, setItems] = useState(() => generateMockItems(200));
  const [selectedId, setSelectedId] = useState(items[0]?.id ?? null);
  const [filter, setFilter] = useState("all");

  const selectedItem = items.find((i) => i.id === selectedId) ?? null;

  // ---- Derived dashboard stats -------------------------------------------
  const stats = useMemo(() => {
    const total = items.length;
    const completed = items.filter((i) => i.completed).length;

    let autoAcceptedDecisions = 0;
    let totalTierADecisions = 0;
    items.forEach((item) => {
      TIER_A_METRICS.forEach(({ key }) => {
        totalTierADecisions += 1;
        if (gateDecision(item.tierARuns[key]) === GATE.AUTO_ACCEPT) {
          autoAcceptedDecisions += 1;
        }
      });
    });

    // Workload reduction: share of ALL human-gradeable decisions
    // (items × 5 dims) eliminated by 3/3 auto-acceptance.
    const totalDecisions =
      total * (TIER_A_METRICS.length + TIER_C_METRICS.length);
    const workloadReductionPct =
      (autoAcceptedDecisions / Math.max(1, totalDecisions)) * 100;

    const pendingReview = items.filter(
      (i) => !i.completed && deriveStatus(i) === STATUS.NEEDS_REVIEW
    ).length;

    return {
      total,
      processed: completed,
      autoAcceptedDecisions,
      totalTierADecisions,
      workloadReductionPct,
      pendingReview,
      completed,
    };
  }, [items]);

  // ---- Submission handler -------------------------------------------------
  const handleSubmit = (itemId, scores) => {
    setItems((prev) => {
      const next = prev.map((item) => {
        if (item.id !== itemId) return item;
        const accepted = {};
        const human = {};
        TIER_A_METRICS.forEach(({ key }) => (accepted[key] = scores[key]));
        TIER_C_METRICS.forEach(({ key }) => (human[key] = scores[key]));
        return {
          ...item,
          acceptedScores: accepted,
          humanScores: human,
          completed: true,
        };
      });
      return next;
    });

    // Advance to next pending item in queue order
    const currentIdx = items.findIndex((i) => i.id === itemId);
    const ordered = [
      ...items.slice(currentIdx + 1),
      ...items.slice(0, currentIdx),
    ];
    const nextPending = ordered.find((i) => !i.completed && i.id !== itemId);
    if (nextPending) setSelectedId(nextPending.id);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      {/* Metric overview cards */}
      <MetricCards stats={stats} filter={filter} />

      {/* Split-screen workspace */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[340px_1fr]">
        <div className="min-h-0">
          <QueueSidebar
            items={items}
            selectedId={selectedId}
            onSelect={setSelectedId}
            filter={filter}
            onFilterChange={setFilter}
          />
        </div>
        <div className="min-h-0">
          <ItemWorkspace item={selectedItem} onSubmit={handleSubmit} />
        </div>
      </div>
    </div>
  );
}
