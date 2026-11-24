import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ListChecks, Zap, Filter } from "lucide-react";

export default function MetricCards({ stats, filter }) {
  const {
    total,
    processed,
    autoAcceptedDecisions,
    totalTierADecisions,
    workloadReductionPct,
    pendingReview,
    completed,
  } = stats;

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {/* Queue Progress */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-500">
            Total Queue Progress
          </CardTitle>
          <ListChecks className="h-4 w-4 text-gray-400" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold tabular-nums">
            {processed}
            <span className="text-lg font-medium text-gray-400">/{total}</span>
          </div>
          <p className="mb-2 text-xs text-muted-foreground">items processed</p>
          <Progress
            value={(processed / Math.max(1, total)) * 100}
            indicatorClassName="bg-gray-900"
          />
        </CardContent>
      </Card>

      {/* Workload Reduction */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-500">
            Workload Reduction
          </CardTitle>
          <Zap className="h-4 w-4 text-emerald-500" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold tabular-nums text-emerald-600">
            {workloadReductionPct.toFixed(1)}%
          </div>
          <p className="mb-2 text-xs text-muted-foreground">
            {autoAcceptedDecisions} of {totalTierADecisions} Tier A decisions
            auto-accepted (3/3 consensus)
          </p>
          <Progress
            value={workloadReductionPct}
            indicatorClassName="bg-emerald-500"
          />
        </CardContent>
      </Card>

      {/* Queue Status */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-500">
            Queue Status
          </CardTitle>
          <Filter className="h-4 w-4 text-indigo-500" />
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline gap-4">
            <div>
              <div className="text-3xl font-bold tabular-nums text-amber-600">
                {pendingReview}
              </div>
              <p className="text-xs text-muted-foreground">pending review</p>
            </div>
            <div>
              <div className="text-3xl font-bold tabular-nums text-gray-700">
                {completed}
              </div>
              <p className="text-xs text-muted-foreground">completed</p>
            </div>
          </div>
          <div className="mt-2">
            <Badge variant="indigo">
              Active filter: {filter === "all" ? "All items" : filter.replace("_", " ")}
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
