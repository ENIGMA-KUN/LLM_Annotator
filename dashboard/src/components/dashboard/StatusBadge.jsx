import React from "react";
import { Badge } from "@/components/ui/badge";
import { STATUS } from "@/data/mockData";

const CONFIG = {
  [STATUS.AUTO_ACCEPTED]: { variant: "emerald", label: "Auto-Accepted" },
  [STATUS.NEEDS_REVIEW]: { variant: "amber", label: "Needs Review" },
  [STATUS.PENDING_HUMAN]: { variant: "indigo", label: "Pending Direct Human" },
  [STATUS.COMPLETED]: { variant: "slate", label: "Completed" },
};

export default function StatusBadge({ status }) {
  const cfg = CONFIG[status] || CONFIG[STATUS.NEEDS_REVIEW];
  return <Badge variant={cfg.variant}>{cfg.label}</Badge>;
}
