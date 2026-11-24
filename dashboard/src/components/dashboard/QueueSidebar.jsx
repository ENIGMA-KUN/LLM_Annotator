import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import StatusBadge from "./StatusBadge";
import { deriveStatus, STATUS } from "@/data/mockData";
import { cn } from "@/lib/utils";

export default function QueueSidebar({
  items,
  selectedId,
  onSelect,
  filter,
  onFilterChange,
}) {
  const filtered = items.filter((item) => {
    if (filter === "all") return true;
    return deriveStatus(item) === filter;
  });

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <CardHeader className="space-y-3 border-b bg-gray-50/50 px-4 py-4">
        <CardTitle className="text-base font-semibold">
          Annotation Queue
          <span className="ml-2 text-sm font-normal text-gray-400">
            {filtered.length} items
          </span>
        </CardTitle>
        <Select value={filter} onValueChange={onFilterChange}>
          <SelectTrigger className="h-9">
            <SelectValue placeholder="Filter queue" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All items</SelectItem>
            <SelectItem value={STATUS.NEEDS_REVIEW}>Needs Review</SelectItem>
            <SelectItem value={STATUS.PENDING_HUMAN}>
              Pending Direct Human
            </SelectItem>
            <SelectItem value={STATUS.COMPLETED}>Completed</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent className="queue-scroll flex-1 overflow-y-auto p-0">
        <ul className="divide-y">
          {filtered.map((item) => {
            const status = deriveStatus(item);
            const isSelected = item.id === selectedId;
            return (
              <li key={item.id}>
                <button
                  onClick={() => onSelect(item.id)}
                  className={cn(
                    "flex w-full flex-col gap-1.5 px-4 py-3 text-left transition-colors hover:bg-gray-50",
                    isSelected &&
                      "border-l-2 border-indigo-500 bg-indigo-50/60 hover:bg-indigo-50/60"
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs font-semibold text-gray-700">
                      {item.id}
                    </span>
                    <StatusBadge status={status} />
                  </div>
                  <p className="line-clamp-2 text-xs text-gray-500">
                    {item.question}
                  </p>
                </button>
              </li>
            );
          })}
          {filtered.length === 0 && (
            <li className="px-4 py-8 text-center text-sm text-gray-400">
              No items match this filter.
            </li>
          )}
        </ul>
      </CardContent>
    </Card>
  );
}
