import React from "react";
import { cn } from "@/lib/utils";

const SCORE_LABELS = { 1: "Poor", 2: "Fair", 3: "Good", 4: "Excellent" };

/**
 * Styled 1-4 ordinal toggle pills for human score entry.
 */
export default function OrdinalSelector({ value, onChange, accent = "indigo" }) {
  const accentClasses = {
    indigo:
      "border-indigo-500 bg-indigo-600 text-white shadow-sm hover:bg-indigo-600",
    amber:
      "border-amber-500 bg-amber-500 text-white shadow-sm hover:bg-amber-500",
  };

  return (
    <div className="flex gap-2" role="radiogroup">
      {[1, 2, 3, 4].map((score) => {
        const selected = value === score;
        return (
          <button
            key={score}
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(score)}
            className={cn(
              "flex h-12 w-16 flex-col items-center justify-center rounded-md border text-sm font-semibold transition-all",
              selected
                ? accentClasses[accent]
                : "border-gray-200 bg-white text-gray-600 hover:border-gray-300 hover:bg-gray-50"
            )}
          >
            <span className="text-base font-bold tabular-nums">{score}</span>
            <span
              className={cn(
                "text-[10px] font-medium",
                selected ? "text-white/80" : "text-gray-400"
              )}
            >
              {SCORE_LABELS[score]}
            </span>
          </button>
        );
      })}
    </div>
  );
}
