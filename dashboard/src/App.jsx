import React, { useState } from "react";
import WorkspaceView from "@/views/WorkspaceView";
import PipelineView from "@/views/PipelineView";
import ActiveLearningView from "@/views/ActiveLearningView";
import FairnessView from "@/views/FairnessView";
import EloView from "@/views/EloView";
import FindingsView from "@/views/FindingsView";
import {
  Workflow, LayoutDashboard, GitBranch, Target, Scale, Swords, FlaskConical,
} from "lucide-react";

// FindingsView is first: it needs no API server, so it always renders.
const VIEWS = [
  { id: "findings", label: "R&D Findings", icon: FlaskConical, component: FindingsView },
  { id: "workspace", label: "Workspace", icon: LayoutDashboard, component: WorkspaceView },
  { id: "pipeline", label: "Pipeline", icon: GitBranch, component: PipelineView },
  { id: "active", label: "Active Learning", icon: Target, component: ActiveLearningView },
  { id: "fairness", label: "Fairness Audit", icon: Scale, component: FairnessView },
  { id: "elo", label: "Pairwise Elo", icon: Swords, component: EloView },
];

export default function App() {
  const [viewId, setViewId] = useState("findings");
  const ActiveView = VIEWS.find((v) => v.id === viewId).component;

  return (
    <div className="flex h-screen flex-col">
      {/* Translucent chrome. Content scrolls underneath rather than the bar
          consuming a fixed strip. See DESIGN_SYSTEM.md. */}
      <header className="material-chrome sticky top-0 z-20">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-3 lg:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-asu-maroon text-white">
              <Workflow className="h-[18px] w-[18px]" />
            </div>
            <div>
              <h1 className="type-heading">LLM as Judge (Annotating)</h1>
              <div className="mt-1 flex items-center gap-2">
                <span className="asu-rule" aria-hidden="true" />
                <p className="type-caption text-gray-500">
                  200 MCQs · 5 dimensions · 3-run confidence gate
                </p>
              </div>
            </div>
          </div>

          {/* Direct, specific labels. Active state carries the brand. */}
          <nav
            aria-label="Sections"
            className="flex items-center gap-0.5 rounded-lg border border-border bg-white/70 p-1 shadow-chip"
          >
            {VIEWS.map(({ id, label, icon: Icon }) => {
              const active = viewId === id;
              return (
                <button
                  key={id}
                  onClick={() => setViewId(id)}
                  aria-current={active ? "page" : undefined}
                  className={`pressable flex items-center gap-1.5 rounded-md px-3 py-1.5 type-caption font-semibold ${
                    active
                      ? "bg-asu-maroon text-white shadow-chip"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" /> {label}
                </button>
              );
            })}
          </nav>

          {viewId === "workspace" && (
            <div className="hidden items-center gap-4 type-caption text-gray-500 xl:flex">
              {[
                ["bg-emerald-500", "Auto-accepted"],
                ["bg-amber-500", "Needs verification"],
                ["bg-indigo-500", "Direct human"],
              ].map(([dot, label]) => (
                <span key={label} className="flex items-center gap-1.5">
                  <span className={`h-2 w-2 rounded-full ${dot}`} />
                  {label}
                </span>
              ))}
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-[1600px] min-h-0 flex-1 flex-col px-4 py-4 lg:px-6">
        {/* Keyed so a tab change materialises rather than snapping. */}
        <div key={viewId} className="materialize flex min-h-0 flex-1 flex-col">
          <ActiveView />
        </div>
      </main>
    </div>
  );
}
