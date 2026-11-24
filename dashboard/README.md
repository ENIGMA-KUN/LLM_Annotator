# LLM as Judge (Annotating): dashboard

React frontend for the hybrid LLM and expert annotation pipeline, backed by the FastAPI server in `src/api/server.py`.

## Tabs

| Tab | Purpose | Needs the API |
|-----|---------|---------------|
| R&D Findings | Measured results, compiled at build time from `findingsData.js` | No |
| Workspace | Mock-data demo of the routing protocol: queue, 3-run gate, blind Tier C review, full-automation view | No |
| Pipeline | Upload a CSV, annotate 3 times (replayed scores or live LLM), blind review queue, finalize (hybrid or full automation), export CSV | Yes |
| Active Learning | Uncertainty-based routing: budget curves (active vs. random) and the most uncertain items | Yes |
| Fairness Audit | Krippendorff's alpha per subgroup (subject, difficulty, rater agreement, passage length) with gap flags | Yes |
| Pairwise Elo | Pairwise tournament for subjective criteria, Elo mapped back to 1 to 4 (simulated judge for now) | Yes |

## Routing logic

- Tier A (Linguistic Feature, Topic Relevance): 3 independent LLM runs.
  - 3/3 agree: auto-accepted (emerald)
  - 2/3 agree: majority kept, needs verification (amber)
  - all differ: escalated to human review (rose)
- Tier C (Writing Clarity, Semantic Plausibility, Semantic Uniqueness): always routed to blind human review (indigo). LLM scores are never shown to the reviewer.
- Full automation (LLM only): forces the 3-run median where there is no consensus, and flags every forced label `auto_lowconf` for audit.

## Run

From the repository root, start the API:

```bash
python -m uvicorn src.api.server:app --port 8000
```

In a second terminal, start the UI:

```bash
cd dashboard && npm install && npm run dev
```

It opens at http://localhost:5173 and proxies `/api` to port 8000. Live LLM calls need `GEMINI_API_KEY` in the repository-root `.env`. Replay annotation, the queue, finalize, export and the analysis tabs cost nothing.

Without the files under `Data/`, the build still works: the Workspace tab runs on mock data, and the R&D Findings tab explains that its data is not included.

## Stack

React 18, Vite, Tailwind CSS, shadcn/ui-style components on Radix primitives, lucide-react; FastAPI and uvicorn for the backend.
