# Deploying the dashboard to Vercel

The deployed dashboard sits behind a password. Do not skip step 3.

## Why the password matters

Vercel's Hobby plan does not include Password Protection (it needs Enterprise, or a paid add-on on Pro). Vercel Authentication on Hobby protects preview URLs only; the production domain stays publicly reachable.

A local build bundles `findingsData.js`, which holds results that are not published in this repository. An unprotected deploy would put them on an open URL.

`dashboard/middleware.js` handles this. It runs on Vercel's Edge, is free on Hobby, covers static assets as well as pages, and fails closed: if the environment variables are missing, every request is blocked.

## Deploy

### 1. Confirm secrets are excluded

`.gitignore` lists `.env`. Check before pushing:

```bash
git status --porcelain | findstr /C:".env"
```

Nothing should come back.

### 2. Import the project

On Vercel, choose Add New, then Project, pick the repository, and set:

| Setting | Value |
|---|---|
| Root Directory | `dashboard` |
| Framework Preset | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |

`dashboard/vercel.json` already sets SPA rewrites and `X-Robots-Tag: noindex`, so search engines do not index the site.

### 3. Set the password before the first deploy

In Project Settings, Environment Variables, for all environments:

| Name | Value |
|---|---|
| `DASHBOARD_USER` | for example `lei` |
| `DASHBOARD_PASSWORD` | a long random string used nowhere else |

Set these before deploying. If you deploy first, the site is reachable without credentials until you do.

Generate a password:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

### 4. Deploy and verify

After the build, open the URL in a private window. You should get a browser credential prompt. If the page loads without one, the middleware is not running: check that Root Directory is `dashboard` and that `middleware.js` sits beside `package.json`.

## What deploys and what does not

| Tab | Works on Vercel | Why |
|---|---|---|
| R&D Findings | Only in a local build | Compiled from `findingsData.js`, which is not in the repository |
| Workspace | Yes | Mock data |
| Pipeline, Active Learning, Fairness, Pairwise Elo | No | They call the FastAPI backend, which is not deployed |

The four API tabs show connection errors. Two options:

1. Deploy the dashboard without the backend and demo the API tabs locally. This is the simple path.
2. Port `src/api/server.py` to Vercel Python serverless functions under `api/`. Before doing that, note that `_runs_for_item()` in `server.py` generates seeded synthetic scores for items outside the replay cache, and the Elo endpoint uses a simulated judge. A public URL would show those as if they were results.

## Building with the findings data

A build from GitHub succeeds, but the R&D Findings tab shows a notice instead of results, because `dashboard/src/data/findingsData.js` is gitignored. To include the results, generate the file locally and deploy from your machine, so it never reaches GitHub:

```bash
python scripts/compute_human_baseline.py
python scripts/build_findings_data.py
```

```bash
npm i -g vercel
```

```bash
cd dashboard && vercel --prod
```

The Vercel CLI uploads your local working tree, including the generated file. Set `DASHBOARD_USER` and `DASHBOARD_PASSWORD` (step 3) before the first `vercel --prod`.

## Before removing the password

1. Confirm the results can be published.
2. Confirm whether rater-derived data can be shared at all.
3. Confirm the OpenStax CC BY-NC-SA terms permit what you show.
4. Remove or clearly label the two synthetic paths (`_runs_for_item` and the simulated Elo judge), so nothing on a public URL looks like a measured result when it is not.
