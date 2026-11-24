"""Pre-demo smoke test. Run this before you present, not during.

    python scripts/preflight.py

Checks, in the order they would fail you on the day:
  1. Python dependencies importable
  2. API server imports and exposes its routes
  3. Required data files present and non-empty
  4. Findings data compiled and fresh
  5. Dashboard installed for THIS platform
  6. API key present if you intend a live call

Exit 0 means the demo path is safe. Exit 1 means fix something before you stand up.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if platform.system() == "Windows" and not os.environ.get("WT_SESSION"):
    GREEN = RED = YELLOW = DIM = RESET = ""

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    mark = f"{GREEN}PASS{RESET}" if ok else (f"{RED}FAIL{RESET}" if fatal else f"{YELLOW}WARN{RESET}")
    print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    results.append((name, ok or not fatal, detail))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    print("=" * 72)
    print("LLM as Judge (Annotating) - pre-demo preflight")
    print(f"repo: {ROOT}")
    print(f"python: {sys.version.split()[0]}  platform: {platform.system()}")
    print("=" * 72)

    section("1. Python dependencies")
    for mod in ("fastapi", "uvicorn", "pandas", "numpy", "pydantic", "krippendorff"):
        try:
            importlib.import_module(mod)
            check(mod, True)
        except Exception as exc:
            check(mod, False, f"{type(exc).__name__}: pip install -r requirements.txt")

    section("2. API server")
    sys.path.insert(0, str(ROOT))
    try:
        from src.api.server import app  # noqa: WPS433

        routes = sorted({r.path for r in app.routes if r.path.startswith("/api")})
        check("src.api.server imports", True, f"{len(routes)} api routes")
        for required in ("/api/health", "/api/upload", "/api/annotate", "/api/queue"):
            check(f"route {required}", required in routes)
    except Exception as exc:
        check("src.api.server imports", False, f"{type(exc).__name__}: {exc}")

    section("3. Data files")
    required_files = [
        "Data/processed/pipeline_upload_200items.csv",
        "Data/processed/rubric.json",
        "Data/results/neurips_experiments/results_with_cis.csv",
        "Data/results/neurips_experiments/item_level_routing.csv",
        "Data/results/neurips_experiments/threshold_sensitivity.csv",
        "Data/results/neurips_experiments/power_analysis.csv",
        "Data/results/neurips_experiments/fairness_audit.csv",
        "Data/results/neurips_experiments/experiment_summary.json",
    ]
    for rel in required_files:
        p = ROOT / rel
        size = p.stat().st_size if p.exists() else 0
        check(rel, p.exists() and size > 0, f"{size:,} bytes" if size else "missing or empty")

    section("4. Findings tab data")
    fd = ROOT / "dashboard" / "src" / "data" / "findingsData.js"
    if check("findingsData.js exists", fd.exists(), "run: python scripts/build_findings_data.py"):
        text = fd.read_text(encoding="utf-8")
        try:
            payload = json.loads(text[text.index("{"): text.rindex("};") + 1])
            n = len(payload.get("bestPerDimension", []))
            check("findingsData parses", True, f"{n} dimensions, generated {payload.get('generatedAt','?')[:10]}")
            check("routing paradox rows", len(payload.get("routingParadox", [])) >= 5)
            check("active learning curve", len(payload.get("activeLearning", [])) >= 1)
        except Exception as exc:
            check("findingsData parses", False, str(exc))

    section("5. Dashboard")
    dash = ROOT / "dashboard"
    nm = dash / "node_modules"
    check("node_modules present", nm.exists(), "run: cd dashboard && npm install")
    check("FindingsView.jsx present", (dash / "src" / "views" / "FindingsView.jsx").exists())
    # node_modules copied between platforms is the classic silent killer.
    esbuild_pkgs = list((nm / "@esbuild").glob("*")) if (nm / "@esbuild").exists() else []
    want = {"Windows": "win32", "Linux": "linux", "Darwin": "darwin"}.get(platform.system(), "")
    if esbuild_pkgs:
        names = [p.name for p in esbuild_pkgs]
        check(
            "esbuild binary matches platform",
            any(want in n for n in names),
            f"found {names}; expected one containing '{want}'. If wrong: rm -rf node_modules && npm install",
        )
    try:
        v = subprocess.run(["node", "-v"], capture_output=True, text=True, timeout=10)
        check("node available", v.returncode == 0, v.stdout.strip())
    except Exception as exc:
        check("node available", False, str(exc))

    section("6. Live LLM call (optional)")
    env = ROOT / ".env"
    has_key = env.exists() and "GEMINI_API_KEY" in env.read_text(encoding="utf-8", errors="ignore")
    check(
        "GEMINI_API_KEY in .env",
        has_key,
        "only needed for a live annotate; replay mode works without it",
        fatal=False,
    )

    print("\n" + "=" * 72)
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"{RED}{len(failed)} check(s) failed:{RESET}")
        for n in failed:
            print(f"  - {n}")
        print("\nFix these before presenting. Replay mode plus the R&D Findings tab")
        print("is the safe path if the API cannot be made to start.")
        return 1
    print(f"{GREEN}All checks passed. Demo path is safe.{RESET}")
    print("\nStart order:")
    print("  1. python -m uvicorn src.api.server:app --port 8000")
    print("  2. cd dashboard && npm run dev")
    print("  3. open http://localhost:5173  (opens on R&D Findings, which needs no API)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
