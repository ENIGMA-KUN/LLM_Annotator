@echo off
REM ============================================================
REM  LLM as Judge (Annotating) - one-click demo launcher
REM  Double-click this file. It checks everything, starts both
REM  servers in their own windows, and opens the browser.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   LLM as Judge (Annotating) - starting demo
echo ============================================================
echo.

REM ---- 1. Preflight -------------------------------------------------
echo [1/4] Running preflight checks...
python scripts\preflight.py
if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------
    echo   PREFLIGHT FAILED. See the failures listed above.
    echo.
    echo   Most common fix:
    echo      pip install -r requirements.txt
    echo      cd dashboard ^&^& npm install
    echo.
    echo   You can still continue: the R^&D Findings tab works
    echo   without the API server. Press Ctrl+C to stop instead.
    echo ------------------------------------------------------------
    echo.
    pause
)

REM ---- 2. Refresh the findings data ---------------------------------
echo.
echo [2/4] Rebuilding findings data from the result CSVs...
python scripts\compute_human_baseline.py
python scripts\build_findings_data.py

REM ---- 3. Start the API server in its own window --------------------
echo.
echo [3/4] Starting API server on port 8000...
start "LLM as Judge - API (port 8000)" cmd /k "python -m uvicorn src.api.server:app --port 8000"

REM Give uvicorn a moment to bind the port before Vite proxies to it.
timeout /t 4 /nobreak >nul

REM ---- 4. Start the dashboard in its own window ---------------------
echo.
echo [4/4] Starting dashboard on port 5173...
start "LLM as Judge - Dashboard (port 5173)" cmd /k "cd dashboard && npm run dev"

REM Vite needs a few seconds on a cold start.
timeout /t 8 /nobreak >nul
start "" "http://localhost:5173"

echo.
echo ============================================================
echo   Running.
echo     Dashboard : http://localhost:5173
echo     API       : http://127.0.0.1:8000/api/health
echo.
echo   It opens on the R^&D Findings tab, which needs no API.
echo   Close the two server windows when you are finished.
echo ============================================================
echo.
pause
