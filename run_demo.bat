@echo off
REM =============================================================================
REM InSight AI / GhostTrack: Resilient Indoor Positioning Demo Launcher (Windows)
REM =============================================================================

echo ===================================================================
echo    InSight AI / GhostTrack: Resilient Indoor Positioning System    
echo ===================================================================

cd /d "%~dp0"

REM 1. Python check
if not exist ".venv\Scripts\python.exe" (
    echo [!] Creating Python virtual environment...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

REM 2. Check MQTT Broker
echo [1/4] Checking MQTT Broker...
netstat -ano | findstr 1883 >nul
if %errorlevel% neq 0 (
    echo [*] Launching embedded MQTT broker...
    start "GhostTrack MQTT Broker" /b .venv\Scripts\python.exe scripts\embedded_broker.py
    timeout /t 1 /nobreak >nul
)

REM 3. Backend
echo [2/4] Starting FastAPI Backend...
start "GhostTrack Backend" /b .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
timeout /t 1 /nobreak >nul

REM 4. Simulator
echo [3/4] Starting Sensor Simulator...
start "GhostTrack Simulator" /b .venv\Scripts\python.exe -m simulator.sim --rate-hz 10.0
timeout /t 1 /nobreak >nul

REM 5. Frontend
if exist "frontend" (
    echo [4/4] Starting React Frontend...
    cd frontend
    if not exist "node_modules" (
        call npm install
    )
    start "GhostTrack Frontend" npm run dev
    cd ..
)

echo ===================================================================
echo    GhostTrack is LIVE!
echo    - Dashboard: http://localhost:5173
echo    - API Docs:  http://127.0.0.1:8000/docs
echo ===================================================================
pause
