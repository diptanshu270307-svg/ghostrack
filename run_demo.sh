#!/usr/bin/env bash
# =============================================================================
# InSight AI / GhostTrack: Resilient Indoor Positioning Demo Launcher
# Starts MQTT Broker, Simulator, FastAPI Backend, and React Dashboard
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}===================================================================${NC}"
echo -e "${CYAN}   InSight AI / GhostTrack: Resilient Indoor Positioning System    ${NC}"
echo -e "${CYAN}===================================================================${NC}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 1. Environment & Python check
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}[!] Python virtual environment (.venv) not found. Creating...${NC}"
    python3 -m venv "$ROOT_DIR/.venv"
    "$ROOT_DIR/.venv/bin/pip" install --upgrade pip
    "$ROOT_DIR/.venv/bin/pip" install -r "$ROOT_DIR/requirements.txt"
fi

PIDS=()

cleanup() {
    echo -e "\n${YELLOW}[*] Shutting down GhostTrack demo services...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null
    echo -e "${GREEN}[✔] All services stopped cleanly.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 2. Start or verify MQTT Broker
echo -e "${CYAN}[1/4] Checking MQTT Broker on 127.0.0.1:1883...${NC}"
MQTT_RUNNING=false
if "$VENV_PYTHON" -c "import socket; s = socket.socket(); s.settimeout(0.5); exit(s.connect_ex(('127.0.0.1', 1883)))" 2>/dev/null; then
    echo -e "${GREEN}[✔] MQTT Broker is already active on port 1883.${NC}"
    MQTT_RUNNING=true
fi

if [ "$MQTT_RUNNING" = false ]; then
    if command -v mosquitto >/dev/null 2>&1; then
        echo -e "${GREEN}[*] Starting local Eclipse Mosquitto binary...${NC}"
        mosquitto -c "$ROOT_DIR/mosquitto/mosquitto.conf" >/dev/null 2>&1 &
        PIDS+=($!)
    elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        echo -e "${GREEN}[*] Starting Mosquitto via Docker Compose...${NC}"
        docker compose up -d mosquitto
    else
        echo -e "${YELLOW}[*] Mosquitto/Docker not found. Launching embedded fallback MQTT broker...${NC}"
        "$VENV_PYTHON" "$ROOT_DIR/scripts/embedded_broker.py" &
        PIDS+=($!)
    fi
    sleep 1
fi

# 3. Start Backend (FastAPI + WebSockets)
echo -e "${CYAN}[2/4] Starting FastAPI Backend on http://127.0.0.1:8000...${NC}"
"$VENV_PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload --log-level info &
PIDS+=($!)
sleep 1

# 4. Start Sensor Simulator
echo -e "${CYAN}[3/4] Starting Sensor Simulator (10 Hz RF + IMU)...${NC}"
"$VENV_PYTHON" -m simulator.sim --rate-hz 10.0 &
PIDS+=($!)
sleep 1

# 5. Start Frontend
echo -e "${CYAN}[4/4] Starting React Dashboard (Vite)...${NC}"
if [ -d "$ROOT_DIR/frontend" ]; then
    cd "$ROOT_DIR/frontend"
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}[*] Installing frontend dependencies...${NC}"
        npm install
    fi
    npm run dev &
    PIDS+=($!)
    cd "$ROOT_DIR"
fi

echo -e "${GREEN}===================================================================${NC}"
echo -e "${GREEN}   GhostTrack is LIVE!                                             ${NC}"
echo -e "${GREEN}   - Dashboard: http://localhost:5173                              ${NC}"
echo -e "${GREEN}   - API Docs:  http://127.0.0.1:8000/docs                         ${NC}"
echo -e "${GREEN}   Press Ctrl+C to stop all services.                              ${NC}"
echo -e "${GREEN}===================================================================${NC}"

# Keep script running
wait
