# ==============================================================================
# Multi-Stage Dockerfile for GhostTrack on Google Cloud Run
# Stage 1: Build the React + Vite Dashboard
# Stage 2: Minimal Python 3.11 image with PyTorch (CPU) & FastAPI Backend
# ==============================================================================

# --- Stage 1: Build Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python Backend Runtime ---
FROM python:3.11-slim AS runner

# Avoid Python buffering and .pyc generation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install minimal OS build tools and libraries needed by scipy / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (keeps image compact, ~180MB instead of 1.5GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend Python codebase and assets
COPY backend/ ./backend/
COPY engine/ ./engine/
COPY model/ ./model/
COPY shared/ ./shared/
COPY simulator/ ./simulator/
COPY data/ ./data/

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose default Cloud Run port
EXPOSE 8080

# Run FastAPI backend with Uvicorn (WebSockets & SPA)
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
