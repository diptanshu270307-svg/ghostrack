# ==============================================================================
# Multi-Stage Dockerfile for GhostTrack on Hugging Face Spaces
# Stage 1: Build the React + Vite Dashboard
# Stage 2: Minimal Python 3.11 runtime with PyTorch (CPU) & FastAPI Backend
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
    PYTHONPATH=/app \
    PORT=7860

# Set up user for Hugging Face Spaces (runs under UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

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
COPY ghosttrack.py .
COPY ghosttrack_model.pt .
COPY backend/ ./backend/
COPY engine/ ./engine/
COPY model/ ./model/
COPY shared/ ./shared/
COPY simulator/ ./simulator/
COPY data/ ./data/

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Set permissions for user 1000
RUN chown -R user:user /app
USER user

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run FastAPI backend with Uvicorn (WebSockets & SPA)
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
