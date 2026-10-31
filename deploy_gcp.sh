#!/usr/bin/env bash
# =============================================================================
# GhostTrack: Google Cloud Run Deployment Script
# Deploys both FastAPI Backend (with WebSockets) and React Dashboard in 1 Step
# =============================================================================

set -e

# Terminal Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}===================================================================${NC}"
echo -e "${CYAN}   GhostTrack: Google Cloud Run Deployment                         ${NC}"
echo -e "${CYAN}===================================================================${NC}"

# 1. Check if gcloud is installed
if ! command -v gcloud >/dev/null 2>&1; then
    echo -e "${RED}[!] Google Cloud SDK ('gcloud') is not installed or not in PATH.${NC}"
    echo -e "    Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 2. Determine GCP Project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$CURRENT_PROJECT" ] || [ "$CURRENT_PROJECT" = "(unset)" ]; then
    echo -e "${YELLOW}[?] No default GCP project found in gcloud config.${NC}"
    read -p "Enter your Google Cloud Project ID: " GCP_PROJECT
    gcloud config set project "$GCP_PROJECT"
else
    GCP_PROJECT="$CURRENT_PROJECT"
    echo -e "${GREEN}[✔] Using GCP Project: ${CYAN}${GCP_PROJECT}${NC}"
fi

# 3. Determine Region (default to us-central1 or asia-south1)
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ghosttrack}"

echo -e "${GREEN}[✔] Target Service: ${CYAN}${SERVICE_NAME}${NC}"
echo -e "${GREEN}[✔] Region:         ${CYAN}${REGION}${NC}"

# 4. Enable required GCP services
echo -e "\n${CYAN}[1/3] Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project "$GCP_PROJECT"

# 5. Build and deploy container directly using Google Cloud Build & Cloud Run
echo -e "\n${CYAN}[2/3] Building & deploying to Cloud Run via Cloud Build...${NC}"
echo -e "${YELLOW}(This builds the React UI and Python container automatically in Google Cloud)${NC}"

gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --project "$GCP_PROJECT" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 3600 \
    --session-affinity \
    --min-instances 0 \
    --max-instances 5

# 6. Retrieve service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project "$GCP_PROJECT" --region "$REGION" --format 'value(status.url)' 2>/dev/null || true)

echo -e "\n${GREEN}===================================================================${NC}"
echo -e "${GREEN}   Deployment Complete! GhostTrack is LIVE on Google Cloud:        ${NC}"
echo -e "${CYAN}   Dashboard & API: ${SERVICE_URL}${NC}"
echo -e "${CYAN}   WebSocket Endpoint: wss://${SERVICE_URL#https://}/ws${NC}"
echo -e "${GREEN}===================================================================${NC}"
