#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Checking compose services..."
docker compose ps

echo
echo "[2/3] Checking API health..."
API_HEALTH="$(curl -fsS http://127.0.0.1:8000/healthz)"
echo "api_server: ${API_HEALTH}"

echo
echo "[3/3] Checking AI health..."
AI_HEALTH="$(curl -fsS http://127.0.0.1:8001/healthz)"
echo "ai_server: ${AI_HEALTH}"

echo
echo "Post-deploy checks passed."
