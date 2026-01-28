#!/bin/bash
# Start the MIA API server
echo "🚀 Starting MIA API Server..."
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
uvicorn backend_api.main:app --reload --host 0.0.0.0 --port 8000
