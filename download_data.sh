#!/bin/bash
# Download training dataset from Roboflow
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python -m ai_core.data.download_data
