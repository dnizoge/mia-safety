#!/bin/bash
# Run inference on a video file
# Usage: ./run_inference.sh <video_file>
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python -m ai_core.inference --source "$1" --weights weights/best_mia.pt
