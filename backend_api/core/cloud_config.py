import os

# Use CPU on cloud (no MPS/CUDA)
DEVICE = "cpu"

# Get port from environment
PORT = int(os.environ.get("PORT", 8000))
