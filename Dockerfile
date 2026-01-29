FROM python:3.11-slim-bookworm

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/outputs logs/api

# Expose port
EXPOSE 8000

# Run the application (use shell form to interpret $PORT)
CMD uvicorn backend_api.main:app --host 0.0.0.0 --port ${PORT:-8000}
