# ---- Tabla Taal & Bol Identifier — Production Dockerfile ----
FROM python:3.11-slim

# System dependencies for librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY api/ api/

# Copy model checkpoints (only ResidualCNN + CRNN included via .gcloudignore)
COPY results/ results/

# Expose port (Cloud Run injects PORT env var, defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Run with uvicorn — single worker is fine for Cloud Run
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
