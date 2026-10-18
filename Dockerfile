# Pinned base image: exact Python patch version, Debian 12 (bookworm)
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install pinned dependencies first (better layer caching).
# torch is taken from the official CPU-only wheel index (no CUDA, smaller image).
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project files (see .dockerignore for exclusions)
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/sample/ data/sample/

# Jupyter is optional and not part of the pinned benchmark environment.
# Default command: smoke test on the bundled sample data.
CMD ["python", "-m", "src.main", "--config", "configs/config.yaml"]
