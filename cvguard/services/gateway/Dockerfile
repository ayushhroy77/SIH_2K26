# Multi-stage Dockerfile for CVGuard Gateway Service
# Pinned Python 3.12 base image for deterministic, reproducible builds
FROM python:3.12.9-slim-bookworm AS builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Create isolated virtual environment for dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install locked dependencies with cryptographic hash verification
COPY services/gateway/requirements.txt /build/services/gateway/
RUN pip install --no-cache-dir --require-hashes -r /build/services/gateway/requirements.txt

# Copy and install internal shared schemas package
COPY libs/schemas /build/libs/schemas
RUN pip install --no-deps /build/libs/schemas

# --- Runtime Stage ---
FROM python:3.12.9-slim-bookworm AS runner

# Security hardening: Create non-root user with dedicated UID/GID
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup --shell /bin/false --no-create-home appuser

# Runtime environment flags
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Copy application source code
COPY --chown=appuser:appgroup services/gateway/ /app/

# Switch to non-root user
USER appuser:appgroup

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
