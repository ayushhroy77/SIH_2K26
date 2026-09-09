# Multi-stage Dockerfile for CVGuard Model Plane Service
# Pinned Python 3.12 base image for deterministic, reproducible builds
#
# SECURITY DIRECTIVE:
# In production, this container MUST be launched with the gVisor (runsc) runtime:
#   docker run --runtime=runsc --security-opt no-new-privileges:true ...
# This ensures that hostile neural network checkpoints attempting kernel-level
# privilege escalation or container escapes are constrained within an application kernel.

FROM python:3.12.9-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY libs/schemas /build/libs/schemas
RUN pip install --no-deps /build/libs/schemas

COPY services/model-plane/pyproject.toml /build/services/model-plane/
RUN pip install \
    "fastapi>=0.110.0,<1.0.0" \
    "uvicorn[standard]>=0.28.0,<1.0.0" \
    "pydantic>=2.7.0,<3.0.0" \
    "httpx>=0.27.0" \
    "python-multipart>=0.0.9" \
    "numpy>=1.26.0" \
    "scipy>=1.12.0" \
    "scikit-learn>=1.4.0"

# --- Runtime Stage ---
FROM python:3.12.9-slim-bookworm AS runner

# Create non-root dedicated application user (UID 10001)
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup --shell /bin/false --no-create-home appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR="/tmp/cvguard_scratch"

WORKDIR /app

# Prepare sandboxed scratch directory writable only by appuser
RUN mkdir -p /tmp/cvguard_scratch && \
    chown -R appuser:appgroup /tmp/cvguard_scratch && \
    chmod 700 /tmp/cvguard_scratch

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --chown=appuser:appgroup services/model-plane/ /app/

USER appuser:appgroup

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
