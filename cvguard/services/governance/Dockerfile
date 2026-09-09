# Multi-stage Dockerfile for CVGuard Governance Service
# Pinned Python 3.12 base image for deterministic, reproducible builds
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

COPY services/governance/pyproject.toml /build/services/governance/
RUN pip install "fastapi>=0.110.0,<1.0.0" "uvicorn[standard]>=0.28.0,<1.0.0" "pydantic>=2.7.0,<3.0.0" "cryptography>=42.0.0" "psycopg[binary]>=3.1.18"

# --- Runtime Stage ---
FROM python:3.12.9-slim-bookworm AS runner

RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup --shell /bin/false --no-create-home appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --chown=appuser:appgroup services/governance/ /app/

# DEV-ONLY secrets storage directory with restricted permissions
RUN mkdir -p /app/secrets && chown -R appuser:appgroup /app/secrets && chmod 700 /app/secrets

USER appuser:appgroup

EXPOSE 8005

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8005/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
