"""CVGuard Gateway Service (External-Facing API Entrypoint).

Phase 2: External client boundary routing /ingest/images to data-plane,
findings/ledger verification to governance, and evidence image proxying.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cvguard_schemas

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("cvguard.gateway")

DATA_PLANE_URL = os.getenv("INTERNAL_DATA_PLANE_URL", "http://data-plane:8001")
GOVERNANCE_URL = os.getenv("INTERNAL_GOVERNANCE_URL", "http://governance:8005")

app = FastAPI(
    title="CVGuard Gateway Service",
    description="External-facing API gateway for CVGuard air-gapped vision assurance platform.",
    version="0.2.0",
)

# Enable CORS for browser frontends and external callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Pydantic v2 health status schema."""

    status: str = Field(default="ok", description="Operational status flag.")
    service: str = Field(default="gateway", description="Name of the reporting service.")
    version: str = Field(default="0.2.0", description="Service semantic version.")
    schemas_version: str = Field(
        default=cvguard_schemas.__version__,
        description="Version of cvguard_schemas linked to this service runtime.",
    )
    upstream_services: dict[str, str] = Field(
        default_factory=dict,
        description="Configured internal routing targets.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness/liveness health check probe endpoint."""
    return HealthResponse(
        status="ok",
        service="gateway",
        version="0.2.0",
        schemas_version=cvguard_schemas.__version__,
        upstream_services={
            "data_plane": DATA_PLANE_URL,
            "governance": GOVERNANCE_URL,
        },
    )


@app.post("/ingest/images")
async def ingest_images(request: Request):
    """Proxy image batch uploads to Data Plane service for ingestion and pHash detector evaluation."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/ingest/images"
    body = await request.body()
    content_type = request.headers.get("content-type")

    headers: dict[str, str] = {}
    if content_type:
        headers["content-type"] = content_type

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(target_url, content=body, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy ingest to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach data-plane at {DATA_PLANE_URL}: {exc}",
            )


@app.post("/reference-distributions")
async def proxy_reference_distribution(request: Request):
    """Proxy reference distribution registration to Data Plane."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/reference-distributions"
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(target_url, content=body, headers={"content-type": content_type})
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy reference-distribution to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach data-plane at {DATA_PLANE_URL}: {exc}",
            )


@app.get("/reference-distributions")
async def get_reference_distributions(dataset_id: str | None = None):
    """Proxy reference distribution query to Data Plane."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/reference-distributions"
    params = {"dataset_id": dataset_id} if dataset_id else {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(target_url, params=params)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy reference-distributions query to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach data-plane at {DATA_PLANE_URL}: {exc}",
            )



@app.get("/findings")
async def get_findings(
    limit: int = 50,
    offset: int = 0,
    severity: str | None = None,
    asset_type: str | None = None,
):
    """Proxy paginated and filtered findings queries directly to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/findings"
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if severity:
        params["severity"] = severity
    if asset_type:
        params["asset_type"] = asset_type

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(target_url, params=params)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy findings query to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/findings/{ledger_id}")
async def get_finding_by_ledger_id(ledger_id: int):
    """Proxy individual finding lookup by ledger entry ID to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/findings/{ledger_id}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(target_url)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy finding lookup %d to %s: %s", ledger_id, target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/audit/verify")
async def verify_audit_ledger(from_id: int = 1):
    """Proxy full cryptographic hash chain and Ed25519 signature audit verification."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/audit/verify"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(target_url, params={"from_id": from_id})
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy audit verification to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/images/{minio_key:path}")
async def get_proxied_image(minio_key: str):
    """Proxy evidence image rendering from data-plane to browser without exposing MinIO directly."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/images/{minio_key}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(target_url)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "image/jpeg"),
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy image '%s' from data-plane: %s", minio_key, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to fetch image from data-plane: {exc}",
            )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
