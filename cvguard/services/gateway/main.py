"""CVGuard Gateway Service (External-Facing API Entrypoint).

Phase 2: External client boundary routing /ingest/images to data-plane,
findings/ledger verification to governance, and evidence image proxying.
Phase 8: Keycloak OIDC Bearer token verification and RBAC enforcement
(analyst, admin, auditor) with upstream mutual TLS (mTLS) to internal microservices.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cvguard_schemas
from cvguard_schemas import (
    Role,
    UserIdentity,
    get_httpx_mtls_kwargs,
    verify_bearer_token,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("cvguard.gateway")

DATA_PLANE_URL = os.getenv("INTERNAL_DATA_PLANE_URL", "https://data-plane:8001")
GOVERNANCE_URL = os.getenv("INTERNAL_GOVERNANCE_URL", "https://governance:8005")

app = FastAPI(
    title="CVGuard Gateway Service",
    description="External-facing API gateway for CVGuard air-gapped vision assurance platform with OIDC RBAC and internal mTLS.",
    version="0.8.0",
)

# Enable CORS for browser frontends and external callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Authentication & RBAC Dependencies
# ==============================================================================

async def get_current_user(request: Request) -> UserIdentity:
    """Extract and validate Bearer JWT token from Authorization header.

    Rejects missing or invalid tokens with HTTP 401.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing or invalid authentication token",
        )

    token = auth_header[7:].strip()
    try:
        user = verify_bearer_token(token)
        return user
    except Exception as exc:
        logger.debug("Bearer token validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing or invalid authentication token",
        ) from exc


def require_roles(allowed_roles: Sequence[Role | str]):
    """Factory dependency ensuring authenticated user holds at least one allowed role.

    Returns 403 Forbidden with uniform message 'Forbidden: Insufficient permissions'
    to prevent leaking internal policy details.
    """
    async def role_checker(user: UserIdentity = Depends(get_current_user)) -> UserIdentity:
        role_strings = [r.value if isinstance(r, Role) else str(r) for r in allowed_roles]
        if not user.has_any_role(role_strings):
            logger.warning(
                "Access denied for user %s (roles: %s) - required one of %s",
                user.username,
                user.roles,
                role_strings,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Insufficient permissions",
            )
        return user

    return role_checker


def get_upstream_headers(user: UserIdentity | None = None, content_type: str | None = None) -> dict[str, str]:
    """Prepare forwarding headers including mTLS client identity and authenticated user context."""
    headers: dict[str, str] = {
        "X-Client-Identity": "gateway",
    }
    if content_type:
        headers["content-type"] = content_type
    if user:
        headers["X-User-Id"] = user.user_id
        headers["X-User-Username"] = user.username
        headers["X-User-Roles"] = ",".join(user.roles)
    return headers


# ==============================================================================
# Health & Status
# ==============================================================================

class HealthResponse(BaseModel):
    """Pydantic v2 health status schema."""

    status: str = Field(default="ok", description="Operational status flag.")
    service: str = Field(default="gateway", description="Name of the reporting service.")
    version: str = Field(default="0.8.0", description="Service semantic version.")
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
        version="0.8.0",
        schemas_version=cvguard_schemas.__version__,
        upstream_services={
            "data_plane": DATA_PLANE_URL,
            "governance": GOVERNANCE_URL,
        },
    )


# ==============================================================================
# Ingest & Data Plane Routing (Authenticated & RBAC Enforced)
# ==============================================================================

@app.post("/ingest/images")
async def ingest_images(
    request: Request,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN])),
):
    """Proxy image batch uploads to Data Plane service for ingestion and pHash detector evaluation."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/ingest/images"
    body = await request.body()
    content_type = request.headers.get("content-type")

    headers = get_upstream_headers(user=user, content_type=content_type)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=60.0, **mtls_kwargs) as client:
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
async def proxy_reference_distribution(
    request: Request,
    user: UserIdentity = Depends(require_roles([Role.ADMIN])),
):
    """Proxy reference distribution registration to Data Plane (Admin only)."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/reference-distributions"
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    headers = get_upstream_headers(user=user, content_type=content_type)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=30.0, **mtls_kwargs) as client:
        try:
            resp = await client.post(target_url, content=body, headers=headers)
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
async def get_reference_distributions(
    dataset_id: str | None = None,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy reference distribution query to Data Plane."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/reference-distributions"
    params = {"dataset_id": dataset_id} if dataset_id else {}
    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=10.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, params=params, headers=headers)
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


# ==============================================================================
# Governance Spine Routing (Authenticated & RBAC Enforced)
# ==============================================================================

@app.get("/findings")
async def get_findings(
    limit: int = 50,
    offset: int = 0,
    severity: str | None = None,
    asset_type: str | None = None,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy paginated and filtered findings queries directly to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/findings"
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if severity:
        params["severity"] = severity
    if asset_type:
        params["asset_type"] = asset_type

    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=10.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, params=params, headers=headers)
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
async def get_finding_by_ledger_id(
    ledger_id: int,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy individual finding lookup by ledger entry ID to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/findings/{ledger_id}"
    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=10.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, headers=headers)
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
async def verify_audit_ledger(
    from_id: int = 1,
    user: UserIdentity = Depends(require_roles([Role.AUDITOR, Role.ADMIN])),
):
    """Proxy full cryptographic hash chain and Ed25519 signature audit verification (Auditor or Admin)."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/audit/verify"
    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=30.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, params={"from_id": from_id}, headers=headers)
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
async def get_proxied_image(
    minio_key: str,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy evidence image rendering from data-plane to browser without exposing MinIO directly."""
    target_url = f"{DATA_PLANE_URL.rstrip('/')}/images/{minio_key}"
    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=15.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, headers=headers)
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


@app.get("/coverage")
async def get_coverage(
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy coverage manifest query to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/coverage"
    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=15.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy coverage query to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/findings/{ledger_id}/verify")
async def verify_finding_signature(
    ledger_id: int,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy individual finding signature verification to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/findings/{ledger_id}/verify"
    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=10.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy finding verification to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/reports/generate")
async def proxy_generate_report(
    since: int = 1,
    timestamp: str | None = None,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy JSON report generation to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/reports/generate"
    params: dict[str, Any] = {"since": since}
    if timestamp:
        params["timestamp"] = timestamp

    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=30.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, params=params, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy report generation to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/reports/generate.html")
async def proxy_generate_report_html(
    since: int = 1,
    timestamp: str | None = None,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy HTML report generation to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/reports/generate.html"
    params: dict[str, Any] = {"since": since}
    if timestamp:
        params["timestamp"] = timestamp

    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=30.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, params=params, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="text/html",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy HTML report generation to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


@app.get("/reports/generate.pdf")
async def proxy_generate_report_pdf(
    since: int = 1,
    timestamp: str | None = None,
    user: UserIdentity = Depends(require_roles([Role.ANALYST, Role.ADMIN, Role.AUDITOR])),
):
    """Proxy PDF report generation to Governance Spine."""
    target_url = f"{GOVERNANCE_URL.rstrip('/')}/reports/generate.pdf"
    params: dict[str, Any] = {"since": since}
    if timestamp:
        params["timestamp"] = timestamp

    headers = get_upstream_headers(user=user)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="gateway")

    async with httpx.AsyncClient(timeout=30.0, **mtls_kwargs) as client:
        try:
            resp = await client.get(target_url, params=params, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/pdf",
                headers={"Content-Disposition": resp.headers.get("content-disposition", 'attachment; filename="cvguard-report.pdf"')},
            )
        except httpx.RequestError as exc:
            logger.error("Failed to proxy PDF report generation to %s: %s", target_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway failed to reach governance at {GOVERNANCE_URL}: {exc}",
            )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
