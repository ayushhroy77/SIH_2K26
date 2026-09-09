"""CVGuard Governance Spine Service.

Phase 1: Cryptographic Ed25519 signer and tamper-evident PostgreSQL audit ledger.
Phase 8: Mutual TLS enforcement and Role-Based Access Control (RBAC).
- POST /findings: Strictly restricted to internal service planes (mTLS authenticated).
- GET /findings, GET /reports/*: Accessible to 'analyst', 'admin', and 'auditor' roles.
"""

from __future__ import annotations

import logging
import os
import ssl
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Sequence

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

import cvguard_schemas
from cvguard_schemas import (
    FINDING_PRODUCER_SERVICES,
    INTERNAL_SERVICE_IDENTITIES,
    AssetType,
    Finding,
    Report,
    Role,
    Severity,
    SignedFinding,
    UserIdentity,
    verify_bearer_token,
)
from ledger import AuditLedger, PostgresConfig, VerificationResult
from reports import (
    generate_report,
    load_coverage_manifest,
    merge_coverage_statement,
    render_report_html,
    render_report_pdf,
)
from signer import LocalFileSigner, Signer

logger = logging.getLogger("cvguard.governance.api")

# Global singleton instances initialized during application lifecycle
_signer: Signer | None = None
_ledger: AuditLedger | None = None


def get_signer() -> Signer:
    """Dependency / accessor for the active Signer instance."""
    global _signer
    if _signer is None:
        _signer = LocalFileSigner()
    return _signer


def get_ledger() -> AuditLedger:
    """Dependency / accessor for the active AuditLedger instance."""
    global _ledger
    if _ledger is None:
        signer = get_signer()
        _ledger = AuditLedger(config=PostgresConfig.from_env(), signer=signer)
    return _ledger


# ==============================================================================
# Security & RBAC Dependencies
# ==============================================================================

async def require_finding_producer_service(request: Request) -> str:
    """Ensure that only authorized internal evaluation planes can POST findings.

    Human users (including administrators) must never directly fabricate findings.
    """
    caller = request.headers.get("X-Client-Identity", "").strip()
    mtls_enabled = os.getenv("CVGUARD_MTLS_ENABLED", "false").lower() in ("true", "1", "yes")

    # If an Authorization header with user token is present without internal service credentials, reject immediately
    auth_header = request.headers.get("authorization", "")
    if auth_header and caller not in FINDING_PRODUCER_SERVICES:
        logger.warning("Rejected direct finding submission attempt with user token from %s", caller)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Insufficient permissions",
        )

    if mtls_enabled:
        if caller not in FINDING_PRODUCER_SERVICES:
            logger.warning("Rejected unauthenticated or non-producer finding dispatch from %s", caller)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Insufficient permissions",
            )

    return caller


async def require_read_role(request: Request) -> UserIdentity | None:
    """Validate that caller has one of: analyst, admin, auditor, or is an internal service."""
    # 1. Internal service caller via mTLS
    caller = request.headers.get("X-Client-Identity", "").strip()
    if caller in INTERNAL_SERVICE_IDENTITIES:
        # Check forwarded user roles if present
        forwarded_roles = request.headers.get("X-User-Roles")
        if forwarded_roles:
            roles = [r.strip() for r in forwarded_roles.split(",") if r.strip()]
            if not any(r in (Role.ANALYST.value, Role.ADMIN.value, Role.AUDITOR.value) for r in roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient permissions",
                )
        return None

    # 2. Check forwarded role header or Bearer token
    forwarded_roles = request.headers.get("X-User-Roles")
    auth_header = request.headers.get("authorization")

    mtls_enabled = os.getenv("CVGUARD_MTLS_ENABLED", "false").lower() in ("true", "1", "yes")

    if not forwarded_roles and not auth_header:
        if mtls_enabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid authentication token",
            )
        return None

    if forwarded_roles:
        roles = [r.strip() for r in forwarded_roles.split(",") if r.strip()]
        if not any(r in (Role.ANALYST.value, Role.ADMIN.value, Role.AUDITOR.value) for r in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Insufficient permissions",
            )
        return UserIdentity(user_id="forwarded", username="forwarded", roles=tuple(roles))

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        try:
            user = verify_bearer_token(token)
            if not user.has_any_role([Role.ANALYST, Role.ADMIN, Role.AUDITOR]):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient permissions",
                )
            return user
        except HTTPException:
            raise
        except Exception as exc:
            logger.debug("Failed bearer token validation: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid authentication token",
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Missing or invalid authentication token",
    )


# ==============================================================================
# Application Lifecycle & Endpoints
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle handler."""
    logger.info("Initializing CVGuard Governance Spine Service...")
    signer = get_signer()
    logger.info(
        "Governance Ed25519 signer ready (public key: %s...)",
        signer.get_public_key_hex()[:16],
    )
    get_ledger()
    yield
    logger.info("CVGuard Governance Spine Service shut down.")


app = FastAPI(
    title="CVGuard Governance Spine Service",
    description="Tamper-evident audit ledger, Ed25519 finding signer, and assurance verification spine with mTLS and RBAC.",
    version="0.8.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    """Pydantic v2 health status schema."""

    status: str = Field(default="ok", description="Operational status flag.")
    service: str = Field(default="governance", description="Name of the reporting service.")
    version: str = Field(default="0.8.0", description="Service semantic version.")
    schemas_version: str = Field(
        default=cvguard_schemas.__version__,
        description="Version of cvguard_schemas linked to this service runtime.",
    )
    signer_public_key: str = Field(
        ...,
        description="Hex-encoded 32-byte Ed25519 public key of the governance signer.",
    )
    mtls_enabled: bool = Field(
        default_factory=lambda: os.getenv("CVGUARD_MTLS_ENABLED", "false").lower() in ("true", "1", "yes"),
        description="Indicates whether mutual TLS client certificate verification is enforced.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness/liveness health check probe endpoint."""
    signer = get_signer()
    return HealthResponse(
        status="ok",
        service="governance",
        version="0.8.0",
        schemas_version=cvguard_schemas.__version__,
        signer_public_key=signer.get_public_key_hex(),
    )


@app.post(
    "/findings",
    response_model=SignedFinding,
    status_code=status.HTTP_201_CREATED,
    summary="Submit and cryptographically seal an evaluation finding (Internal Planes Only)",
)
async def create_finding(
    finding: Finding,
    caller: str = Depends(require_finding_producer_service),
) -> SignedFinding:
    """Accept an evaluation Finding from an authorized plane, hash-chain it, and sign it with Ed25519.

    Rejects any payload that fails canonical schema validation.
    Restricted strictly to authenticated internal plane callers.
    """
    ledger = get_ledger()
    try:
        signed_finding = ledger.append_entry(finding)
        return signed_finding
    except ConnectionError as exc:
        logger.error("Ledger database connection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit ledger database is temporarily unreachable.",
        ) from exc
    except Exception as exc:
        logger.error("Failed to append finding to audit ledger: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to seal finding into audit ledger: {exc}",
        ) from exc


@app.get(
    "/findings/{ledger_id}",
    response_model=SignedFinding,
    summary="Retrieve a specific signed finding by ledger position ID",
)
async def get_finding(
    ledger_id: int,
    _user: UserIdentity | None = Depends(require_read_role),
) -> SignedFinding:
    """Retrieve an immutable SignedFinding record from the audit ledger by primary key ID."""
    if ledger_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ledger_id must be a positive integer.",
        )
    ledger = get_ledger()
    try:
        entry = ledger.get_entry_by_id(ledger_id)
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Finding with ledger_id {ledger_id} does not exist.",
            )
        return entry
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error retrieving ledger entry %s: %s", ledger_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query audit ledger.",
        ) from exc


@app.get(
    "/findings",
    response_model=list[SignedFinding],
    summary="List paginated signed findings with optional filters",
)
async def list_findings(
    severity: Severity | None = Query(default=None, description="Filter by finding severity"),
    asset_type: AssetType | None = Query(default=None, description="Filter by evaluation asset type"),
    limit: int = Query(default=50, ge=1, le=500, description="Max number of records to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    _user: UserIdentity | None = Depends(require_read_role),
) -> list[SignedFinding]:
    """Retrieve a paginated list of signed findings, filterable by severity and asset_type."""
    ledger = get_ledger()
    try:
        return ledger.list_entries(
            limit=limit,
            offset=offset,
            severity=severity,
            asset_type=asset_type,
        )
    except Exception as exc:
        logger.error("Error querying findings list: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query audit ledger entries.",
        ) from exc


@app.get(
    "/audit/verify",
    response_model=VerificationResult,
    summary="Verify cryptographic integrity and continuity of the entire audit chain",
)
async def verify_audit_chain(
    from_id: int | None = Query(
        default=None,
        ge=1,
        description="Optional ledger ID to start verification from (defaults to genesis)",
    ),
    _user: UserIdentity | None = Depends(require_read_role),
) -> VerificationResult:
    """Walk the audit ledger table, recomputing all SHA256 hashes and verifying all Ed25519 signatures."""
    ledger = get_ledger()
    try:
        result = ledger.verify_chain(from_id=from_id)
        return result
    except Exception as exc:
        logger.error("Error verifying audit chain: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification procedure encountered an error: {exc}",
        ) from exc


@app.get(
    "/findings/{ledger_id}/verify",
    summary="Verify cryptographic signature of an individual finding ledger entry",
)
async def verify_finding_signature(
    ledger_id: int,
    _user: UserIdentity | None = Depends(require_read_role),
) -> dict[str, Any]:
    """Verify Ed25519 signature and hash integrity for a single finding."""
    ledger = get_ledger()
    entry = ledger.get_entry_by_id(ledger_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Finding with ledger_id {ledger_id} not found.",
        )
    is_valid = ledger.verify_entry(entry)
    return {
        "ledger_id": ledger_id,
        "valid": is_valid,
        "entry_hash": entry.entry_hash,
        "prev_hash": entry.prev_hash,
    }


@app.get(
    "/coverage",
    summary="Retrieve canonical merged attack class coverage manifest",
)
async def get_coverage(
    _user: UserIdentity | None = Depends(require_read_role),
) -> dict[str, Any]:
    """Return the merged threat model and attack class coverage manifest."""
    manifest_data, manifest_hash = load_coverage_manifest()
    coverage = merge_coverage_statement(manifest_data, [])
    return {
        "manifest_hash": manifest_hash,
        "summary": manifest_data.get("summary", ""),
        "version": manifest_data.get("version", "0.6.0"),
        "coverage": coverage.model_dump(mode="json"),
    }


@app.get(
    "/reports/generate",
    response_model=Report,
    summary="Generate deterministic JSON integrity assurance report from a ledger position",
)
async def generate_report_json(
    since: int = Query(default=1, ge=1, description="Ledger ID starting position (inclusive)"),
    timestamp: str | None = Query(default=None, description="Optional explicit generation timestamp for determinism"),
    _user: UserIdentity | None = Depends(require_read_role),
) -> Report:
    """Pull SignedFindings from the ledger and generate a schema-validated Report object."""
    ledger = get_ledger()
    try:
        report = generate_report(ledger=ledger, since_id=since, explicit_timestamp=timestamp)
        return report
    except Exception as exc:
        logger.error("Failed to generate report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {exc}",
        ) from exc


@app.get(
    "/reports/generate.html",
    summary="Generate human-readable HTML integrity assurance report from a ledger position",
)
async def generate_report_html_endpoint(
    since: int = Query(default=1, ge=1, description="Ledger ID starting position (inclusive)"),
    timestamp: str | None = Query(default=None, description="Optional explicit generation timestamp"),
    _user: UserIdentity | None = Depends(require_read_role),
) -> Response:
    """Render the schema-validated Report as an HTML document using Jinja2."""
    ledger = get_ledger()
    try:
        report = generate_report(ledger=ledger, since_id=since, explicit_timestamp=timestamp)
        html_content = render_report_html(report)
        return Response(content=html_content, media_type="text/html")
    except Exception as exc:
        logger.error("Failed to render HTML report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HTML report rendering failed: {exc}",
        ) from exc


@app.get(
    "/reports/generate.pdf",
    summary="Generate downloadable PDF integrity assurance report from a ledger position",
)
async def generate_report_pdf_endpoint(
    since: int = Query(default=1, ge=1, description="Ledger ID starting position (inclusive)"),
    timestamp: str | None = Query(default=None, description="Optional explicit generation timestamp"),
    _user: UserIdentity | None = Depends(require_read_role),
) -> Response:
    """Render the schema-validated Report as a PDF document using WeasyPrint with fallback."""
    ledger = get_ledger()
    try:
        report = generate_report(ledger=ledger, since_id=since, explicit_timestamp=timestamp)
        pdf_bytes = render_report_pdf(report)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="cvguard-report-{report.report_id}.pdf"'},
        )
    except Exception as exc:
        logger.error("Failed to render PDF report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF report rendering failed: {exc}",
        ) from exc


if __name__ == "__main__":
    mtls = os.getenv("CVGUARD_MTLS_ENABLED", "false").lower() in ("true", "1", "yes")
    ssl_kwargs: dict[str, Any] = {}
    if mtls:
        cert_path = os.getenv("CVGUARD_CERT_PATH", "/etc/cvguard/certs/service.crt")
        key_path = os.getenv("CVGUARD_KEY_PATH", "/etc/cvguard/certs/service.key")
        ca_path = os.getenv("CVGUARD_CA_CERT_PATH", "/etc/cvguard/certs/ca.crt")
        if os.path.isfile(cert_path) and os.path.isfile(key_path) and os.path.isfile(ca_path):
            ssl_kwargs = {
                "ssl_certfile": cert_path,
                "ssl_keyfile": key_path,
                "ssl_ca_certs": ca_path,
                "ssl_cert_reqs": ssl.CERT_REQUIRED,
            }
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=False, **ssl_kwargs)
