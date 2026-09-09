"""CVGuard Governance Spine Service.

Phase 1: Cryptographic Ed25519 signer and tamper-evident PostgreSQL audit ledger.
All evaluation planes submit findings here to receive cryptographically signed,
hash-chained assurance receipts.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

import cvguard_schemas
from cvguard_schemas import AssetType, Finding, Severity, SignedFinding
from ledger import AuditLedger, PostgresConfig, VerificationResult
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle handler."""
    logger.info("Initializing CVGuard Governance Spine Service...")
    # Pre-initialize cryptographic signer and audit ledger
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
    description="Tamper-evident audit ledger, Ed25519 finding signer, and assurance verification spine.",
    version="0.2.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    """Pydantic v2 health status schema."""

    status: str = Field(default="ok", description="Operational status flag.")
    service: str = Field(default="governance", description="Name of the reporting service.")
    version: str = Field(default="0.2.0", description="Service semantic version.")
    schemas_version: str = Field(
        default=cvguard_schemas.__version__,
        description="Version of cvguard_schemas linked to this service runtime.",
    )
    signer_public_key: str = Field(
        ...,
        description="Hex-encoded 32-byte Ed25519 public key of the governance signer.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness/liveness health check probe endpoint."""
    signer = get_signer()
    return HealthResponse(
        status="ok",
        service="governance",
        version="0.2.0",
        schemas_version=cvguard_schemas.__version__,
        signer_public_key=signer.get_public_key_hex(),
    )


@app.post(
    "/findings",
    response_model=SignedFinding,
    status_code=status.HTTP_201_CREATED,
    summary="Submit and cryptographically seal an evaluation finding",
)
async def create_finding(finding: Finding) -> SignedFinding:
    """Accept an evaluation Finding, hash-chain it into the audit ledger, and sign it with Ed25519.

    Rejects any payload that fails canonical schema validation (no type coercion).
    Fails explicitly (500) if database persistence or cryptographic signing fails.
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
async def get_finding(ledger_id: int) -> SignedFinding:
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
) -> VerificationResult:
    """Walk the audit ledger table, recomputing all SHA256 hashes and verifying all Ed25519 signatures.

    Returns:
        VerificationResult detailing validity, total entries checked, and the first corrupted entry ID (if any).
    """
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=False)
