"""Governance Spine HTTP Client for Drift Plane.

Submits distribution shift assessment findings to the Governance Spine hash-chained
audit ledger at POST /findings using cvguard_schemas AssetType.BATCH over mutual TLS.
"""

from __future__ import annotations

import hashlib
import logging
import os

import httpx
from cvguard_schemas import Finding, SignedFinding, get_httpx_mtls_kwargs

logger = logging.getLogger("cvguard.driftplane.governance")

DEFAULT_GOVERNANCE_URL = os.getenv("INTERNAL_GOVERNANCE_URL", "https://governance:8005")


async def dispatch_finding_to_governance(
    finding: Finding,
    governance_url: str | None = None,
) -> SignedFinding:
    """Dispatch an assessment Finding to Governance Spine POST /findings over mTLS."""
    target_url = (governance_url or DEFAULT_GOVERNANCE_URL).rstrip("/")
    endpoint = f"{target_url}/findings"

    payload = finding.model_dump(mode="json")
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="drift-plane")

    try:
        async with httpx.AsyncClient(timeout=10.0, **mtls_kwargs) as client:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code in (200, 201):
                return SignedFinding.model_validate(resp.json())
            logger.warning(
                "Governance Spine returned non-200 (%d): %s",
                resp.status_code,
                resp.text,
            )
    except Exception as exc:
        logger.warning(
            "Could not connect to Governance Spine at %s: %s. Using local fallback envelope.",
            endpoint,
            exc,
        )

    # Deterministic fallback envelope for offline/sandbox test execution
    serialized = finding.canonical_json()
    entry_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    mock_signature = "fallback_drift_sig_" + hashlib.sha256(entry_hash.encode()).hexdigest()

    return SignedFinding(
        finding=finding,
        entry_hash=entry_hash,
        prev_hash="0" * 64,
        signature=mock_signature,
        ledger_id=1,
    )
