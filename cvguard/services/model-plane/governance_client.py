"""HTTP client for dispatching model findings to the CVGuard Governance Spine."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from cvguard_schemas import Finding, SignedFinding

logger = logging.getLogger("cvguard.modelplane.governance_client")

DEFAULT_GOVERNANCE_URL = os.getenv("INTERNAL_GOVERNANCE_URL", "http://governance:8005")


class GovernanceDispatchError(RuntimeError):
    """Raised when a model finding cannot be sealed into the Governance Spine after retries."""


async def dispatch_finding_to_governance(
    finding: Finding,
    governance_url: str | None = None,
) -> SignedFinding:
    """Post a Finding to the Governance Spine /findings endpoint with strict retry semantics.

    Planes stay decoupled (no direct DB writes). Fails loudly if unsealed — never drops a finding.
    """
    base_url = governance_url or os.getenv("INTERNAL_GOVERNANCE_URL", DEFAULT_GOVERNANCE_URL)
    endpoint = f"{base_url.rstrip('/')}/findings"
    payload = finding.model_dump(mode="json")

    last_error: str | None = None
    timeout = httpx.Timeout(10.0, connect=5.0)

    # Retry once (total 2 attempts): attempt 0 (initial), attempt 1 (retry)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(2):
            try:
                response = await client.post(endpoint, json=payload)
                if response.status_code == 201:
                    signed_finding_data = response.json()
                    signed_finding = SignedFinding.model_validate(signed_finding_data)
                    logger.info(
                        "Model Finding %s sealed in Governance Ledger ID %s",
                        finding.finding_id,
                        signed_finding.ledger_id,
                    )
                    return signed_finding
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(
                        "Attempt %d: Governance Spine returned error: %s",
                        attempt + 1,
                        last_error,
                    )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Attempt %d: Connection to Governance Spine (%s) failed: %s",
                    attempt + 1,
                    endpoint,
                    last_error,
                )

            if attempt == 0:
                await asyncio.sleep(0.5)  # Brief backoff before retry

    # Fail loudly per CVGuard assurance specification
    raise GovernanceDispatchError(
        f"CRITICAL: Failed to seal finding {finding.finding_id} into Governance Spine "
        f"at {endpoint} after retry. Error: {last_error}"
    )
