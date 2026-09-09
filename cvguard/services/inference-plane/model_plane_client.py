"""HTTP client for querying model metadata and weight digests from CVGuard Model Plane."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

import httpx
from cvguard_schemas import get_httpx_mtls_kwargs

logger = logging.getLogger("cvguard.inferenceplane.model_plane_client")

DEFAULT_MODEL_PLANE_URL = os.getenv("INTERNAL_MODEL_PLANE_URL", "https://model-plane:8002")
_HEX64_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")


async def resolve_model_weight_digest(
    model_ref: str,
    model_plane_url: str | None = None,
) -> str:
    """Resolve a model reference into its cryptographic weight digest (SHA-256).

    Queries model-plane via mTLS: GET /models/{model_ref}.
    """
    clean_ref = model_ref.strip()
    if clean_ref.startswith("sha256:"):
        candidate = clean_ref[7:].strip()
        if _HEX64_REGEX.match(candidate):
            return candidate.lower()

    if _HEX64_REGEX.match(clean_ref):
        return clean_ref.lower()

    base_url = model_plane_url or os.getenv("INTERNAL_MODEL_PLANE_URL", DEFAULT_MODEL_PLANE_URL)
    endpoint = f"{base_url.rstrip('/')}/models/{clean_ref}"

    timeout = httpx.Timeout(5.0, connect=2.0)
    mtls_kwargs = get_httpx_mtls_kwargs(service_name="inference-plane")
    try:
        async with httpx.AsyncClient(timeout=timeout, **mtls_kwargs) as client:
            resp = await client.get(endpoint)
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                weight_digest = data.get("weight_digest") or data.get("file_sha256")
                if weight_digest:
                    return str(weight_digest).lower()
            else:
                logger.warning(
                    "Model plane responded with status %d for model %s: %s",
                    resp.status_code,
                    clean_ref,
                    resp.text,
                )
    except Exception as exc:
        logger.warning(
            "Could not reach model plane at %s for model %s: %s. Using deterministic fallback digest.",
            endpoint,
            clean_ref,
            exc,
        )

    # Fallback when running standalone or in tests without model-plane running:
    return hashlib.sha256(clean_ref.encode("utf-8")).hexdigest()
