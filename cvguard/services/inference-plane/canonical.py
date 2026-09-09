"""Canonical serialization and cryptographic hashing utilities for CVGuard Inference Plane.

Provides deterministic cryptographic binding between:
1. Input image bytes (input_hash = SHA-256(image_bytes))
2. Model identity reference (model_id = ingested model weight digest / identifier)
3. Preprocessing / inference config (config_hash = SHA-256(canonical_json(config)))
4. Model prediction output (output_hash = SHA-256(canonical_json(output)))
5. Atomically-enforced monotonic sequence number (sequence_number)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize any JSON-compatible structure into deterministic, compact canonical JSON.

    Rules:
    - Dict keys sorted lexicographically
    - Compact delimiters: separators=(',', ':')
    - No extraneous whitespace
    - UTF-8 compatible
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(data: Any) -> bytes:
    """Encode canonical JSON string into UTF-8 bytes."""
    return canonical_json(data).encode("utf-8")


def compute_sha256(data: bytes | str) -> str:
    """Compute SHA256 hex digest for arbitrary bytes or string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_input_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hex digest of raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def compute_config_hash(config: dict[str, Any] | str) -> str:
    """Compute SHA-256 hex digest of canonical JSON config."""
    if isinstance(config, str):
        try:
            parsed = json.loads(config)
            return compute_sha256(canonical_json_bytes(parsed))
        except Exception:
            return compute_sha256(config.encode("utf-8"))
    return compute_sha256(canonical_json_bytes(config))


def compute_output_hash(output: dict[str, Any] | list[Any] | str) -> str:
    """Compute SHA-256 hex digest of canonical JSON model output."""
    if isinstance(output, (dict, list)):
        return compute_sha256(canonical_json_bytes(output))
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            return compute_sha256(canonical_json_bytes(parsed))
        except Exception:
            return compute_sha256(output.encode("utf-8"))
    return compute_sha256(str(output).encode("utf-8"))


def canonical_record_payload(
    record_id: str,
    model_id: str,
    sequence_number: int,
    input_hash: str,
    config_hash: str,
    output_hash: str,
) -> dict[str, Any]:
    """Construct deterministic dictionary binding all inference provenance components."""
    return {
        "config_hash": config_hash,
        "input_hash": input_hash,
        "model_id": model_id,
        "output_hash": output_hash,
        "record_id": record_id,
        "sequence_number": sequence_number,
    }


def compute_record_hash(
    record_id: str,
    model_id: str,
    sequence_number: int,
    input_hash: str,
    config_hash: str,
    output_hash: str,
) -> str:
    """Compute cryptographic binding hash of an inference record.

    record_hash = SHA-256(canonical_json({
        "config_hash": config_hash,
        "input_hash": input_hash,
        "model_id": model_id,
        "output_hash": output_hash,
        "record_id": record_id,
        "sequence_number": sequence_number
    }))
    """
    payload = canonical_record_payload(
        record_id=record_id,
        model_id=model_id,
        sequence_number=sequence_number,
        input_hash=input_hash,
        config_hash=config_hash,
        output_hash=output_hash,
    )
    return compute_sha256(canonical_json_bytes(payload))
