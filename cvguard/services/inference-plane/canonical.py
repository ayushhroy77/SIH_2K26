"""Canonical serialization and cryptographic hashing utilities for CVGuard Inference Plane.

Phase 5: Verifiable cryptographic binding between:
1. raw input image bytes (input_hash = SHA-256(image_bytes))
2. model identity (model_id = weight digest of the actual model file/weights)
3. preprocessing & inference config (config_hash = SHA-256(canonical_json(config)))
4. prediction output (output)
5. ISO-8601 timestamp (timestamp)
6. atomically-enforced monotonic sequence number (monotonic_sequence_no)
7. cryptographic entropy nonce (nonce)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize any JSON-compatible structure into deterministic, compact canonical JSON.

    Rules:
    - Dict keys sorted lexicographically at all levels
    - Compact delimiters: separators=(',', ':')
    - No extraneous whitespace
    - ensure_ascii=False for standard UTF-8 representations
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(data: Any) -> bytes:
    """Encode canonical JSON string into UTF-8 bytes."""
    return canonical_json(data).encode("utf-8")


def compute_sha256(data: bytes | str) -> str:
    """Compute 64-character lowercase hexadecimal SHA-256 digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_input_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hex digest of raw image bytes (binds to exact image)."""
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


def compute_output_hash(output: Any) -> str:
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
    input_hash: str,
    model_id: str,
    config_hash: str,
    output: Any,
    timestamp: str | float,
    monotonic_sequence_no: int,
    nonce: str,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Construct deterministic dictionary binding all inference provenance components.

    Record schema:
    {
        "config_hash": <SHA-256 of canonical preprocessing + inference config>,
        "input_hash": <SHA-256 of raw image bytes>,
        "model_id": <model weight digest>,
        "monotonic_sequence_no": <strictly monotonic sequence number>,
        "nonce": <unique cryptographic entropy nonce>,
        "output": <model prediction output structure>,
        "timestamp": <ISO-8601 or float timestamp>
    }
    """
    payload: dict[str, Any] = {
        "config_hash": config_hash,
        "input_hash": input_hash,
        "model_id": model_id,
        "monotonic_sequence_no": monotonic_sequence_no,
        "nonce": nonce,
        "output": output,
        "timestamp": timestamp,
    }
    if record_id is not None:
        payload["record_id"] = record_id
    return payload


def compute_record_hash(
    input_hash: str,
    model_id: str,
    config_hash: str,
    output: Any,
    timestamp: str | float,
    monotonic_sequence_no: int,
    nonce: str,
    record_id: str | None = None,
) -> str:
    """Compute cryptographic binding hash (leaf hash) of an inference record.

    record_hash = SHA-256(canonical_json({
        "config_hash": config_hash,
        "input_hash": input_hash,
        "model_id": model_id,
        "monotonic_sequence_no": monotonic_sequence_no,
        "nonce": nonce,
        "output": output,
        "timestamp": timestamp
    }))
    """
    payload = canonical_record_payload(
        input_hash=input_hash,
        model_id=model_id,
        config_hash=config_hash,
        output=output,
        timestamp=timestamp,
        monotonic_sequence_no=monotonic_sequence_no,
        nonce=nonce,
        record_id=record_id,
    )
    return compute_sha256(canonical_json_bytes(payload))
