"""Canonical serialization and hashing utilities for CVGuard Governance.

This module provides the SINGLE SOURCE OF TRUTH for:
1. Canonical JSON bytes signed by the Ed25519 signer: canonical_chain_payload(entry_hash, prev_hash)
2. Canonical JSON bytes of Finding payloads hashed into the audit ledger: canonical_finding_bytes(finding)
3. Deterministic SHA256 entry hash calculation: compute_entry_hash(prev_hash, finding)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cvguard_schemas import Finding

# Fixed 64-character hex genesis digest for the initial root entry of the audit chain
GENESIS_HASH: str = "0000000000000000000000000000000000000000000000000000000000000000"


def canonical_chain_payload(entry_hash: str, prev_hash: str) -> bytes:
    """Canonical JSON serialization of (entry_hash + prev_hash).

    CRITICAL SECURITY SPECIFICATION:
    The bytes produced by this function represent the exact payload signed and verified
    by the Ed25519 signer. It intentionally binds the current entry hash to the previous
    chain hash, mathematically sealing the order and continuity of the ledger.

    Rules:
    - Keys are strictly sorted in lexicographical order: ("entry_hash", "prev_hash")
    - Compact JSON formatting with no whitespace: separators=(',', ':')
    - Encoded as UTF-8 bytes
    """
    envelope = {
        "entry_hash": entry_hash,
        "prev_hash": prev_hash,
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_finding_bytes(finding: Finding) -> bytes:
    """Deterministic canonical JSON serialization of a Finding model.

    Rules:
    - Serialized via Pydantic v2 `model_dump(mode='json')` to normalize datetimes and enums
    - Keys are sorted lexicographically
    - Compact JSON delimiters with no extraneous whitespace: separators=(',', ':')
    - Encoded as UTF-8 bytes
    """
    finding_dict: dict[str, Any] = finding.model_dump(mode="json")
    return json.dumps(finding_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_entry_hash(prev_hash: str, finding: Finding) -> str:
    """Compute SHA256 entry hash = SHA256(prev_hash || canonical_json(payload)).

    Parameters:
        prev_hash: The 64-char hex SHA256 hash of the immediate predecessor ledger entry
                   (or GENESIS_HASH if first entry).
        finding: The immutable Finding domain object to seal.

    Returns:
        64-character lowercase hex string representing the cryptographic digest.
    """
    payload_bytes = canonical_finding_bytes(finding)
    hasher = hashlib.sha256()
    hasher.update(prev_hash.encode("utf-8"))
    hasher.update(payload_bytes)
    return hasher.hexdigest()
