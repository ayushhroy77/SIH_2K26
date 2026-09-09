"""Tamper-evident PostgreSQL audit ledger for CVGuard Governance.

Maintains an append-only, cryptographically hash-chained audit ledger of all
evaluation findings across CVGuard planes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field

from canonical import (
    GENESIS_HASH,
    canonical_chain_payload,
    compute_entry_hash,
)
from cvguard_schemas import AssetType, Finding, Severity, SignedFinding
from signer import Signer

logger = logging.getLogger("cvguard.governance.ledger")


class VerificationResult(BaseModel):
    """Result of audit ledger hash-chain and cryptographic signature validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool = Field(..., description="True if all checked entries passed cryptographic and chain verification.")
    first_invalid_entry_id: int | None = Field(
        default=None,
        description="Database primary key (id) of the first corrupted or invalid ledger entry.",
    )
    entries_checked: int = Field(
        default=0,
        description="Total number of consecutive ledger records inspected.",
    )
    reason: str | None = Field(
        default=None,
        description="Specific verification failure description (e.g. hash mismatch, signature invalid).",
    )


@dataclass(frozen=True)
class PostgresConfig:
    """Connection parameters for PostgreSQL."""

    host: str = "postgres"
    port: int = 5432
    dbname: str = "cvguard"
    user: str = "cvguard_ledger"
    password: str = "cvguard_ledger_secret_change_me_in_prod"

    @classmethod
    def from_env(cls, prefix: str = "POSTGRES_") -> PostgresConfig:
        """Load connection config from environment variables."""
        return cls(
            host=os.getenv(f"{prefix}HOST", os.getenv("POSTGRES_HOST", "postgres")),
            port=int(os.getenv(f"{prefix}PORT", os.getenv("POSTGRES_PORT", "5432"))),
            dbname=os.getenv(f"{prefix}DB", os.getenv("POSTGRES_DB", "cvguard")),
            user=os.getenv(f"{prefix}USER", os.getenv("POSTGRES_USER", "cvguard_ledger")),
            password=os.getenv(f"{prefix}PASSWORD", os.getenv("POSTGRES_PASSWORD", "cvguard_ledger_secret_change_me_in_prod")),
        )

    def conninfo(self) -> str:
        """Format connection string."""
        return f"host={self.host} port={self.port} dbname={self.dbname} user={self.user} password={self.password}"


class AuditLedger:
    """PostgreSQL-backed tamper-evident audit ledger with Ed25519 signature binding."""

    def __init__(self, config: PostgresConfig | None = None, signer: Signer | None = None) -> None:
        self.config = config or PostgresConfig.from_env()
        self.signer = signer

    def _get_connection(self) -> psycopg.Connection[Any]:
        """Establish a new connection using the configured role credentials."""
        try:
            return psycopg.connect(self.config.conninfo(), row_factory=dict_row)
        except Exception as exc:
            logger.error("Failed to connect to PostgreSQL at %s:%s: %s", self.config.host, self.config.port, exc)
            raise ConnectionError(f"Database connection error: {exc}") from exc

    def append_entry(self, finding: Finding) -> SignedFinding:
        """Append a new Finding to the audit ledger.

        Workflow:
        1. Acquire exclusive advisory lock to prevent concurrent forks in chain sequence.
        2. Retrieve entry_hash of latest row (or GENESIS_HASH if table is empty).
        3. Compute SHA256 entry_hash = SHA256(prev_hash || canonical_json(finding)).
        4. Sign canonical_chain_payload(entry_hash, prev_hash) using Ed25519 signer.
        5. Insert row into audit_log using restricted ledger credentials.
        6. Return SignedFinding envelope populated with database ledger_id.
        """
        if self.signer is None:
            raise RuntimeError("AuditLedger cannot append entries without an initialized Signer.")

        # Serialize finding to canonical JSON dict for database insertion
        finding_payload_dict = finding.model_dump(mode="json")
        finding_payload_json_str = json.dumps(finding_payload_dict)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Use PostgreSQL transaction-level advisory lock to serialize appends
                cur.execute("SELECT pg_advisory_xact_lock(482910482);")

                # Fetch predecessor hash
                cur.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1;")
                latest = cur.fetchone()
                prev_hash = latest["entry_hash"] if latest else GENESIS_HASH

                # Compute deterministic entry hash
                entry_hash = compute_entry_hash(prev_hash, finding)

                # Sign canonical (entry_hash + prev_hash)
                chain_bytes = canonical_chain_payload(entry_hash, prev_hash)
                signature = self.signer.sign(chain_bytes)

                # Insert immutable row using restricted credentials
                cur.execute(
                    """
                    INSERT INTO audit_log (entry_hash, prev_hash, payload, signature)
                    VALUES (%s, %s, %s::jsonb, %s)
                    RETURNING id;
                    """,
                    (entry_hash, prev_hash, finding_payload_json_str, signature),
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Failed to insert record into audit_log table.")
                ledger_id = int(row["id"])

            conn.commit()

        return SignedFinding(
            finding=finding,
            entry_hash=entry_hash,
            prev_hash=prev_hash,
            signature=signature,
            ledger_id=ledger_id,
        )

    def get_entry_by_id(self, ledger_id: int) -> SignedFinding | None:
        """Retrieve a single signed finding by its database ledger ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, entry_hash, prev_hash, payload, signature
                    FROM audit_log
                    WHERE id = %s;
                    """,
                    (ledger_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                finding = Finding.model_validate(row["payload"])
                return SignedFinding(
                    finding=finding,
                    entry_hash=row["entry_hash"],
                    prev_hash=row["prev_hash"],
                    signature=row["signature"],
                    ledger_id=int(row["id"]),
                )

    def list_entries(
        self,
        limit: int = 50,
        offset: int = 0,
        severity: Severity | None = None,
        asset_type: AssetType | None = None,
    ) -> list[SignedFinding]:
        """Retrieve paginated findings with optional filtering by severity or asset_type."""
        query = "SELECT id, entry_hash, prev_hash, payload, signature FROM audit_log WHERE 1=1"
        params: list[Any] = []

        if severity is not None:
            query += " AND payload->>'severity' = %s"
            params.append(severity.value)

        if asset_type is not None:
            query += " AND payload->>'asset_type' = %s"
            params.append(asset_type.value)

        query += " ORDER BY id ASC LIMIT %s OFFSET %s;"
        params.extend([limit, offset])

        results: list[SignedFinding] = []
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                for row in rows:
                    finding = Finding.model_validate(row["payload"])
                    results.append(
                        SignedFinding(
                            finding=finding,
                            entry_hash=row["entry_hash"],
                            prev_hash=row["prev_hash"],
                            signature=row["signature"],
                            ledger_id=int(row["id"]),
                        )
                    )
        return results

    def verify_chain(self, from_id: int | None = None) -> VerificationResult:
        """Walk the audit ledger table, recomputing every hash and validating every signature.

        Detects:
        - Payload tampering (hash mismatch)
        - Broken chain continuity (prev_hash mismatch)
        - Forged or invalid signatures
        - Deleted intermediate rows
        """
        if self.signer is None:
            raise RuntimeError("AuditLedger cannot verify signatures without an initialized Signer.")

        start_id = from_id if from_id is not None else 1
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # If from_id > 1, retrieve predecessor's entry_hash to verify link
                expected_prev_hash: str | None = None
                if start_id > 1:
                    cur.execute("SELECT entry_hash FROM audit_log WHERE id < %s ORDER BY id DESC LIMIT 1;", (start_id,))
                    pred = cur.fetchone()
                    if pred:
                        expected_prev_hash = pred["entry_hash"]
                else:
                    expected_prev_hash = GENESIS_HASH

                cur.execute(
                    """
                    SELECT id, entry_hash, prev_hash, payload, signature
                    FROM audit_log
                    WHERE id >= %s
                    ORDER BY id ASC;
                    """,
                    (start_id,),
                )
                rows = cur.fetchall()

        if not rows:
            return VerificationResult(
                valid=True,
                first_invalid_entry_id=None,
                entries_checked=0,
                reason="Ledger contains no entries to verify.",
            )

        checked_count = 0
        current_expected_prev = expected_prev_hash

        for row in rows:
            entry_id = int(row["id"])
            entry_hash = str(row["entry_hash"])
            prev_hash = str(row["prev_hash"])
            payload_data = row["payload"]
            signature = str(row["signature"])

            # 1. Verify chain continuity
            if current_expected_prev is not None and prev_hash != current_expected_prev:
                return VerificationResult(
                    valid=False,
                    first_invalid_entry_id=entry_id,
                    entries_checked=checked_count,
                    reason=(
                        f"Hash chain broken at entry ID {entry_id}: recorded prev_hash '{prev_hash}' "
                        f"does not match expected predecessor hash '{current_expected_prev}'."
                    ),
                )

            # 2. Re-validate Finding model and recompute SHA256 entry hash
            try:
                finding = Finding.model_validate(payload_data)
            except Exception as exc:
                return VerificationResult(
                    valid=False,
                    first_invalid_entry_id=entry_id,
                    entries_checked=checked_count,
                    reason=f"Payload failed Pydantic validation at entry ID {entry_id}: {exc}",
                )

            recomputed_hash = compute_entry_hash(prev_hash, finding)
            if recomputed_hash != entry_hash:
                return VerificationResult(
                    valid=False,
                    first_invalid_entry_id=entry_id,
                    entries_checked=checked_count,
                    reason=(
                        f"Payload tampering detected at entry ID {entry_id}: recorded entry_hash '{entry_hash}' "
                        f"does not match recomputed hash '{recomputed_hash}'."
                    ),
                )

            # 3. Verify cryptographic Ed25519 signature
            chain_payload_bytes = canonical_chain_payload(entry_hash, prev_hash)
            is_valid_sig = self.signer.verify(chain_payload_bytes, signature)
            if not is_valid_sig:
                return VerificationResult(
                    valid=False,
                    first_invalid_entry_id=entry_id,
                    entries_checked=checked_count,
                    reason=f"Signature verification failed for entry ID {entry_id}: signature is forged or corrupted.",
                )

            # Advance expected previous hash to current entry hash
            current_expected_prev = entry_hash
            checked_count += 1

        return VerificationResult(
            valid=True,
            first_invalid_entry_id=None,
            entries_checked=checked_count,
            reason=f"All {checked_count} ledger entries verified successfully.",
        )
