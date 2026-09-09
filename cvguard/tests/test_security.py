"""Security Test Suite for CVGuard Phase 1 Governance Spine.

Tests the cryptographic integrity of the Ed25519-signed hash chain and verifies
that database-level permissions prevent tampering at the storage engine layer.

Required Security Tests:
1. test_append_and_verify_chain_is_valid_on_clean_data
2. test_tampering_with_a_payload_is_detected
3. test_forged_signature_is_rejected
4. test_ledger_role_cannot_update_or_delete
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator
import pytest

from cryptography.hazmat.primitives.asymmetric import ed25519
import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from canonical import GENESIS_HASH, canonical_chain_payload
from cvguard_schemas import AssetType, Disposition, Finding, Severity
from ledger import AuditLedger, PostgresConfig
from signer import LocalFileSigner


def get_test_db_host() -> str:
    """Resolve database host for test environments (defaults to localhost for workstation test runs)."""
    return os.getenv("TEST_POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost"))


def get_test_db_port() -> int:
    return int(os.getenv("TEST_POSTGRES_PORT", os.getenv("POSTGRES_PORT", "5432")))


def get_admin_config() -> PostgresConfig:
    """Connection configuration using admin credentials (used for test setup/teardown and simulated attacks)."""
    return PostgresConfig(
        host=get_test_db_host(),
        port=get_test_db_port(),
        dbname=os.getenv("POSTGRES_DB", "cvguard"),
        user=os.getenv("POSTGRES_ADMIN_USER", os.getenv("POSTGRES_USER", "cvguard_admin")),
        password=os.getenv("POSTGRES_ADMIN_PASSWORD", os.getenv("POSTGRES_PASSWORD", "cvguard_dev_secret_change_me_in_prod")),
    )


def get_ledger_config() -> PostgresConfig:
    """Connection configuration using restricted append-only role credentials (cvguard_ledger)."""
    return PostgresConfig(
        host=get_test_db_host(),
        port=get_test_db_port(),
        dbname=os.getenv("POSTGRES_DB", "cvguard"),
        user=os.getenv("POSTGRES_LEDGER_USER", "cvguard_ledger"),
        password=os.getenv("POSTGRES_LEDGER_PASSWORD", "cvguard_ledger_secret_change_me_in_prod"),
    )


@pytest.fixture(scope="session")
def check_postgres_available() -> None:
    """Check that PostgreSQL is running before executing live database security tests."""
    admin_cfg = get_admin_config()
    try:
        with psycopg.connect(admin_cfg.conninfo(), connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL is not accessible at {admin_cfg.host}:{admin_cfg.port} ({exc}). "
            "Please start the infrastructure stack with 'make up' or 'docker compose up' "
            "before executing Phase 1 security tests."
        )


@pytest.fixture(autouse=True)
def clean_audit_log(check_postgres_available: None) -> Generator[None, None, None]:
    """Ensure the audit_log table is clean and identity sequence reset before each test."""
    admin_cfg = get_admin_config()
    with psycopg.connect(admin_cfg.conninfo()) as conn:
        with conn.cursor() as cur:
            # Clean all rows and reset auto-increment primary key ID sequence to 1
            cur.execute("TRUNCATE TABLE audit_log RESTART IDENTITY CASCADE;")
        conn.commit()

    yield

    with psycopg.connect(admin_cfg.conninfo()) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE audit_log RESTART IDENTITY CASCADE;")
        conn.commit()


@pytest.fixture
def test_signer(tmp_path: Path) -> LocalFileSigner:
    """Create an isolated test Ed25519 signer using a temporary directory."""
    key_file = tmp_path / "test_governance_ed25519.pem"
    return LocalFileSigner(key_path=key_file)


@pytest.fixture
def test_ledger(test_signer: LocalFileSigner) -> AuditLedger:
    """Create an AuditLedger configured with the restricted role credentials and test signer."""
    ledger_cfg = get_ledger_config()
    return AuditLedger(config=ledger_cfg, signer=test_signer)


def create_sample_finding(idx: int, asset_type: AssetType, severity: Severity) -> Finding:
    """Helper to produce diverse deterministic finding instances for test assertions."""
    return Finding(
        asset_type=asset_type,
        asset_ref=f"sha256:4a35b00{idx}/sample_artifact_{idx}.onnx",
        detector=f"cvguard.detector.test_integrity:v{idx}.0",
        reason=f"Assurance scan index {idx} evaluated with status pass/flag.",
        evidence=[f"metric_{idx}:0.42", f"threshold_{idx}:0.80"],
        confidence=0.85 + (idx * 0.02),
        severity=severity,
        disposition=Disposition.REVIEW if severity == Severity.MEDIUM else Disposition.QUARANTINE,
        assumptions=[f"Assumption rule {idx}"],
        limitations=[f"Limitation rule {idx}"],
    )


# ==============================================================================
# 1. Clean Data Hash-Chain Verification Test
# ==============================================================================
def test_append_and_verify_chain_is_valid_on_clean_data(test_ledger: AuditLedger) -> None:
    """Validate that sequential clean appends build a valid cryptographic hash chain."""
    f1 = create_sample_finding(1, AssetType.MODEL, Severity.LOW)
    f2 = create_sample_finding(2, AssetType.SAMPLE, Severity.MEDIUM)
    f3 = create_sample_finding(3, AssetType.INFERENCE_RECORD, Severity.CRITICAL)

    # Append entries sequentially through the ledger abstraction
    signed1 = test_ledger.append_entry(f1)
    signed2 = test_ledger.append_entry(f2)
    signed3 = test_ledger.append_entry(f3)

    # Assert returned SignedFinding envelopes have populated ledger IDs
    assert signed1.ledger_id == 1
    assert signed2.ledger_id == 2
    assert signed3.ledger_id == 3

    # Assert genesis and chained link integrity
    assert signed1.prev_hash == GENESIS_HASH
    assert signed2.prev_hash == signed1.entry_hash
    assert signed3.prev_hash == signed2.entry_hash

    # Assert signatures are 128-hex-char Ed25519 signatures
    for s in [signed1, signed2, signed3]:
        assert len(s.signature) == 128
        assert len(s.entry_hash) == 64

    # Run chain verification across all rows
    result = test_ledger.verify_chain()

    assert result.valid is True, f"Expected clean chain to be valid, but got: {result.reason}"
    assert result.first_invalid_entry_id is None
    assert result.entries_checked == 3


# ==============================================================================
# 2. Out-of-Band Database Payload Tampering Detection Test
# ==============================================================================
def test_tampering_with_a_payload_is_detected(test_ledger: AuditLedger) -> None:
    """Assert that directly modifying a row's payload in PostgreSQL is immediately caught by verify_chain."""
    f1 = create_sample_finding(1, AssetType.MODEL, Severity.LOW)
    f2 = create_sample_finding(2, AssetType.SAMPLE, Severity.HIGH)
    f3 = create_sample_finding(3, AssetType.BATCH, Severity.INFO)

    test_ledger.append_entry(f1)
    test_ledger.append_entry(f2)
    test_ledger.append_entry(f3)

    # Verify clean state before attack
    initial_verification = test_ledger.verify_chain()
    assert initial_verification.valid is True

    # Simulate an adversary directly modifying row #2 in the database bypassing the application layer
    admin_cfg = get_admin_config()
    with psycopg.connect(admin_cfg.conninfo(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Fetch existing payload for row 2
            cur.execute("SELECT payload FROM audit_log WHERE id = 2;")
            row = cur.fetchone()
            assert row is not None
            tampered_payload = row["payload"]

            # Malicious alteration: downgrade severity and change confidence
            tampered_payload["confidence"] = 0.05
            tampered_payload["reason"] = "TAMPERED: Malicious actor altered the evaluation reason."

            # Update row 2 directly via SQL
            cur.execute(
                "UPDATE audit_log SET payload = %s::jsonb WHERE id = 2;",
                (json.dumps(tampered_payload),),
            )
        conn.commit()

    # Execute chain verification
    tampered_verification = test_ledger.verify_chain()

    # The ledger verification MUST fail and accurately point to entry ID 2
    assert tampered_verification.valid is False
    assert tampered_verification.first_invalid_entry_id == 2
    assert tampered_verification.entries_checked == 1  # Entry 1 passed, failed on entry 2
    assert "tampering detected" in (tampered_verification.reason or "").lower()


# ==============================================================================
# 3. Forged Cryptographic Signature Detection Test
# ==============================================================================
def test_forged_signature_is_rejected(test_ledger: AuditLedger) -> None:
    """Assert that substituting an entry's signature with one from an unauthorized keypair is caught."""
    f1 = create_sample_finding(1, AssetType.MODEL, Severity.LOW)
    f2 = create_sample_finding(2, AssetType.SOURCE, Severity.CRITICAL)

    s1 = test_ledger.append_entry(f1)
    s2 = test_ledger.append_entry(f2)

    # Generate a separate rogue Ed25519 keypair
    rogue_private_key = ed25519.Ed25519PrivateKey.generate()

    # Craft a forged signature covering entry 2's canonical (entry_hash + prev_hash)
    chain_payload_bytes = canonical_chain_payload(s2.entry_hash, s2.prev_hash)
    forged_signature = rogue_private_key.sign(chain_payload_bytes).hex()

    # Directly inject the forged signature into row 2 using admin credentials
    admin_cfg = get_admin_config()
    with psycopg.connect(admin_cfg.conninfo()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audit_log SET signature = %s WHERE id = 2;",
                (forged_signature,),
            )
        conn.commit()

    # Run chain verification
    verification = test_ledger.verify_chain()

    assert verification.valid is False
    assert verification.first_invalid_entry_id == 2
    assert "signature verification failed" in (verification.reason or "").lower()


# ==============================================================================
# 4. Storage Engine Permission Enforcement Test (Restricted Role)
# ==============================================================================
def test_ledger_role_cannot_update_or_delete(test_ledger: AuditLedger) -> None:
    """Assert that PostgreSQL itself rejects UPDATE and DELETE queries when executed by cvguard_ledger."""
    f = create_sample_finding(1, AssetType.MODEL, Severity.LOW)
    test_ledger.append_entry(f)

    # Connect to PostgreSQL directly as the restricted role: cvguard_ledger
    ledger_cfg = get_ledger_config()
    with psycopg.connect(ledger_cfg.conninfo()) as conn:
        with conn.cursor() as cur:
            # 1. SELECT is allowed
            cur.execute("SELECT id, entry_hash FROM audit_log WHERE id = 1;")
            row = cur.fetchone()
            assert row is not None

            # 2. UPDATE must be rejected at the PostgreSQL engine level
            with pytest.raises(errors.InsufficientPrivilege) as exc_update:
                cur.execute("UPDATE audit_log SET entry_hash = 'malicious_hash' WHERE id = 1;")
            assert "permission denied for table audit_log" in str(exc_update.value).lower()

        # Roll back failed transaction state to test DELETE
        conn.rollback()

        with conn.cursor() as cur:
            # 3. DELETE must be rejected at the PostgreSQL engine level
            with pytest.raises(errors.InsufficientPrivilege) as exc_delete:
                cur.execute("DELETE FROM audit_log WHERE id = 1;")
            assert "permission denied for table audit_log" in str(exc_delete.value).lower()

        conn.rollback()

        with conn.cursor() as cur:
            # 4. TRUNCATE must also be rejected
            with pytest.raises(errors.InsufficientPrivilege) as exc_truncate:
                cur.execute("TRUNCATE TABLE audit_log;")
            assert "permission denied for table audit_log" in str(exc_truncate.value).lower()

        conn.rollback()

    # Verify that the record remains intact and unaltered
    admin_cfg = get_admin_config()
    with psycopg.connect(admin_cfg.conninfo(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, entry_hash FROM audit_log WHERE id = 1;")
            record = cur.fetchone()
            assert record is not None
            assert record["id"] == 1
            assert len(record["entry_hash"]) == 64
