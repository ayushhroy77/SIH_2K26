"""PostgreSQL persistence layer and replay-resistant sequence enforcement for CVGuard Inference Plane.

Phase 5: Atomically increments and enforces monotonic sequence numbers per model identity
in PostgreSQL to prevent replay attacks at write time.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger("cvguard.inferenceplane.db")

# Thread lock for in-memory fallback state to guarantee atomic sequence increments and replay checks
_IN_MEMORY_LOCK = threading.Lock()
_IN_MEMORY_SEQUENCES: dict[str, int] = {}
_IN_MEMORY_RECORDS: dict[str, dict[str, Any]] = {}
_IN_MEMORY_BATCHES: dict[str, dict[str, Any]] = {}


class ReplayAttackError(RuntimeError):
    """Raised when an operation attempts to persist a duplicate sequence number for a model."""


def get_db_connection():
    """Establish and return a direct connection to PostgreSQL."""
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "cvguard")
    user = os.getenv("POSTGRES_USER", "cvguard_admin")
    password = os.getenv("POSTGRES_PASSWORD", "cvguard_dev_secret_change_me_in_prod")

    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        row_factory=dict_row,
    )


def init_db() -> bool:
    """Verify and initialize inference plane database schema. Returns True if live Postgres ready."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inference_sequences (
                        model_id TEXT PRIMARY KEY,
                        current_sequence BIGINT NOT NULL DEFAULT 0,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
                    );

                    CREATE TABLE IF NOT EXISTS inference_records (
                        record_id TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        monotonic_sequence_no BIGINT NOT NULL,
                        sequence_number BIGINT NOT NULL,
                        input_hash TEXT NOT NULL,
                        config_hash TEXT NOT NULL,
                        output_hash TEXT NOT NULL,
                        record_hash TEXT NOT NULL,
                        nonce TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        config_json JSONB NOT NULL,
                        output_json JSONB NOT NULL,
                        batch_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                        CONSTRAINT uq_model_sequence UNIQUE (model_id, monotonic_sequence_no)
                    );

                    CREATE INDEX IF NOT EXISTS idx_records_model_seq ON inference_records (model_id, monotonic_sequence_no);
                    CREATE INDEX IF NOT EXISTS idx_records_batch_id ON inference_records (batch_id);
                    CREATE INDEX IF NOT EXISTS idx_records_record_hash ON inference_records (record_hash);

                    CREATE TABLE IF NOT EXISTS inference_batches (
                        batch_id TEXT PRIMARY KEY,
                        merkle_root TEXT NOT NULL,
                        size INT NOT NULL,
                        first_sequence BIGINT NOT NULL,
                        last_sequence BIGINT NOT NULL,
                        model_id TEXT,
                        finding_id TEXT NOT NULL,
                        ledger_id BIGINT,
                        signed_finding JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
                    );

                    CREATE INDEX IF NOT EXISTS idx_batches_merkle_root ON inference_batches (merkle_root);
                    """
                )
            conn.commit()
        logger.info("Successfully initialized PostgreSQL tables for Inference Plane.")
        return True
    except Exception as exc:
        logger.warning(
            "PostgreSQL unreachable at %s:%s (%s). Operating with in-memory persistence fallback.",
            os.getenv("POSTGRES_HOST", "postgres"),
            os.getenv("POSTGRES_PORT", "5432"),
            exc,
        )
        return False


def get_next_sequence_number(model_id: str) -> int:
    """Atomically increment and return the next monotonically increasing sequence number for a model.

    In PostgreSQL, uses INSERT ... ON CONFLICT DO UPDATE ... RETURNING current_sequence.
    This guarantees atomic row-level increment with zero race conditions across concurrent callers.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inference_sequences (model_id, current_sequence, updated_at)
                    VALUES (%s, 1, clock_timestamp())
                    ON CONFLICT (model_id)
                    DO UPDATE SET
                        current_sequence = inference_sequences.current_sequence + 1,
                        updated_at = clock_timestamp()
                    RETURNING current_sequence;
                    """,
                    (model_id,),
                )
                row = cur.fetchone()
                conn.commit()
                if row:
                    return int(row["current_sequence"])
    except Exception as exc:
        logger.debug("Database error during get_next_sequence_number (%s); using in-memory fallback.", exc)

    with _IN_MEMORY_LOCK:
        current = _IN_MEMORY_SEQUENCES.get(model_id, 0) + 1
        _IN_MEMORY_SEQUENCES[model_id] = current
        return current


def save_inference_record(
    record_id: str,
    model_id: str,
    monotonic_sequence_no: int,
    input_hash: str,
    config_hash: str,
    output_hash: str,
    record_hash: str,
    nonce: str,
    timestamp: str,
    config_json: dict[str, Any] | str,
    output_json: dict[str, Any],
    batch_id: str | None = None,
) -> None:
    """Persist an immutable inference record bound to its atomic sequence number.

    Rejects insertion at write time if (model_id, monotonic_sequence_no) violates the
    uniqueness constraint (the core database replay defense).
    """
    config_str = json.dumps(config_json) if isinstance(config_json, dict) else str(config_json)
    output_str = json.dumps(output_json)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO inference_records (
                            record_id, model_id, monotonic_sequence_no, sequence_number,
                            input_hash, config_hash, output_hash, record_hash,
                            nonce, timestamp, config_json, output_json, batch_id
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s);
                        """,
                        (
                            record_id,
                            model_id,
                            monotonic_sequence_no,
                            monotonic_sequence_no,
                            input_hash,
                            config_hash,
                            output_hash,
                            record_hash,
                            nonce,
                            str(timestamp),
                            config_str,
                            output_str,
                            batch_id,
                        ),
                    )
                    conn.commit()
                    return
                except psycopg.errors.UniqueViolation as u_exc:
                    conn.rollback()
                    raise ReplayAttackError(
                        f"REPLAY ATTACK REJECTED: Sequence number {monotonic_sequence_no} for model {model_id} "
                        f"already exists in database."
                    ) from u_exc
    except ReplayAttackError:
        raise
    except Exception as exc:
        logger.debug("Database error during save_inference_record (%s); falling back to in-memory.", exc)

    with _IN_MEMORY_LOCK:
        # Check uniqueness constraint in in-memory store
        for r in _IN_MEMORY_RECORDS.values():
            if r["model_id"] == model_id and r["monotonic_sequence_no"] == monotonic_sequence_no:
                raise ReplayAttackError(
                    f"REPLAY ATTACK REJECTED: Sequence number {monotonic_sequence_no} for model {model_id} "
                    f"has already been persisted in memory."
                )

        _IN_MEMORY_RECORDS[record_id] = {
            "record_id": record_id,
            "model_id": model_id,
            "monotonic_sequence_no": monotonic_sequence_no,
            "sequence_number": monotonic_sequence_no,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "output_hash": output_hash,
            "record_hash": record_hash,
            "nonce": nonce,
            "timestamp": str(timestamp),
            "config_json": config_json,
            "output_json": output_json,
            "batch_id": batch_id,
        }


def tamper_record_output(record_id: str, tampered_output: dict[str, Any]) -> None:
    """Direct database update modifying record output post-hoc, bypassing API controls.

    Used strictly for security testing to prove that post-hoc tampering of the output field
    invalidates Merkle verification and cryptographic binding.
    """
    output_str = json.dumps(tampered_output)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE inference_records
                    SET output_json = %s::jsonb
                    WHERE record_id = %s;
                    """,
                    (output_str, record_id),
                )
                conn.commit()
                return
    except Exception as exc:
        logger.debug("Database error during tamper_record_output (%s); tampering in-memory.", exc)

    with _IN_MEMORY_LOCK:
        if record_id in _IN_MEMORY_RECORDS:
            _IN_MEMORY_RECORDS[record_id]["output_json"] = tampered_output


def get_inference_record(record_id: str) -> dict[str, Any] | None:
    """Retrieve an inference record by record_id."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT record_id, model_id, monotonic_sequence_no, sequence_number,
                           input_hash, config_hash, output_hash, record_hash,
                           nonce, timestamp, config_json, output_json, batch_id, created_at
                    FROM inference_records
                    WHERE record_id = %s;
                    """,
                    (record_id,),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
    except Exception as exc:
        logger.debug("Database error during get_inference_record (%s); using in-memory store.", exc)

    with _IN_MEMORY_LOCK:
        return _IN_MEMORY_RECORDS.get(record_id)


def update_records_batch_id(record_ids: list[str], batch_id: str) -> None:
    """Associate records with their containing batch ID once sealed."""
    if not record_ids:
        return

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE inference_records
                    SET batch_id = %s
                    WHERE record_id = ANY(%s);
                    """,
                    (batch_id, record_ids),
                )
                conn.commit()
                return
    except Exception as exc:
        logger.debug("Database error during update_records_batch_id (%s); using in-memory store.", exc)

    with _IN_MEMORY_LOCK:
        for rid in record_ids:
            if rid in _IN_MEMORY_RECORDS:
                _IN_MEMORY_RECORDS[rid]["batch_id"] = batch_id


def save_inference_batch(
    batch_id: str,
    merkle_root: str,
    size: int,
    first_sequence: int,
    last_sequence: int,
    model_id: str | None,
    finding_id: str,
    ledger_id: int | None,
    signed_finding: dict[str, Any],
) -> None:
    """Store a sealed Merkle batch and its signed finding envelope."""
    signed_finding_str = json.dumps(signed_finding)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inference_batches (
                        batch_id, merkle_root, size, first_sequence,
                        last_sequence, model_id, finding_id, ledger_id, signed_finding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (batch_id) DO NOTHING;
                    """,
                    (
                        batch_id,
                        merkle_root,
                        size,
                        first_sequence,
                        last_sequence,
                        model_id,
                        finding_id,
                        ledger_id,
                        signed_finding_str,
                    ),
                )
                conn.commit()
                return
    except Exception as exc:
        logger.debug("Database error during save_inference_batch (%s); using in-memory store.", exc)

    with _IN_MEMORY_LOCK:
        _IN_MEMORY_BATCHES[batch_id] = {
            "batch_id": batch_id,
            "merkle_root": merkle_root,
            "size": size,
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "model_id": model_id,
            "finding_id": finding_id,
            "ledger_id": ledger_id,
            "signed_finding": signed_finding,
        }


def get_inference_batch(batch_id: str) -> dict[str, Any] | None:
    """Retrieve an inference batch by batch_id."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT batch_id, merkle_root, size, first_sequence,
                           last_sequence, model_id, finding_id, ledger_id, signed_finding, created_at
                    FROM inference_batches
                    WHERE batch_id = %s;
                    """,
                    (batch_id,),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
    except Exception as exc:
        logger.debug("Database error during get_inference_batch (%s); using in-memory store.", exc)

    with _IN_MEMORY_LOCK:
        return _IN_MEMORY_BATCHES.get(batch_id)


def reset_in_memory_state() -> None:
    """Helper for unit tests to wipe in-memory fallback structures."""
    with _IN_MEMORY_LOCK:
        _IN_MEMORY_SEQUENCES.clear()
        _IN_MEMORY_RECORDS.clear()
        _IN_MEMORY_BATCHES.clear()
