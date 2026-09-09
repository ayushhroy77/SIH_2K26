"""PostgreSQL metadata, reference distributions, and detection persistence layer for CVGuard Data Plane."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger("cvguard.dataplane.db")

# In-memory fallback caches for isolated unit tests / environments without live Postgres
_IN_MEMORY_DISTRIBUTIONS: dict[tuple[str, str], dict[str, Any]] = {}
_IN_MEMORY_IMAGES: list[dict[str, Any]] = []
_IN_MEMORY_DETECTIONS: list[dict[str, Any]] = []


def get_db_connection():
    """Establish and return a direct connection to PostgreSQL."""
    host = os.getenv("POSTGRES_HOST", "localhost")
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


def init_db() -> None:
    """Ensure data plane tables and indexes exist on service initialization."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS images (
                        id BIGSERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        minio_key TEXT NOT NULL,
                        uploaded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                        contributor_id TEXT NOT NULL,
                        phash TEXT NOT NULL,
                        label TEXT,
                        dataset_id TEXT DEFAULT 'default'
                    );
                    CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images (sha256);
                    CREATE INDEX IF NOT EXISTS idx_images_contributor ON images (contributor_id);
                    CREATE INDEX IF NOT EXISTS idx_images_label ON images (label);

                    CREATE TABLE IF NOT EXISTS reference_distributions (
                        id BIGSERIAL PRIMARY KEY,
                        dataset_id TEXT NOT NULL,
                        class_name TEXT NOT NULL,
                        centroid JSONB NOT NULL,
                        covariance_inv JSONB NOT NULL,
                        num_samples INT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                        CONSTRAINT uq_dataset_class UNIQUE (dataset_id, class_name)
                    );
                    CREATE INDEX IF NOT EXISTS idx_ref_dist_dataset_class ON reference_distributions (dataset_id, class_name);

                    CREATE TABLE IF NOT EXISTS detections (
                        id BIGSERIAL PRIMARY KEY,
                        finding_id TEXT NOT NULL,
                        ledger_id BIGINT,
                        asset_type TEXT NOT NULL,
                        contributor_id TEXT,
                        evidence JSONB NOT NULL,
                        detected_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
                    );
                    CREATE INDEX IF NOT EXISTS idx_detections_finding ON detections (finding_id);
                    CREATE INDEX IF NOT EXISTS idx_detections_ledger ON detections (ledger_id);
                    """
                )
            conn.commit()
            logger.info("Data plane database tables initialized successfully.")
    except Exception as exc:
        logger.warning("Database auto-initialization could not connect to Postgres: %s", exc)


def save_image_metadata(
    filename: str,
    sha256_hash: str,
    minio_key: str,
    contributor_id: str,
    phash: str,
    label: str | None = None,
    dataset_id: str = "default",
) -> int:
    """Persist ingested image metadata and return generated image ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO images (filename, sha256, minio_key, contributor_id, phash, label, dataset_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (filename, sha256_hash, minio_key, contributor_id, phash, label, dataset_id),
                )
                result = cur.fetchone()
                conn.commit()
                return int(result["id"]) if result else 0
    except Exception as exc:
        logger.debug("Postgres unavailable, writing image metadata to in-memory store: %s", exc)
        next_id = len(_IN_MEMORY_IMAGES) + 1
        _IN_MEMORY_IMAGES.append(
            {
                "id": next_id,
                "filename": filename,
                "sha256": sha256_hash,
                "minio_key": minio_key,
                "contributor_id": contributor_id,
                "phash": phash,
                "label": label,
                "dataset_id": dataset_id,
            }
        )
        return next_id


def save_reference_distribution(
    dataset_id: str,
    class_name: str,
    centroid: list[float],
    covariance_inv: list[list[float]],
    num_samples: int,
) -> None:
    """Upsert reference distribution centroid and regularized precision matrix (inverse covariance)."""
    # Always keep in-memory cache updated
    _IN_MEMORY_DISTRIBUTIONS[(dataset_id, class_name)] = {
        "dataset_id": dataset_id,
        "class_name": class_name,
        "centroid": centroid,
        "covariance_inv": covariance_inv,
        "num_samples": num_samples,
    }

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reference_distributions (dataset_id, class_name, centroid, covariance_inv, num_samples)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_id, class_name)
                    DO UPDATE SET
                        centroid = EXCLUDED.centroid,
                        covariance_inv = EXCLUDED.covariance_inv,
                        num_samples = EXCLUDED.num_samples,
                        created_at = clock_timestamp();
                    """,
                    (dataset_id, class_name, json.dumps(centroid), json.dumps(covariance_inv), num_samples),
                )
            conn.commit()
            logger.info("Saved reference distribution for dataset='%s' class='%s'", dataset_id, class_name)
    except Exception as exc:
        logger.debug("Postgres unavailable, stored reference distribution in-memory: %s", exc)


def get_reference_distribution(dataset_id: str, class_name: str) -> dict[str, Any] | None:
    """Retrieve reference distribution by dataset and class name."""
    # Check in-memory cache first
    key = (dataset_id, class_name)
    if key in _IN_MEMORY_DISTRIBUTIONS:
        return _IN_MEMORY_DISTRIBUTIONS[key]

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dataset_id, class_name, centroid, covariance_inv, num_samples
                    FROM reference_distributions
                    WHERE dataset_id = %s AND class_name = %s
                    LIMIT 1;
                    """,
                    (dataset_id, class_name),
                )
                row = cur.fetchone()
                if row:
                    dist = {
                        "dataset_id": row["dataset_id"],
                        "class_name": row["class_name"],
                        "centroid": row["centroid"] if isinstance(row["centroid"], list) else json.loads(row["centroid"]),
                        "covariance_inv": row["covariance_inv"] if isinstance(row["covariance_inv"], list) else json.loads(row["covariance_inv"]),
                        "num_samples": row["num_samples"],
                    }
                    _IN_MEMORY_DISTRIBUTIONS[key] = dist
                    return dist
                return None
    except Exception as exc:
        logger.debug("Postgres unavailable during distribution lookup: %s", exc)
        return _IN_MEMORY_DISTRIBUTIONS.get(key)


def list_reference_distributions(dataset_id: str | None = None) -> list[dict[str, Any]]:
    """List all registered reference distributions."""
    results: list[dict[str, Any]] = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if dataset_id:
                    cur.execute(
                        "SELECT dataset_id, class_name, num_samples, created_at FROM reference_distributions WHERE dataset_id = %s",
                        (dataset_id,),
                    )
                else:
                    cur.execute("SELECT dataset_id, class_name, num_samples, created_at FROM reference_distributions")
                for r in cur.fetchall():
                    results.append(dict(r))
        return results
    except Exception:
        for (ds, cls), val in _IN_MEMORY_DISTRIBUTIONS.items():
            if dataset_id is None or ds == dataset_id:
                results.append({"dataset_id": ds, "class_name": cls, "num_samples": val["num_samples"]})
        return results


def save_detection(
    finding_id: str,
    ledger_id: int | None,
    asset_type: str,
    contributor_id: str | None,
    evidence: list[str],
) -> int:
    """Persist local detection record linking finding to Governance Spine ledger ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO detections (finding_id, ledger_id, asset_type, contributor_id, evidence)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (finding_id, ledger_id, asset_type, contributor_id, json.dumps(evidence)),
                )
                result = cur.fetchone()
                conn.commit()
                return int(result["id"]) if result else 0
    except Exception as exc:
        logger.debug("Postgres unavailable, stored detection in-memory: %s", exc)
        next_id = len(_IN_MEMORY_DETECTIONS) + 1
        _IN_MEMORY_DETECTIONS.append(
            {
                "id": next_id,
                "finding_id": finding_id,
                "ledger_id": ledger_id,
                "asset_type": asset_type,
                "contributor_id": contributor_id,
                "evidence": evidence,
            }
        )
        return next_id


def get_image_records(limit: int = 100) -> list[dict[str, Any]]:
    """Retrieve recent image records from the catalog."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, sha256, minio_key, uploaded_at, contributor_id, phash, label, dataset_id
                    FROM images
                    ORDER BY id DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                return cur.fetchall()
    except Exception:
        return _IN_MEMORY_IMAGES[-limit:]
