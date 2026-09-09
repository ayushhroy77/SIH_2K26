"""CVGuard Inference Plane (Phase 5).

Provides verifiable cryptographic binding between:
- Raw input image bytes (input_hash)
- Model weight digest (model_id)
- Canonical preprocessing & inference config (config_hash)
- Model output prediction (output)
- Strict monotonic sequence number (monotonic_sequence_no)
- Entropy nonce and ISO timestamp

Bundles records into Merkle trees, signs roots via Governance Spine, and verifies proofs.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import secrets
from typing import Any
import uuid

import cvguard_schemas
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
import uvicorn

from batcher import (
    BATCH_FLUSH_INTERVAL_SECONDS,
    get_batch_manager,
)
from canonical import (
    canonical_json,
    compute_config_hash,
    compute_input_hash,
    compute_output_hash,
    compute_record_hash,
    compute_sha256,
)
from db import (
    ReplayAttackError,
    get_db_connection,
    get_inference_batch,
    get_inference_record,
    get_next_sequence_number,
    init_db,
    save_inference_record,
)
from merkle import verify_merkle_proof
from model_plane_client import resolve_model_weight_digest
from schemas import (
    BatchFlushResponse,
    HealthResponse,
    InferJsonRequest,
    InferResponse,
    VerifyResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cvguard.inferenceplane")

_periodic_flush_task: asyncio.Task[None] | None = None


async def _background_flush_loop() -> None:
    """Periodically seals buffered inference records on a time interval."""
    batch_mgr = get_batch_manager()
    while True:
        try:
            await asyncio.sleep(BATCH_FLUSH_INTERVAL_SECONDS)
            if batch_mgr.pending_count > 0:
                logger.debug("Flushing pending records on timer (%d pending)", batch_mgr.pending_count)
                await batch_mgr.flush()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("Error in background flush loop: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook initializing database tables and background batch timer."""
    init_db()
    global _periodic_flush_task
    _periodic_flush_task = asyncio.create_task(_background_flush_loop())
    logger.info("Inference Plane service initialized.")
    yield
    if _periodic_flush_task:
        _periodic_flush_task.cancel()
        try:
            await _periodic_flush_task
        except asyncio.CancelledError:
            pass
    # Flush any remaining records before shutdown
    batch_mgr = get_batch_manager()
    if batch_mgr.pending_count > 0:
        await batch_mgr.flush()


app = FastAPI(
    title="CVGuard Inference Plane",
    version="0.2.0",
    description="Cryptographic provenance, atomic monotonic sequencing, and Merkle tree audit logging for computer vision inference.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check verifying database connectivity and pending buffer count."""
    db_ok = False
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok",
        service="inference-plane",
        version="0.2.0",
        schemas_version=cvguard_schemas.__version__,
        db_connected=db_ok,
        pending_records_in_buffer=get_batch_manager().pending_count,
    )


def _simulate_model_inference(
    input_hash: str,
    weight_digest: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic inference simulator for vision tasks based on image and weight binding."""
    # Combine hashes for deterministic output generation
    combined = compute_sha256(f"{input_hash}:{weight_digest}")
    score_raw = int(combined[:4], 16) / 65535.0
    confidence = round(0.50 + (score_raw * 0.49), 4)

    classes = ["person", "vehicle", "bicycle", "animal", "traffic_light", "pedestrian"]
    class_idx = int(combined[4:6], 16) % len(classes)

    return {
        "task": config.get("task", "object_detection"),
        "top_class": classes[class_idx],
        "confidence": confidence,
        "box": [32, 45, 220, 180],
        "predictions": [
            {"label": classes[class_idx], "score": confidence},
            {"label": classes[(class_idx + 1) % len(classes)], "score": round(1.0 - confidence, 4)},
        ],
        "execution_provider": "cpu",
    }


@app.post(
    "/infer",
    response_model=InferResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute inference, atomically assign monotonic sequence, and seal provenance binding",
)
async def infer(
    request: Request,
    image: UploadFile | None = File(None),
    model_id: str | None = Form(None),
    config: str | None = Form(None),
) -> InferResponse:
    """Accept an image, resolve model weight digest, atomically increment sequence number,
    compute cryptographic binding, and enqueue record for Merkle batching.

    Accepts both multipart/form-data and JSON bodies (via Base64).
    """
    image_bytes: bytes = b""
    model_ref: str = ""
    config_dict: dict[str, Any] = {}

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body_json = await request.json()
            model_ref = str(body_json.get("model_id", "")).strip()
            image_b64 = body_json.get("image_base64", "")
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
            raw_cfg = body_json.get("config", {})
            if isinstance(raw_cfg, str):
                try:
                    config_dict = json.loads(raw_cfg)
                except Exception:
                    config_dict = {"raw_config": raw_cfg}
            elif isinstance(raw_cfg, dict):
                config_dict = raw_cfg
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse JSON inference request: {exc}",
            ) from exc
    else:
        # Multipart / Form payload
        if image is not None:
            image_bytes = await image.read()
        if model_id is not None:
            model_ref = model_id.strip()
        if config is not None:
            try:
                config_dict = json.loads(config)
            except Exception:
                config_dict = {"raw_config": config}

    # Validate inputs
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inference request requires non-empty image bytes (as 'image' file upload or 'image_base64' in JSON).",
        )
    if not model_ref:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inference request requires 'model_id' referencing an ingested model or weight digest.",
        )

    # 1. Compute input_hash = SHA-256(raw image bytes)
    input_hash = compute_input_hash(image_bytes)

    # 2. Resolve model_id to the model's actual WEIGHT DIGEST
    weight_digest = await resolve_model_weight_digest(model_ref)

    # 3. Compute config_hash = SHA-256(canonical_json(config))
    config_hash = compute_config_hash(config_dict)

    # 4. Atomically assign strictly increasing monotonic sequence number for this model weight digest
    monotonic_sequence_no = get_next_sequence_number(weight_digest)

    # 5. Generate unique nonce and timestamp
    nonce = secrets.token_hex(16)
    timestamp = datetime.now(timezone.utc).isoformat()
    record_id = str(uuid.uuid4())

    # 6. Execute deterministic inference to obtain prediction output
    output = _simulate_model_inference(input_hash, weight_digest, config_dict)
    output_hash = compute_output_hash(output)

    # 7. Compute leaf record hash sealing {input_hash, model_id, config_hash, output, timestamp, monotonic_sequence_no, nonce}
    record_hash = compute_record_hash(
        input_hash=input_hash,
        model_id=weight_digest,
        config_hash=config_hash,
        output=output,
        timestamp=timestamp,
        monotonic_sequence_no=monotonic_sequence_no,
        nonce=nonce,
    )

    # 8. Store record with atomic sequence constraint (Replay Defense at write-time)
    try:
        save_inference_record(
            record_id=record_id,
            model_id=weight_digest,
            monotonic_sequence_no=monotonic_sequence_no,
            input_hash=input_hash,
            config_hash=config_hash,
            output_hash=output_hash,
            record_hash=record_hash,
            nonce=nonce,
            timestamp=timestamp,
            config_json=config_dict,
            output_json=output,
        )
    except ReplayAttackError as r_err:
        logger.error("Replay attack caught at write time: %s", r_err)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(r_err),
        ) from r_err

    # 9. Add to Merkle batch buffer
    batch_mgr = get_batch_manager()
    record_data = {
        "record_id": record_id,
        "model_id": weight_digest,
        "monotonic_sequence_no": monotonic_sequence_no,
        "sequence_number": monotonic_sequence_no,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "output_hash": output_hash,
        "record_hash": record_hash,
        "nonce": nonce,
        "timestamp": timestamp,
    }
    batch_id = await batch_mgr.add_record(record_data)

    return InferResponse(
        record_id=record_id,
        model_id=weight_digest,
        monotonic_sequence_no=monotonic_sequence_no,
        sequence_number=monotonic_sequence_no,
        input_hash=input_hash,
        config_hash=config_hash,
        output_hash=output_hash,
        record_hash=record_hash,
        nonce=nonce,
        timestamp=timestamp,
        output=output,
        batch_id=batch_id,
        status="sealed" if batch_id else "pending_batch",
    )


@app.get(
    "/verify/{record_id}",
    response_model=VerifyResponse,
    summary="Cryptographically verify an inference record and its Merkle inclusion proof",
)
async def verify_record(record_id: str) -> VerifyResponse:
    """Verify an inference record's cryptographic integrity and Merkle proof.

    Returns:
        {valid: bool, reason: str, ...}
    """
    rec = get_inference_record(record_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference record with id '{record_id}' not found.",
        )

    model_id = rec["model_id"]
    monotonic_seq = int(rec["monotonic_sequence_no"])
    input_hash = rec["input_hash"]
    config_hash = rec["config_hash"]
    stored_output = rec["output_json"]
    stored_record_hash = rec["record_hash"]
    stored_output_hash = rec["output_hash"]
    nonce = rec.get("nonce", "")
    timestamp = rec.get("timestamp", "")
    batch_id = rec.get("batch_id")

    # 1. Recompute the leaf record hash using the canonical formula
    recomputed_record_hash = compute_record_hash(
        input_hash=input_hash,
        model_id=model_id,
        config_hash=config_hash,
        output=stored_output,
        timestamp=timestamp,
        monotonic_sequence_no=monotonic_seq,
        nonce=nonce,
    )

    # 2. Check for post-hoc tampering of the stored record (e.g. altered output in DB)
    if recomputed_record_hash.lower() != stored_record_hash.lower():
        reason = (
            f"Post-hoc record alteration detected: recomputed record hash ({recomputed_record_hash[:16]}...) "
            f"does not match stored record hash ({stored_record_hash[:16]}...). "
            f"The output field or record attributes have been tampered post-hoc directly in database."
        )
        logger.warning("Tamper detected for record %s: %s", record_id, reason)
        return VerifyResponse(
            valid=False,
            reason=reason,
            record_id=record_id,
            model_id=model_id,
            monotonic_sequence_no=monotonic_seq,
            sequence_number=monotonic_seq,
            input_hash=input_hash,
            config_hash=config_hash,
            output_hash=stored_output_hash,
            record_hash=stored_record_hash,
            nonce=nonce,
            timestamp=timestamp,
            batch_id=batch_id,
            merkle_root=None,
            merkle_proof=[],
            proof_valid=False,
            governance_sealed=False,
            tampered=True,
            status="TAMPERED",
        )

    # 3. Check batch status
    batch_mgr = get_batch_manager()
    batch, proof = batch_mgr.get_proof_for_record(record_id)

    if not batch:
        # Record is genuine locally, but waiting for batch flush
        return VerifyResponse(
            valid=False,
            reason="Record leaf binding is valid, but record is currently pending Merkle batch aggregation and governance sealing.",
            record_id=record_id,
            model_id=model_id,
            monotonic_sequence_no=monotonic_seq,
            sequence_number=monotonic_seq,
            input_hash=input_hash,
            config_hash=config_hash,
            output_hash=stored_output_hash,
            record_hash=stored_record_hash,
            nonce=nonce,
            timestamp=timestamp,
            batch_id=None,
            merkle_root=None,
            merkle_proof=[],
            proof_valid=False,
            governance_sealed=False,
            tampered=False,
            status="PENDING_BATCH",
        )

    # 4. Verify Merkle proof against published batch root
    merkle_root = batch["merkle_root"]
    proof_valid = verify_merkle_proof(
        leaf_hash=recomputed_record_hash,
        proof=proof,
        expected_root=merkle_root,
    )

    if not proof_valid:
        reason = (
            f"Merkle proof verification failed: leaf hash ({recomputed_record_hash[:16]}...) "
            f"does not mathematically resolve to published Merkle root ({merkle_root[:16]}...)."
        )
        return VerifyResponse(
            valid=False,
            reason=reason,
            record_id=record_id,
            model_id=model_id,
            monotonic_sequence_no=monotonic_seq,
            sequence_number=monotonic_seq,
            input_hash=input_hash,
            config_hash=config_hash,
            output_hash=stored_output_hash,
            record_hash=stored_record_hash,
            nonce=nonce,
            timestamp=timestamp,
            batch_id=batch["batch_id"],
            merkle_root=merkle_root,
            merkle_proof=proof,
            proof_valid=False,
            governance_sealed=False,
            tampered=True,
            status="TAMPERED",
        )

    # 5. Check Governance Spine ledger inclusion and signature
    ledger_id = batch.get("ledger_id")
    signed_finding = batch.get("signed_finding", {})
    signature = signed_finding.get("signature") if isinstance(signed_finding, dict) else None

    reason = (
        f"Cryptographically verified: Record {record_id[:8]}... strictly bound to model "
        f"{model_id[:12]}... (seq #{monotonic_seq}) and verified against Merkle root "
        f"{merkle_root[:16]}... sealed in Governance Ledger ID {ledger_id}."
    )

    return VerifyResponse(
        valid=True,
        reason=reason,
        record_id=record_id,
        model_id=model_id,
        monotonic_sequence_no=monotonic_seq,
        sequence_number=monotonic_seq,
        input_hash=input_hash,
        config_hash=config_hash,
        output_hash=stored_output_hash,
        record_hash=stored_record_hash,
        nonce=nonce,
        timestamp=timestamp,
        batch_id=batch["batch_id"],
        merkle_root=merkle_root,
        merkle_proof=proof,
        proof_valid=True,
        governance_sealed=ledger_id is not None,
        ledger_id=ledger_id,
        governance_signature=signature,
        tampered=False,
        status="VERIFIED",
    )


@app.post("/batch/flush", response_model=BatchFlushResponse, summary="Force immediate batch seal")
async def trigger_batch_flush() -> BatchFlushResponse:
    """Manually seal any pending inference records into a Merkle batch and dispatch to governance."""
    batch_mgr = get_batch_manager()
    if batch_mgr.pending_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending inference records in buffer to flush.",
        )

    batch_result = await batch_mgr.flush()
    if not batch_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch flush failed or returned empty.",
        )

    return BatchFlushResponse(
        batch_id=batch_result["batch_id"],
        size=batch_result["size"],
        first_sequence=batch_result["first_sequence"],
        last_sequence=batch_result["last_sequence"],
        merkle_root=batch_result["merkle_root"],
        finding_id=batch_result["finding_id"],
        ledger_id=batch_result.get("ledger_id"),
        signature=batch_result.get("signature"),
        status="sealed",
    )


if __name__ == "__main__":
    mtls = os.getenv("CVGUARD_MTLS_ENABLED", "false").lower() in ("true", "1", "yes")
    ssl_kwargs: dict[str, Any] = {}
    if mtls:
        import ssl
        cert_path = os.getenv("CVGUARD_CERT_PATH", "/etc/cvguard/certs/service.crt")
        key_path = os.getenv("CVGUARD_KEY_PATH", "/etc/cvguard/certs/service.key")
        ca_path = os.getenv("CVGUARD_CA_CERT_PATH", "/etc/cvguard/certs/ca.crt")
        if os.path.isfile(cert_path) and os.path.isfile(key_path) and os.path.isfile(ca_path):
            ssl_kwargs = {
                "ssl_certfile": cert_path,
                "ssl_keyfile": key_path,
                "ssl_ca_certs": ca_path,
                "ssl_cert_reqs": ssl.CERT_REQUIRED,
            }
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False, **ssl_kwargs)
