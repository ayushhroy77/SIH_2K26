"""CVGuard Data Plane Service.

Phase 3: Multi-detector vision integrity assurance:
- 64-bit DCT Perceptual Hashing (Near-duplicate detector)
- Dense feature embeddings via ONNX Runtime CPU vision backbone
- Out-of-Distribution (OOD) detection via Mahalanobis distance against reference centroids & covariances
- Label-flip / Systematic mislabelling detection via k-Nearest Neighbors
- Frequency-domain 2D FFT spectral anomaly & trigger/backdoor pattern detection with spatial localization
- Unified Source-Level Aggregator with multi-detector weighted z-scoring
- Tamper-evident Governance Spine sealing for all findings
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cvguard_schemas
from cvguard_schemas import Finding, Role, SignedFinding, verify_bearer_token
from db import (
    get_image_records,
    get_reference_distribution,
    init_db,
    list_reference_distributions,
    save_detection,
    save_image_metadata,
    save_reference_distribution,
)
from detector import IngestedImage, NearDuplicateDetector, compute_image_phash
from embeddings import get_embedding_extractor
from governance_client import GovernanceDispatchError, dispatch_finding_to_governance
from labelflip_detector import LabeledSample, LabelFlipDetector
from ood_detector import OODDetector, fit_reference_distribution
from source_aggregator import AnomalySignal, SourceAggregator
from storage import MinIOStorage
from trigger_detector import TriggerDetector

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("cvguard.dataplane")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize database schema, MinIO storage, and feature models."""
    logger.info("Initializing CVGuard Data Plane storage and database schema...")
    init_db()
    try:
        storage = MinIOStorage()
        storage.ensure_bucket("images")
    except Exception as exc:
        logger.warning("MinIO bucket initialization deferred: %s", exc)

    # Warm up feature extractor
    try:
        get_embedding_extractor()
    except Exception as exc:
        logger.warning("Feature extractor warm-up: %s", exc)

    yield


app = FastAPI(
    title="CVGuard Data Plane Service",
    description="Vision integrity assurance: pHash, OOD Mahalanobis, kNN label-flip, and FFT trigger detectors.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = MinIOStorage()


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = Field(default="ok")
    service: str = Field(default="data-plane")
    version: str = Field(default="0.3.0")
    schemas_version: str = Field(default=cvguard_schemas.__version__)


class IngestResponse(BaseModel):
    """Payload returned upon completed multi-image ingestion and detector run."""

    status: str = "ok"
    ingested_count: int
    findings_count: int
    images: list[dict[str, Any]]
    findings: list[SignedFinding]


class ReferenceDistributionRequest(BaseModel):
    """Request payload to register a reference class distribution."""

    dataset_id: str = Field(default="default", description="Identifier of dataset or domain.")
    class_name: str = Field(..., description="Target class or category name.")
    centroid: list[float] | None = Field(default=None, description="Precomputed centroid vector.")
    covariance_inv: list[list[float]] | None = Field(
        default=None,
        description="Precomputed regularized inverse covariance (precision) matrix.",
    )
    num_samples: int | None = Field(default=None, description="Number of reference samples used.")
    reference_embeddings: list[list[float]] | None = Field(
        default=None,
        description="Optional list of raw embedding vectors to automatically fit centroid and precision matrix.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness/liveness health check probe endpoint."""
    return HealthResponse(
        status="ok",
        service="data-plane",
        version="0.3.0",
        schemas_version=cvguard_schemas.__version__,
    )


def require_admin_role(request: Request) -> None:
    """Ensure that only admin users can register reference distributions."""
    user_roles = request.headers.get("X-User-Roles")
    auth_header = request.headers.get("authorization")

    if user_roles:
        roles = [r.strip() for r in user_roles.split(",") if r.strip()]
        if Role.ADMIN.value not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Insufficient permissions",
            )
        return

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        try:
            user = verify_bearer_token(token)
            if not user.has_role(Role.ADMIN):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient permissions",
                )
            return
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid authentication token",
            )

    mtls_enabled = os.getenv("CVGUARD_MTLS_ENABLED", "false").lower() in ("true", "1", "yes")
    if mtls_enabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing or invalid authentication token",
        )


@app.post("/reference-distributions", status_code=status.HTTP_201_CREATED)
async def register_reference_distribution(
    payload: ReferenceDistributionRequest,
    _admin: None = Depends(require_admin_role),
):
    """Register or update an empirical reference distribution for OOD Mahalanobis detection."""
    if payload.reference_embeddings is not None and len(payload.reference_embeddings) >= 2:
        try:
            centroid, cov_inv = fit_reference_distribution(payload.reference_embeddings)
            num_samples = len(payload.reference_embeddings)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fit reference distribution: {exc}",
            )
    elif payload.centroid is not None and payload.covariance_inv is not None:
        centroid = payload.centroid
        cov_inv = payload.covariance_inv
        num_samples = payload.num_samples or 100
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either reference_embeddings (>=2) OR both centroid and covariance_inv must be supplied.",
        )

    save_reference_distribution(
        dataset_id=payload.dataset_id,
        class_name=payload.class_name,
        centroid=centroid,
        covariance_inv=cov_inv,
        num_samples=num_samples,
    )

    return {
        "status": "ok",
        "dataset_id": payload.dataset_id,
        "class_name": payload.class_name,
        "num_samples": num_samples,
        "centroid_dim": len(centroid),
    }


@app.get("/reference-distributions")
async def get_all_reference_distributions(dataset_id: str | None = None):
    """List registered class reference distributions."""
    distributions = list_reference_distributions(dataset_id=dataset_id)
    return {"reference_distributions": distributions}


@app.post("/ingest/images", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_images(
    files: list[UploadFile] = File(..., description="Multipart list of image files to ingest"),
    contributor_id: str = Form(default="default", description="Default contributor ID for this batch."),
    contributor_ids: str | None = Form(
        default=None,
        description="Optional comma-separated list of contributor IDs matching files in order.",
    ),
    labels: str | None = Form(
        default=None,
        description="Optional comma-separated list of class labels matching files in order.",
    ),
    default_label: str | None = Form(default=None, description="Default class label for batch."),
    dataset_id: str = Form(default="default", description="Dataset identifier for OOD reference lookup."),
    run_all_detectors: bool = Form(default=True, description="Whether to execute the full multi-detector suite."),
) -> IngestResponse:
    """Ingest images, extract pHash + embeddings, run multi-detector suite, aggregate sources, and seal to Governance."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image file must be uploaded.",
        )

    # Parse contributor IDs
    c_list: list[str] = []
    if contributor_ids:
        c_list = [c.strip() for c in contributor_ids.split(",") if c.strip()]

    # Parse labels
    l_list: list[str] = []
    if labels:
        l_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]

    extractor = get_embedding_extractor()
    ingested_records: list[IngestedImage] = []
    response_images: list[dict[str, Any]] = []
    image_bytes_map: dict[str, bytes] = {}
    embeddings_map: dict[str, Any] = {}
    all_contributors: set[str] = set()

    for index, file in enumerate(files):
        item_contributor = c_list[index] if index < len(c_list) else contributor_id
        item_label = l_list[index] if index < len(l_list) else default_label
        all_contributors.add(item_contributor)

        contents = await file.read()
        if not contents:
            continue

        sha256_digest = hashlib.sha256(contents).hexdigest()

        # Compute pHash
        try:
            phash_str = compute_image_phash(contents)
        except Exception:
            phash_str = "0" * 16

        safe_filename = file.filename or f"img_{index}.jpg"
        minio_key = f"{sha256_digest[:16]}_{safe_filename}"
        content_type = file.content_type or "image/jpeg"

        # Cache raw bytes for detectors
        image_bytes_map[minio_key] = contents

        # Store in MinIO
        try:
            storage.put_image(
                object_name=minio_key,
                data=contents,
                content_type=content_type,
                bucket_name="images",
            )
        except Exception as exc:
            logger.error("Failed to store image '%s' in MinIO: %s", minio_key, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"MinIO storage error for {safe_filename}: {exc}",
            )

        # Save metadata to DB
        try:
            image_id = save_image_metadata(
                filename=safe_filename,
                sha256_hash=sha256_digest,
                minio_key=minio_key,
                contributor_id=item_contributor,
                phash=phash_str,
                label=item_label,
                dataset_id=dataset_id,
            )
        except Exception as exc:
            logger.error("Database error saving image metadata: %s", exc)
            image_id = index + 1

        record = IngestedImage(
            id=image_id,
            filename=safe_filename,
            sha256=sha256_digest,
            minio_key=minio_key,
            contributor_id=item_contributor,
            phash=phash_str,
            label=item_label,
            dataset_id=dataset_id,
        )
        ingested_records.append(record)
        response_images.append(
            {
                "id": image_id,
                "filename": safe_filename,
                "sha256": sha256_digest,
                "minio_key": minio_key,
                "contributor_id": item_contributor,
                "phash": phash_str,
                "label": item_label,
                "dataset_id": dataset_id,
            }
        )

        # Extract visual feature embedding
        try:
            embedding = extractor.extract(contents)
            embeddings_map[minio_key] = embedding
        except Exception as exc:
            logger.warning("Failed to extract embedding for %s: %s", safe_filename, exc)

    all_findings: list[Finding] = []
    all_signals: list[AnomalySignal] = []

    # =========================================================================
    # 1. Detector: Near-Duplicate (Perceptual Hashing)
    # =========================================================================
    dup_detector = NearDuplicateDetector()
    dup_findings, dup_signals = dup_detector.evaluate_with_signals(ingested_records)
    all_findings.extend(dup_findings)
    all_signals.extend(dup_signals)

    if run_all_detectors:
        # =====================================================================
        # 2. Detector: Out-of-Distribution (OOD Mahalanobis)
        # =====================================================================
        ood_detector = OODDetector()
        for record in ingested_records:
            if record.label and record.minio_key in embeddings_map:
                ref_dist = get_reference_distribution(dataset_id=record.dataset_id, class_name=record.label)
                if ref_dist:
                    eval_res = ood_detector.evaluate_sample(
                        image_id=record.id,
                        filename=record.filename,
                        contributor_id=record.contributor_id,
                        minio_key=record.minio_key,
                        embedding=embeddings_map[record.minio_key],
                        class_name=record.label,
                        ref_distribution=ref_dist,
                    )
                    if eval_res.is_ood and eval_res.finding:
                        all_findings.append(eval_res.finding)
                        all_signals.append(
                            AnomalySignal(
                                image_id=record.id,
                                contributor_id=record.contributor_id,
                                minio_key=record.minio_key,
                                detector_name="ood_mahalanobis",
                                score=eval_res.confidence,
                                is_anomaly=True,
                                weight=1.0,
                            )
                        )

        # =====================================================================
        # 3. Detector: Label-Flip / Systematic Mislabelling (kNN Consensus)
        # =====================================================================
        labeled_samples: list[LabeledSample] = []
        for record in ingested_records:
            if record.label and record.minio_key in embeddings_map:
                labeled_samples.append(
                    LabeledSample(
                        image_id=record.id,
                        filename=record.filename,
                        contributor_id=record.contributor_id,
                        minio_key=record.minio_key,
                        embedding=embeddings_map[record.minio_key],
                        label=record.label,
                    )
                )

        if len(labeled_samples) >= 2:
            labelflip_detector = LabelFlipDetector()
            lf_evals = labelflip_detector.evaluate_batch(labeled_samples)
            for eval_res in lf_evals:
                if eval_res.is_mislabeled and eval_res.finding:
                    all_findings.append(eval_res.finding)
                    all_signals.append(
                        AnomalySignal(
                            image_id=eval_res.image_id,
                            contributor_id=eval_res.contributor_id,
                            minio_key=eval_res.minio_key,
                            detector_name="label_flip_knn",
                            score=eval_res.discordance_rate,
                            is_anomaly=True,
                            weight=1.2,
                        )
                    )

        # =====================================================================
        # 4. Detector: Frequency-Domain FFT Spectral Trigger / Backdoor
        # =====================================================================
        img_dict_list = [
            {
                "id": r.id,
                "filename": r.filename,
                "contributor_id": r.contributor_id,
                "minio_key": r.minio_key,
            }
            for r in ingested_records
        ]
        trigger_detector = TriggerDetector()
        trigger_evals = trigger_detector.evaluate_batch(img_dict_list, image_bytes_map)
        for eval_res in trigger_evals:
            if eval_res.is_trigger_anomaly and eval_res.finding:
                all_findings.append(eval_res.finding)
                all_signals.append(
                    AnomalySignal(
                        image_id=eval_res.image_id,
                        contributor_id=eval_res.contributor_id,
                        minio_key=eval_res.minio_key,
                        detector_name="trigger_fft_spectral",
                        score=eval_res.confidence,
                        is_anomaly=True,
                        weight=0.8,
                    )
                )

        # =====================================================================
        # 5. Shared Source-Level Aggregator
        # =====================================================================
        aggregator = SourceAggregator()
        source_findings = aggregator.aggregate(
            signals=all_signals,
            all_contributors=sorted(list(all_contributors)),
        )
        all_findings.extend(source_findings)

    # =========================================================================
    # 6. Dispatch All Sealed Findings to Governance Spine
    # =========================================================================
    signed_findings: list[SignedFinding] = []
    for finding in all_findings:
        try:
            signed_finding = await dispatch_finding_to_governance(finding)
            signed_findings.append(signed_finding)

            try:
                save_detection(
                    finding_id=finding.finding_id,
                    ledger_id=signed_finding.ledger_id,
                    asset_type=finding.asset_type.value,
                    contributor_id=(
                        finding.asset_ref.replace("source:", "")
                        if finding.asset_type == cvguard_schemas.AssetType.SOURCE
                        else None
                    ),
                    evidence=finding.evidence,
                )
            except Exception as db_exc:
                logger.warning("Could not persist local detection row: %s", db_exc)

        except GovernanceDispatchError as gov_exc:
            logger.error("Governance dispatch failed: %s", gov_exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(gov_exc),
            )

    logger.info(
        "Batch processing complete: %d images ingested, %d signed findings sealed into ledger.",
        len(ingested_records),
        len(signed_findings),
    )

    return IngestResponse(
        status="ok",
        ingested_count=len(ingested_records),
        findings_count=len(signed_findings),
        images=response_images,
        findings=signed_findings,
    )


@app.get("/images/{minio_key:path}")
async def get_raw_image(minio_key: str):
    """Retrieve raw image bytes from MinIO blob store for authorized evidence viewing."""
    try:
        data, content_type = storage.get_image(object_name=minio_key, bucket_name="images")
        return Response(content=data, media_type=content_type)
    except Exception as exc:
        logger.error("Failed to read image '%s' from MinIO: %s", minio_key, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{minio_key}' not found in storage.",
        )


@app.get("/images")
async def list_images(limit: int = 50):
    """Retrieve list of recently cataloged images."""
    try:
        records = get_image_records(limit=limit)
        return {"images": records}
    except Exception as exc:
        logger.warning("Could not query images from DB: %s", exc)
        return {"images": []}


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
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False, **ssl_kwargs)
