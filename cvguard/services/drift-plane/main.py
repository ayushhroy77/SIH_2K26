"""CVGuard Drift Plane Service.

Phase 6: Visual domain shift detection, feature distribution drift,
calibrated risk estimation (MMD & KS-tests), and drift-vs-manipulation classification.
"""

from __future__ import annotations

import base64
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status

import cvguard_schemas
from cvguard_schemas import AssetType, Disposition, Finding, Role, Severity, verify_bearer_token

from extractor import get_embedding_extractor
from governance_client import dispatch_finding_to_governance
from profiles import ReferenceProfile, get_profile_registry
from schemas import (
    AssessBatchRequest,
    AssessBatchResponse,
    CreateProfileRequest,
    CreateProfileResponse,
    SampleInput,
)
from stats import (
    classify_drift_vs_manipulation,
    compute_calibrated_risk,
    compute_ks_tests,
    compute_mmd,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring extractor and registry are warm."""
    # Warm up shared embedding extractor
    try:
        get_embedding_extractor()
    except Exception:
        pass
    yield


app = FastAPI(
    title="CVGuard Drift Plane Service",
    description="Statistical distribution shift, Maximum Mean Discrepancy (MMD), Kolmogorov-Smirnov hypothesis testing, and operational drift vs. manipulation classification.",
    version="0.6.0",
    lifespan=lifespan,
)


def _resolve_sample_embedding(sample: SampleInput) -> list[float]:
    """Resolve embedding vector either from precomputed field or raw image bytes."""
    if sample.embedding is not None and len(sample.embedding) > 0:
        return [float(v) for v in sample.embedding]

    if sample.image_base64:
        try:
            image_bytes = base64.b64decode(sample.image_base64)
            extractor = get_embedding_extractor()
            emb = extractor.extract(image_bytes)
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return list(emb)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process sample image_base64: {exc}",
            )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Each sample must provide either 'image_base64' or 'embedding'.",
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Readiness/liveness health check probe endpoint."""
    registry = get_profile_registry()
    return {
        "status": "ok",
        "service": "drift-plane",
        "version": "0.6.0",
        "schemas_version": cvguard_schemas.__version__,
        "registered_profiles_count": len(registry.list_profiles()),
    }


def require_admin_role(request: Request) -> None:
    """Ensure that only admin users can create reference profiles."""
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


@app.post(
    "/reference-profile",
    response_model=CreateProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_reference_profile(
    payload: CreateProfileRequest,
    _admin: None = Depends(require_admin_role),
) -> CreateProfileResponse:
    """Ingest reference batch representing declared normal operational state and persist versioned profile."""
    registry = get_profile_registry()

    profile_id = payload.profile_id or f"prof_{uuid.uuid4().hex[:12]}"

    # Determine version: auto-increment if not explicitly specified
    latest_ver = registry.get_latest_version(profile_id)
    if payload.version is not None:
        version = payload.version
    else:
        version = (latest_ver + 1) if latest_ver is not None else 1

    # Extract embeddings and collect metadata
    embeddings: list[list[float]] = []
    metadata_list: list[dict[str, Any]] = []

    for sample in payload.samples:
        emb = _resolve_sample_embedding(sample)
        embeddings.append(emb)
        metadata_list.append(sample.metadata or {})

    # Compute centroid, covariance, and metadata histograms
    profile = ReferenceProfile.compute_from_samples(
        profile_id=profile_id,
        version=version,
        embeddings=embeddings,
        metadata_list=metadata_list,
    )

    registry.save_profile(profile)

    return CreateProfileResponse(
        profile_id=profile.profile_id,
        version=profile.version,
        num_samples=profile.num_samples,
        embedding_dim=profile.dim,
        centroid_summary=profile.centroid[:8],
        metadata_histograms=profile.metadata_histograms,
        created_at=profile.created_at,
    )


@app.post("/assess-batch", response_model=AssessBatchResponse)
async def assess_batch(payload: AssessBatchRequest) -> AssessBatchResponse:
    """Assess a new batch against declared reference distribution and classify drift vs manipulation."""
    registry = get_profile_registry()

    # Retrieve reference profile
    ref_profile = registry.get_profile(payload.profile_id, payload.version)
    if ref_profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference profile '{payload.profile_id}' (version={payload.version}) not found.",
        )

    batch_id = payload.batch_id or f"batch_{uuid.uuid4().hex[:12]}"

    # Extract query embeddings and metadata
    batch_embeddings: list[list[float]] = []
    batch_metadata: list[dict[str, Any]] = []

    for sample in payload.samples:
        emb = _resolve_sample_embedding(sample)
        batch_embeddings.append(emb)
        batch_metadata.append(sample.metadata or {})

    # 1. Compute Maximum Mean Discrepancy (MMD)
    mmd_stat, mmd_se = compute_mmd(ref_profile.embeddings, batch_embeddings)

    # 2. Compute per-dimension Kolmogorov-Smirnov (KS) tests
    ks_results = compute_ks_tests(ref_profile.embeddings, batch_embeddings)

    # 3. Compute calibrated risk score with 95% confidence interval and standard error
    risk_score, (ci_lower, ci_upper), std_error = compute_calibrated_risk(
        mmd_stat=mmd_stat,
        mmd_se=mmd_se,
        ks_shifted_ratio=ks_results["shifted_dimension_ratio"],
        mean_ks_stat=ks_results["mean_ks_statistic"],
    )

    # 4. Execute Drift-vs-Manipulation Classification (with explicit indeterminate branch)
    classification, reason_text, class_evidence = classify_drift_vs_manipulation(
        ref_centroid=ref_profile.centroid,
        batch_embeddings=batch_embeddings,
        batch_metadata=batch_metadata,
        risk_score=risk_score,
        ref_metadata_histograms=ref_profile.metadata_histograms,
    )

    # Compile structured evidence list
    evidence_strings = [
        f"mmd_statistic:{mmd_stat:.5f}",
        f"mmd_standard_error:{mmd_se:.5f}",
        f"ks_shifted_dimension_ratio:{ks_results['shifted_dimension_ratio']:.4f}",
        f"ks_mean_statistic:{ks_results['mean_ks_statistic']:.4f}",
        f"risk_score:{risk_score:.4f}",
        f"confidence_interval:[{ci_lower:.4f},{ci_upper:.4f}]",
        f"classification:{classification}",
    ]

    correlated_field = class_evidence.get("correlated_field")
    if correlated_field:
        evidence_strings.append(f"correlated_metadata_field:{correlated_field}")
        evidence_strings.append(f"correlated_metadata_value:{class_evidence.get('correlated_value')}")
        evidence_strings.append(f"metadata_correlation_score:{class_evidence.get('correlation_score')}")
    else:
        evidence_strings.append("correlated_metadata_field:none")

    for wd in ks_results.get("worst_dimensions", [])[:3]:
        evidence_strings.append(
            f"shifted_dim_{wd['dimension']}:ks={wd['statistic']:.4f},p={wd['p_value']:.4f}"
        )

    # Explicit limitations statement per task requirements
    limitations = [
        "Classification quality depends strictly on metadata completeness and accuracy.",
        "Does not (yet) account for adversarially-crafted metadata disguised as operational parameters.",
        "Assumes IID distribution of reference profile baseline samples.",
    ]

    # Map classification to Severity & Disposition
    if classification == "suspicious_manipulation":
        severity = Severity.CRITICAL if risk_score > 0.60 else Severity.HIGH
        disposition = Disposition.QUARANTINE
    elif classification == "probable_operational_drift":
        severity = Severity.MEDIUM
        disposition = Disposition.REVIEW
    elif classification == "indeterminate":
        severity = Severity.LOW if risk_score < 0.50 else Severity.MEDIUM
        disposition = Disposition.REVIEW
    else:
        severity = Severity.INFO
        disposition = Disposition.ACCEPT

    # Confidence derived from statistical confidence interval width
    ci_width = ci_upper - ci_lower
    confidence = max(0.50, min(0.99, round(1.0 - (ci_width / 2.0), 3)))

    # Emit Finding via Governance Spine POST /findings
    finding = Finding(
        asset_type=AssetType.BATCH,
        asset_ref=f"batch:{ref_profile.profile_id}:{batch_id}",
        detector="cvguard.driftplane.distribution_verifier:v1.0",
        reason=f"[{classification.upper()}] {reason_text}",
        evidence=evidence_strings,
        confidence=confidence,
        severity=severity,
        disposition=disposition,
        assumptions=[
            f"Reference profile '{ref_profile.profile_id}:v{ref_profile.version}' models declared normal state.",
            "Embedding space preserves visual domain geometry.",
        ],
        limitations=limitations,
    )

    signed_finding = await dispatch_finding_to_governance(finding)

    return AssessBatchResponse(
        batch_id=batch_id,
        profile_id=ref_profile.profile_id,
        profile_version=ref_profile.version,
        sample_count=len(batch_embeddings),
        mmd_statistic=mmd_stat,
        mmd_standard_error=mmd_se,
        ks_shifted_dimension_ratio=ks_results["shifted_dimension_ratio"],
        ks_mean_statistic=ks_results["mean_ks_statistic"],
        risk_score=risk_score,
        confidence_interval=[ci_lower, ci_upper],
        standard_error=std_error,
        classification=classification,
        reason=reason_text,
        evidence=evidence_strings,
        correlated_metadata_field=correlated_field,
        governance_ledger_id=signed_finding.ledger_id,
        signed_finding=signed_finding.model_dump(mode="json"),
        limitations=limitations,
    )


@app.get("/profiles")
async def list_profiles() -> list[dict[str, Any]]:
    """List all registered versioned reference profiles."""
    return get_profile_registry().list_profiles()


@app.get("/profiles/{profile_id}")
async def get_latest_profile(profile_id: str) -> dict[str, Any]:
    """Retrieve summary metadata for the latest version of a profile."""
    prof = get_profile_registry().get_profile(profile_id)
    if prof is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return prof.to_summary_dict()


@app.get("/profiles/{profile_id}/{version}")
async def get_versioned_profile(profile_id: str, version: int) -> dict[str, Any]:
    """Retrieve summary metadata for a specific version of a profile."""
    prof = get_profile_registry().get_profile(profile_id, version)
    if prof is None:
        raise HTTPException(
            status_code=404,
            detail=f"Profile '{profile_id}' version {version} not found.",
        )
    return prof.to_summary_dict()


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
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=False, **ssl_kwargs)
