"""Test Suite for CVGuard Drift Plane (Phase 6).

Verifies:
1. Synthetic clean batch drawn from identical distribution scores low risk and classifies as neither drift nor manipulation (no_drift).
2. Synthetic batch with distribution shift correlated with a changed sensor_id classifies as probable_operational_drift.
3. Synthetic batch with an isolated, metadata-uncorrelated anomalous subset classifies as suspicious_manipulation.
4. Governance Spine audit hash-chain integrity (GET /audit/verify) remains valid=true after emitting drift findings.
5. Material distribution shift without metadata explicitly resolves to indeterminate rather than forcing an ungrounded guess.
6. MMD and Kolmogorov-Smirnov statistical divergence tests accurately quantify distribution distance.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure services are on sys.path for direct testing
_BASE_DIR = Path(__file__).resolve().parent.parent
_DRIFT_PLANE_DIR = _BASE_DIR / "services" / "drift-plane"
_DATA_PLANE_DIR = _BASE_DIR / "services" / "data-plane"
_GOVERNANCE_DIR = _BASE_DIR / "services" / "governance"
_SCHEMAS_DIR = _BASE_DIR / "libs" / "schemas"

for p in [_DRIFT_PLANE_DIR, _DATA_PLANE_DIR, _GOVERNANCE_DIR, _SCHEMAS_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cvguard_schemas
from cvguard_schemas import AssetType, Finding, SignedFinding

from main import app
from profiles import ReferenceProfile, get_profile_registry
from stats import (
    classify_drift_vs_manipulation,
    compute_calibrated_risk,
    compute_ks_tests,
    compute_mmd,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the reference profile registry before and after each test."""
    registry = get_profile_registry()
    registry.clear()
    yield
    registry.clear()


def generate_gaussian_vector(dim: int, mean: float = 0.0, std: float = 1.0, rng: random.Random | None = None) -> list[float]:
    """Generate a pseudo-random synthetic embedding vector on a hypersphere."""
    r = rng or random.Random(42)
    vec = [r.gauss(mean, std) for _ in range(dim)]
    norm = math.sqrt(sum(v ** 2 for v in vec))
    return [round(v / max(norm, 1e-12), 6) for v in vec]


def generate_sample_distribution(
    n: int,
    dim: int = 32,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 100,
) -> list[list[float]]:
    """Generate a batch of synthetic embedding vectors from a specified distribution."""
    rng = random.Random(seed)
    return [generate_gaussian_vector(dim=dim, mean=mean, std=std, rng=rng) for _ in range(n)]


# ==============================================================================
# 1. STATISTICAL ALGORITHM TESTS
# ==============================================================================

def test_mmd_identical_vs_shifted_distributions():
    """Ensure MMD divergence is near zero for identical distributions and elevated for shifted distributions."""
    dim = 16
    samples_a = generate_sample_distribution(n=40, dim=dim, mean=0.0, std=1.0, seed=1)
    samples_b = generate_sample_distribution(n=40, dim=dim, mean=0.0, std=1.0, seed=2)
    samples_shifted = generate_sample_distribution(n=40, dim=dim, mean=3.0, std=2.0, seed=3)

    # Identical underlying distribution
    mmd_null, se_null = compute_mmd(samples_a, samples_b)
    assert mmd_null < 0.08, f"Null MMD should be close to zero, got {mmd_null}"
    assert se_null > 0.0

    # Divergent underlying distribution
    mmd_alt, se_alt = compute_mmd(samples_a, samples_shifted)
    assert mmd_alt > mmd_null * 3, f"Shifted MMD ({mmd_alt}) should be significantly larger than null ({mmd_null})"


def test_ks_per_dimension_testing():
    """Ensure per-dimension Kolmogorov-Smirnov detects shifted dimensions."""
    dim = 16
    samples_ref = generate_sample_distribution(n=50, dim=dim, mean=0.0, std=1.0, seed=10)
    # Shift half of the dimensions
    samples_test = []
    rng = random.Random(20)
    for _ in range(50):
        vec = generate_gaussian_vector(dim=dim, mean=0.0, std=1.0, rng=rng)
        # Add shift to first 8 dimensions
        for d in range(8):
            vec[d] += 0.8
        # Re-normalize
        norm = math.sqrt(sum(v ** 2 for v in vec))
        samples_test.append([v / norm for v in vec])

    ks_res = compute_ks_tests(samples_ref, samples_test)
    assert ks_res["total_dimensions"] == dim
    assert ks_res["shifted_dimension_count"] >= 4
    assert ks_res["shifted_dimension_ratio"] > 0.20


# ==============================================================================
# 2. REQUIRED SCENARIO 1: CLEAN BATCH (LOW RISK / NO DRIFT)
# ==============================================================================

@pytest.mark.asyncio
async def test_clean_batch_low_risk_no_drift():
    """TASK 5 TEST 1: A synthetic clean batch drawn from the same distribution as the
    reference profile — should score low risk, classify as neither drift nor manipulation.
    """
    dim = 16
    ref_samples = [
        {"embedding": vec, "metadata": {"sensor_id": "sensor_north", "season": "summer"}}
        for vec in generate_sample_distribution(n=40, dim=dim, mean=0.0, std=1.0, seed=100)
    ]
    query_samples = [
        {"embedding": vec, "metadata": {"sensor_id": "sensor_north", "season": "summer"}}
        for vec in generate_sample_distribution(n=30, dim=dim, mean=0.0, std=1.0, seed=101)
    ]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Ingest reference profile
        ref_resp = await client.post(
            "/reference-profile",
            json={"profile_id": "prof_clean_test", "samples": ref_samples},
        )
        assert ref_resp.status_code == 201
        ref_data = ref_resp.json()
        assert ref_data["profile_id"] == "prof_clean_test"
        assert ref_data["num_samples"] == 40

        # 2. Assess clean batch
        assess_resp = await client.post(
            "/assess-batch",
            json={"profile_id": "prof_clean_test", "samples": query_samples},
        )
        assert assess_resp.status_code == 200
        result = assess_resp.json()

        # Should score low risk
        assert result["risk_score"] < 0.35
        # Classify as neither drift nor manipulation
        assert result["classification"] == "no_drift"
        assert "normal baseline" in result["reason"].lower()
        assert result["correlated_metadata_field"] is None
        # Confidence interval must be valid
        assert result["confidence_interval"][0] <= result["risk_score"] <= result["confidence_interval"][1]


# ==============================================================================
# 3. REQUIRED SCENARIO 2: SENSOR_ID CORRELATED OPERATIONAL DRIFT
# ==============================================================================

@pytest.mark.asyncio
async def test_sensor_correlated_operational_drift():
    """TASK 5 TEST 2: A synthetic batch with a shift correlated with a changed sensor_id
    metadata field — should classify as probable_operational_drift.
    """
    dim = 16
    ref_samples = [
        {"embedding": vec, "metadata": {"sensor_id": "cam_standard_01", "season": "spring"}}
        for vec in generate_sample_distribution(n=40, dim=dim, mean=0.0, std=1.0, seed=200)
    ]

    # Query batch: shifted samples all share a new sensor_id "cam_upgraded_02"
    rng_shift = random.Random(300)
    query_samples = []

    # 10 non-shifted samples from cam_standard_01
    for _ in range(10):
        vec = generate_gaussian_vector(dim=dim, mean=0.0, std=1.0, rng=rng_shift)
        query_samples.append({"embedding": vec, "metadata": {"sensor_id": "cam_standard_01"}})

    # 25 shifted samples with heavy offset from cam_upgraded_02
    for _ in range(25):
        vec = generate_gaussian_vector(dim=dim, mean=2.5, std=1.2, rng=rng_shift)
        query_samples.append({"embedding": vec, "metadata": {"sensor_id": "cam_upgraded_02"}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Ingest reference profile
        await client.post(
            "/reference-profile",
            json={"profile_id": "prof_sensor_test", "samples": ref_samples},
        )

        # Assess batch
        assess_resp = await client.post(
            "/assess-batch",
            json={"profile_id": "prof_sensor_test", "samples": query_samples},
        )
        assert assess_resp.status_code == 200
        result = assess_resp.json()

        # Must detect shift
        assert result["risk_score"] >= 0.25
        # Must classify cleanly as operational drift
        assert result["classification"] == "probable_operational_drift"
        assert result["correlated_metadata_field"] == "sensor_id"
        assert "cam_upgraded_02" in result["reason"]
        assert "operational" in result["reason"].lower()


# ==============================================================================
# 4. REQUIRED SCENARIO 3: ISOLATED METADATA-UNCORRELATED MANIPULATION
# ==============================================================================

@pytest.mark.asyncio
async def test_isolated_uncorrelated_suspicious_manipulation():
    """TASK 5 TEST 3: A synthetic batch with an isolated, metadata-uncorrelated anomalous
    subset — should classify as suspicious_manipulation.
    """
    dim = 16
    ref_samples = [
        {"embedding": vec, "metadata": {"sensor_id": "cam_secure", "illumination": "day"}}
        for vec in generate_sample_distribution(n=40, dim=dim, mean=0.0, std=1.0, seed=400)
    ]

    # Query batch: 30 total samples
    # 26 normal samples matching reference metadata
    # 4 isolated perturbed samples (13% of batch) with IDENTICAL metadata (cam_secure, illumination=day)
    rng = random.Random(500)
    query_samples = []

    for _ in range(26):
        vec = generate_gaussian_vector(dim=dim, mean=0.0, std=1.0, rng=rng)
        query_samples.append({"embedding": vec, "metadata": {"sensor_id": "cam_secure", "illumination": "day"}})

    # Isolated manipulated samples (backdoor / trigger perturbations)
    for _ in range(4):
        vec = generate_gaussian_vector(dim=dim, mean=4.0, std=0.5, rng=rng)
        query_samples.append({"embedding": vec, "metadata": {"sensor_id": "cam_secure", "illumination": "day"}})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Ingest reference profile
        await client.post(
            "/reference-profile",
            json={"profile_id": "prof_manip_test", "samples": ref_samples},
        )

        # Assess batch
        assess_resp = await client.post(
            "/assess-batch",
            json={"profile_id": "prof_manip_test", "samples": query_samples},
        )
        assert assess_resp.status_code == 200
        result = assess_resp.json()

        assert result["risk_score"] >= 0.25
        assert result["classification"] == "suspicious_manipulation"
        assert result["correlated_metadata_field"] is None
        assert "isolated" in result["reason"].lower()
        assert "manipulation" in result["reason"].lower()


# ==============================================================================
# 5. REQUIRED SCENARIO 4: GOVERNANCE AUDIT CHAIN CONTINUITY
# ==============================================================================

@pytest.mark.asyncio
async def test_governance_audit_verify_after_traffic(monkeypatch: pytest.MonkeyPatch):
    """TASK 5 TEST 4: Confirm GET /audit/verify (governance) still reports valid=true after this traffic."""
    # Wire in the real Governance Spine LocalFileSigner and AuditLedger
    from ledger import AuditLedger
    from signer import LocalFileSigner

    signer = LocalFileSigner()
    governance_ledger = AuditLedger(signer=signer)

    async def mock_dispatch_governance(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        assert finding.asset_type == AssetType.BATCH
        return governance_ledger.append_entry(finding)

    monkeypatch.setattr("main.dispatch_finding_to_governance", mock_dispatch_governance)

    dim = 16
    ref_samples = [
        {"embedding": vec, "metadata": {"sensor_id": "cam_audit_01"}}
        for vec in generate_sample_distribution(n=30, dim=dim, mean=0.0, std=1.0, seed=600)
    ]
    query_samples = [
        {"embedding": vec, "metadata": {"sensor_id": "cam_audit_01"}}
        for vec in generate_sample_distribution(n=20, dim=dim, mean=1.5, std=1.0, seed=601)
    ]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create profile
        await client.post(
            "/reference-profile",
            json={"profile_id": "prof_gov_audit", "samples": ref_samples},
        )

        # Run multiple batch assessments to generate finding traffic
        for i in range(3):
            resp = await client.post(
                "/assess-batch",
                json={"profile_id": "prof_gov_audit", "batch_id": f"gov_batch_{i}", "samples": query_samples},
            )
            assert resp.status_code == 200
            assert resp.json()["governance_ledger_id"] is not None

        # Verify Governance Spine audit ledger chain
        audit_result = governance_ledger.verify_chain()
        assert audit_result.valid is True
        assert audit_result.total_entries >= 3
        assert audit_result.corrupted_entry_id is None
        assert "verified successfully" in audit_result.reason.lower()


# ==============================================================================
# 6. EXPLICIT INDETERMINATE BRANCH TEST
# ==============================================================================

@pytest.mark.asyncio
async def test_indeterminate_classification_when_metadata_missing():
    """Verify that a material shift without metadata resolves to indeterminate,
    never falsely forcing drift or manipulation.
    """
    dim = 16
    ref_samples = [
        {"embedding": vec, "metadata": {}}
        for vec in generate_sample_distribution(n=30, dim=dim, mean=0.0, std=1.0, seed=700)
    ]
    # Heavy shift across entire query batch, but NO metadata provided
    query_samples = [
        {"embedding": vec, "metadata": {}}
        for vec in generate_sample_distribution(n=30, dim=dim, mean=3.0, std=1.5, seed=701)
    ]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/reference-profile",
            json={"profile_id": "prof_indet_test", "samples": ref_samples},
        )

        assess_resp = await client.post(
            "/assess-batch",
            json={"profile_id": "prof_indet_test", "samples": query_samples},
        )
        assert assess_resp.status_code == 200
        result = assess_resp.json()

        # Shift is material
        assert result["risk_score"] >= 0.25
        # Must resolve to indeterminate because no metadata is present
        assert result["classification"] == "indeterminate"
        assert "indeterminate" in result["reason"].lower()
        assert "no sample metadata was provided" in result["reason"].lower()
