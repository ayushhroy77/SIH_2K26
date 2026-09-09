"""CVGuard Phase 2 End-to-End Pipeline Integration Test.

Tests the full vertical slice:
Image Ingest -> Near-Duplicate pHash Detector -> Governance Spine Signing ->
Append-Only Hash-Chained Audit Ledger -> Gateway Proxy -> Cryptographic Verification.
"""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import pytest
from PIL import Image, ImageDraw

from cvguard_schemas import AssetType, Finding, Severity, SignedFinding
from cvguard_schemas.security import create_test_jwt


def create_synthetic_test_image(
    base_color: tuple[int, int, int],
    shape_type: str = "rect",
    offset: int = 0,
) -> bytes:
    """Generate deterministic synthetic test image in memory as JPEG bytes."""
    img = Image.new("RGB", (128, 128), color=base_color)
    draw = ImageDraw.Draw(img)

    if shape_type == "rect":
        # Draw contrasting rectangles
        draw.rectangle([20 + offset, 20 + offset, 80 + offset, 80 + offset], fill=(255, 215, 0))
        draw.line([(0, 0), (128, 128)], fill=(255, 255, 255), width=2)
    elif shape_type == "stripes":
        for x in range(0, 128, 16):
            draw.line([(x, 0), (x, 128)], fill=(0, 255, 128), width=4)
    elif shape_type == "circle":
        draw.ellipse([20, 20, 108, 108], fill=(200, 50, 50), outline=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture(scope="module")
def image_test_set() -> dict[str, Any]:
    """Build test image fixture with 2 real near-duplicates and 2 unrelated images.

    - image_1 & image_2: Near-duplicate pair (offset by only 1px; pHash Hamming distance <= 2).
      Both share contributor_id = "contributor-shared-adversary".
    - image_3: Unrelated vertical striped pattern (contributor_id = "contributor-benign-1").
    - image_4: Unrelated circular geometry (contributor_id = "contributor-benign-2").
    """
    img1_bytes = create_synthetic_test_image((30, 60, 180), shape_type="rect", offset=0)
    # img2 is a near-duplicate: same color palette and geometric structure, offset by 1 pixel
    img2_bytes = create_synthetic_test_image((30, 60, 180), shape_type="rect", offset=1)
    # img3 & img4 are completely distinct image classes
    img3_bytes = create_synthetic_test_image((10, 10, 10), shape_type="stripes")
    img4_bytes = create_synthetic_test_image((220, 220, 220), shape_type="circle")

    return {
        "files": [
            ("files", ("dataset_dup_alpha.jpg", img1_bytes, "image/jpeg")),
            ("files", ("dataset_dup_beta.jpg", img2_bytes, "image/jpeg")),
            ("files", ("dataset_stripes.jpg", img3_bytes, "image/jpeg")),
            ("files", ("dataset_circle.jpg", img4_bytes, "image/jpeg")),
        ],
        "contributor_ids": (
            "contributor-shared-adversary,contributor-shared-adversary,"
            "contributor-benign-1,contributor-benign-2"
        ),
    }


def test_detector_evaluation_offline_pure():
    """Verify NearDuplicateDetector logic in isolation without network dependencies."""
    import sys
    sys.path.insert(0, os.path.abspath("services/data-plane"))

    from detector import IngestedImage, NearDuplicateDetector, compute_image_phash

    img1_bytes = create_synthetic_test_image((30, 60, 180), shape_type="rect", offset=0)
    img2_bytes = create_synthetic_test_image((30, 60, 180), shape_type="rect", offset=1)
    img3_bytes = create_synthetic_test_image((10, 10, 10), shape_type="stripes")
    img4_bytes = create_synthetic_test_image((220, 220, 220), shape_type="circle")

    phash1 = compute_image_phash(img1_bytes)
    phash2 = compute_image_phash(img2_bytes)
    phash3 = compute_image_phash(img3_bytes)
    phash4 = compute_image_phash(img4_bytes)

    batch = [
        IngestedImage(
            id=1,
            filename="dup_alpha.jpg",
            sha256="1111111111111111111111111111111111111111111111111111111111111111",
            minio_key="k1_dup_alpha.jpg",
            contributor_id="contributor-shared-adversary",
            phash=phash1,
        ),
        IngestedImage(
            id=2,
            filename="dup_beta.jpg",
            sha256="2222222222222222222222222222222222222222222222222222222222222222",
            minio_key="k2_dup_beta.jpg",
            contributor_id="contributor-shared-adversary",
            phash=phash2,
        ),
        IngestedImage(
            id=3,
            filename="stripes.jpg",
            sha256="3333333333333333333333333333333333333333333333333333333333333333",
            minio_key="k3_stripes.jpg",
            contributor_id="contributor-benign-1",
            phash=phash3,
        ),
        IngestedImage(
            id=4,
            filename="circle.jpg",
            sha256="4444444444444444444444444444444444444444444444444444444444444444",
            minio_key="k4_circle.jpg",
            contributor_id="contributor-benign-2",
            phash=phash4,
        ),
    ]

    detector = NearDuplicateDetector(threshold=10)
    findings = detector.evaluate_batch(batch)

    # 1. Confirm finding with asset_type=SAMPLE exists for the duplicate pair
    sample_findings = [f for f in findings if f.asset_type == AssetType.SAMPLE]
    assert len(sample_findings) == 1, f"Expected 1 SAMPLE finding, got {len(sample_findings)}"
    assert "dup_alpha.jpg" in sample_findings[0].reason
    assert "dup_beta.jpg" in sample_findings[0].reason
    assert "k1_dup_alpha.jpg" in sample_findings[0].evidence
    assert "k2_dup_beta.jpg" in sample_findings[0].evidence
    assert sample_findings[0].confidence >= 0.90

    # 2. Confirm finding with asset_type=SOURCE exists for shared contributor
    source_findings = [f for f in findings if f.asset_type == AssetType.SOURCE]
    assert len(source_findings) == 1, f"Expected 1 SOURCE finding, got {len(source_findings)}"
    assert source_findings[0].asset_ref == "source:contributor-shared-adversary"
    assert source_findings[0].severity == Severity.HIGH


def test_full_pipeline_via_gateway(image_test_set):
    """Execute end-to-end HTTP pipeline against Gateway service.

    Flow:
    1. POST /ingest/images via Gateway
    2. Confirm SAMPLE finding for duplicate pair
    3. Confirm SOURCE finding for shared contributor
    4. GET /findings confirms both are queryable
    5. GET /audit/verify confirms hash chain validity
    """
    gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8000")

    # Probe gateway health; skip live HTTP test if gateway service is offline
    try:
        health_resp = httpx.get(f"{gateway_url}/health", timeout=3.0)
        if health_resp.status_code != 200:
            pytest.skip(f"Gateway at {gateway_url} returned status {health_resp.status_code}")
    except Exception as exc:
        pytest.skip(f"Live Gateway not reachable at {gateway_url} ({exc}); skipping network E2E run.")

    # Setup auth token for hardened gateway
    auth_headers = {"Authorization": f"Bearer {create_test_jwt(roles=['analyst', 'admin', 'auditor'])}"}

    # 1. POST /ingest/images with test image set
    files = image_test_set["files"]
    data = {
        "contributor_id": "contributor-fallback",
        "contributor_ids": image_test_set["contributor_ids"],
    }

    ingest_resp = httpx.post(
        f"{gateway_url}/ingest/images",
        files=files,
        data=data,
        headers=auth_headers,
        timeout=30.0,
    )
    assert ingest_resp.status_code == 201, (
        f"Ingest failed: HTTP {ingest_resp.status_code} - {ingest_resp.text}"
    )
    ingest_data = ingest_resp.json()
    assert ingest_data["status"] == "ok"
    assert ingest_data["ingested_count"] == 4

    returned_findings = ingest_data["findings"]
    assert len(returned_findings) >= 2, (
        f"Expected at least 2 findings (SAMPLE + SOURCE), got {len(returned_findings)}"
    )

    # 2. Confirm SAMPLE finding for duplicate pair
    sample_findings = [
        f for f in returned_findings if f["finding"]["asset_type"] == AssetType.SAMPLE.value
    ]
    assert len(sample_findings) >= 1, "Expected SAMPLE finding for duplicate pair"
    sample_finding = sample_findings[0]["finding"]
    assert "dataset_dup_alpha.jpg" in sample_finding["reason"]
    assert "dataset_dup_beta.jpg" in sample_finding["reason"]
    assert len(sample_finding["evidence"]) == 2

    # 3. Confirm SOURCE finding for shared contributor
    source_findings = [
        f for f in returned_findings if f["finding"]["asset_type"] == AssetType.SOURCE.value
    ]
    assert len(source_findings) >= 1, "Expected SOURCE finding for contributor-shared-adversary"
    source_finding = source_findings[0]["finding"]
    assert source_finding["asset_ref"] == "source:contributor-shared-adversary"
    assert source_finding["severity"] == Severity.HIGH.value

    # 4. Query GET /findings via Gateway
    findings_resp = httpx.get(f"{gateway_url}/findings?limit=20", headers=auth_headers, timeout=10.0)
    assert findings_resp.status_code == 200
    all_findings = findings_resp.json()
    ledger_ids = [item["ledger_id"] for item in all_findings]
    assert sample_findings[0]["ledger_id"] in ledger_ids
    assert source_findings[0]["ledger_id"] in ledger_ids

    # 5. GET /audit/verify reports valid=true afterward
    verify_resp = httpx.get(f"{gateway_url}/audit/verify", headers=auth_headers, timeout=10.0)
    assert verify_resp.status_code == 200
    audit_data = verify_resp.json()
    assert audit_data["valid"] is True, f"Audit verification failed: {audit_data}"
    assert audit_data["first_invalid_entry_id"] is None
    assert audit_data["entries_checked"] >= 2
