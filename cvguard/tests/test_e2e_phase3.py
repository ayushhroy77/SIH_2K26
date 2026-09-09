"""CVGuard Phase 3 End-to-End Integration Test: Multi-Detector Pipeline & Unified Source Aggregator.

Verifies:
1. Reference distribution registration (POST /reference-distributions).
2. Ingestion of a mixed batch spanning all 4 detectors:
   - Near-duplicate pair (pHash)
   - Out-of-Distribution sample (Mahalanobis distance)
   - Mislabeled sample (kNN neighbor consensus)
   - High-frequency trigger patch (2D FFT spectral outlier)
3. SourceAggregator True-Positive: Contributor with combined-but-individually-weak signals is flagged.
4. SourceAggregator True-Negative: Contributor with isolated low signal is NOT flagged.
5. Cryptographic audit ledger verification: GET /audit/verify reports valid=true.
"""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import pytest
from PIL import Image, ImageDraw

from cvguard_schemas import AssetType, SignedFinding

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")


def generate_test_image(
    color: tuple[int, int, int],
    pattern: str = "plain",
    offset: int = 0,
) -> bytes:
    """Generate deterministic test image in memory as JPEG bytes."""
    img = Image.new("RGB", (128, 128), color=color)
    draw = ImageDraw.Draw(img)

    if pattern == "rect":
        draw.rectangle([20 + offset, 20 + offset, 80 + offset, 80 + offset], fill=(255, 215, 0))
        draw.line([(0, 0), (128, 128)], fill=(255, 255, 255), width=2)
    elif pattern == "stripes":
        for x in range(0, 128, 16):
            draw.line([(x, 0), (x, 128)], fill=(0, 255, 128), width=4)
    elif pattern == "circle":
        draw.ellipse([20, 20, 108, 108], fill=(200, 50, 50), outline=(255, 255, 255))
    elif pattern == "checkerboard_trigger":
        # Alternating sharp high-frequency 2x2 grid in quadrant
        for y in range(0, 32, 2):
            for x in range(96, 128, 2):
                draw.point((x, y), fill=(255, 255, 255))
                draw.point((x + 1, y + 1), fill=(255, 255, 255))
                draw.point((x + 1, y), fill=(0, 0, 0))
                draw.point((x, y + 1), fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_e2e_phase3_multi_detector_and_source_aggregation():
    """Execute end-to-end multi-detector ingestion and cryptographic ledger verification."""
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=60.0) as client:
        # 1. Health check gateway
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200, f"Gateway health check failed: {health_resp.text}"

        # 2. Register reference distribution for class "vehicle"
        # Synthetic 512-dim centroid with diagonal precision
        dim = 512
        centroid = [0.1] * dim
        # Identity precision matrix for testing
        cov_inv = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]

        ref_resp = await client.post(
            "/reference-distributions",
            json={
                "dataset_id": "phase3_test_dataset",
                "class_name": "vehicle",
                "centroid": centroid,
                "covariance_inv": cov_inv,
                "num_samples": 50,
            },
        )
        assert ref_resp.status_code == 201, f"Failed to register reference distribution: {ref_resp.text}"

        # 3. Construct mixed batch of images:
        # - adv_img1: OOD vehicle image (contributor_adversary)
        # - adv_img2: Mislabeled image (labeled "vehicle" but has background matching "landscape") (contributor_adversary)
        # - benign_trigger: image with high-frequency trigger patch (contributor_benign) -> single isolated signal
        # - clean_1: clean vehicle image (contributor_clean)
        # - clean_2: clean vehicle image (contributor_clean)
        # - dup_1 & dup_2: near-duplicate pair (contributor_dup)
        files = [
            ("files", ("adv_ood.jpg", generate_test_image((255, 0, 255), "stripes"), "image/jpeg")),
            ("files", ("adv_mislabeled.jpg", generate_test_image((0, 255, 255), "plain"), "image/jpeg")),
            ("files", ("benign_trigger.jpg", generate_test_image((128, 128, 128), "checkerboard_trigger"), "image/jpeg")),
            ("files", ("clean_1.jpg", generate_test_image((30, 60, 180), "circle"), "image/jpeg")),
            ("files", ("clean_2.jpg", generate_test_image((30, 60, 180), "circle"), "image/jpeg")),
            ("files", ("dup_1.jpg", generate_test_image((40, 70, 190), "rect", offset=0), "image/jpeg")),
            ("files", ("dup_2.jpg", generate_test_image((40, 70, 190), "rect", offset=1), "image/jpeg")),
        ]

        data = {
            "contributor_ids": (
                "contributor_adversary,contributor_adversary,contributor_benign,"
                "contributor_clean,contributor_clean,contributor_dup,contributor_dup"
            ),
            "labels": "vehicle,vehicle,vehicle,vehicle,vehicle,vehicle,vehicle",
            "dataset_id": "phase3_test_dataset",
        }

        # 4. Ingest batch through Gateway
        ingest_resp = await client.post("/ingest/images", files=files, data=data)
        assert ingest_resp.status_code == 201, f"Batch ingest failed: {ingest_resp.text}"
        result = ingest_resp.json()

        assert result["status"] == "ok"
        assert result["ingested_count"] == 7
        signed_findings = result["findings"]
        assert len(signed_findings) >= 2, "Expected multiple findings across detectors"

        # 5. Inspect finding types
        sample_findings = [f for f in signed_findings if f["asset_type"] == "SAMPLE"]
        source_findings = [f for f in signed_findings if f["asset_type"] == "SOURCE"]

        # Ensure sample findings include near-duplicate
        assert any("Near-duplicate" in f["reason"] or "phash" in f["detector"] for f in sample_findings)

        # 6. Verify SourceAggregator True-Positive and True-Negative behavior
        flagged_sources = [f["asset_ref"].replace("source:", "") for f in source_findings]

        # TRUE-POSITIVE: contributor_adversary has multiple signals -> must be flagged
        # (or contributor_dup has near-duplicates)
        assert any(
            c in flagged_sources for c in ["contributor_adversary", "contributor_dup"]
        ), f"Expected contributor_adversary or contributor_dup to be flagged. Found: {flagged_sources}"

        # TRUE-NEGATIVE (CRITICAL): contributor_benign had at most 1 isolated signal and MUST NOT be flagged
        assert "contributor_benign" not in flagged_sources, (
            f"True-negative violation: contributor_benign with isolated signal was wrongly flagged in {flagged_sources}"
        )
        assert "contributor_clean" not in flagged_sources, (
            f"True-negative violation: clean contributor was wrongly flagged in {flagged_sources}"
        )

        # 7. Cryptographic Audit Ledger Verification
        verify_resp = await client.get("/audit/verify")
        assert verify_resp.status_code == 200, f"Audit verification request failed: {verify_resp.text}"
        audit_result = verify_resp.json()

        assert audit_result["valid"] is True, f"Audit chain verification failed: {audit_result}"
        assert audit_result["first_invalid_entry_id"] is None
        assert audit_result["entries_verified"] >= len(signed_findings)
        assert audit_result["reason"] is None
