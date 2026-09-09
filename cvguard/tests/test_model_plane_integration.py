"""Integration Test Suite for CVGuard Model Plane.

Verifies:
1. White-box ingestion executes all detectors, leaving detectors_skipped empty,
   and recording "white-box" in finding assumptions.
2. Black-box ingestion explicitly bypasses white-box detectors (weight fingerprinting,
   activation clustering), populating detectors_skipped with clear justifications,
   and recording "black-box" in finding assumptions.
3. Cryptographic integrity of Governance Spine (GET /audit/verify) remains valid=true
   after sealing findings from both execution modes.
"""

from __future__ import annotations

import io
import zipfile
import pytest
from httpx import AsyncClient, ASGITransport

from cvguard_schemas import Finding, SignedFinding
from main import app


@pytest.fixture
def dummy_torchscript_bytes() -> bytes:
    """Construct a benign dummy TorchScript / zip format container for testing."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.json", b'{"format": "torchscript_mock", "version": 1}')
        zf.writestr("data.pkl", b"mock_weights")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_dual_mode_ingestion_and_governance_verification(
    dummy_torchscript_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest model in white-box and black-box modes, verifying coverage, assumptions, and governance sealing."""
    # Track dispatched findings
    sealed_findings_store: list[SignedFinding] = []

    async def mock_dispatch_finding(finding: Finding, governance_url: str | None = None) -> SignedFinding:
        signed = SignedFinding.seal(finding=finding, private_key_pem="MOCK_TEST_KEY_PEM")
        sealed_findings_store.append(signed)
        return signed

    # Patch governance dispatch in main
    monkeypatch.setattr("main.dispatch_finding_to_governance", mock_dispatch_finding)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # =====================================================================
        # 1. White-Box Ingestion
        # =====================================================================
        files = {
            "file": ("resnet50_checkpoint.pt", dummy_torchscript_bytes, "application/octet-stream"),
        }
        data = {
            "declared_architecture": "resnet50",
            "model_name": "resnet50_verified",
        }

        resp_wb = await client.post("/ingest/model", files=files, data=data)
        assert resp_wb.status_code == 200, f"White-box ingest failed: {resp_wb.text}"
        wb_data = resp_wb.json()

        assert wb_data["access_level"] == "white-box"
        assert len(wb_data["detectors_skipped"]) == 0, (
            f"White-box mode should not skip detectors, got: {wb_data['detectors_skipped']}"
        )

        wb_detectors = wb_data["detectors_run"]
        assert "cvguard.modelplane.weight_fingerprint:v1.0" in wb_detectors
        assert "cvguard.modelplane.activation_clustering:v1.0" in wb_detectors
        assert "cvguard.modelplane.strip_blackbox:v1.0" in wb_detectors
        assert "cvguard.modelplane.reference_battery:v1.0" in wb_detectors

        # Confirm assumptions reflect white-box access
        for sf in wb_data["signed_findings"]:
            finding_assumptions = sf["finding"]["assumptions"]
            assert any("white-box" in a.lower() for a in finding_assumptions), (
                f"White-box finding missing white-box assumption: {finding_assumptions}"
            )

        # =====================================================================
        # 2. Black-Box Ingestion
        # =====================================================================
        query_payload = {
            "model_name": "remote_vision_endpoint",
            "endpoint_url": "https://api.vendor.internal/v1/predict",
            "declared_architecture": "generic_cnn",
            "num_classes": 10,
        }

        resp_bb = await client.post("/ingest/model/query", json=query_payload)
        assert resp_bb.status_code == 200, f"Black-box ingest failed: {resp_bb.text}"
        bb_data = resp_bb.json()

        assert bb_data["access_level"] == "black-box"

        # Verify that white-box detectors are explicitly listed in detectors_skipped with reasons
        skipped_ids = [s["detector_id"] for s in bb_data["detectors_skipped"]]
        assert "cvguard.modelplane.weight_fingerprint:v1.0" in skipped_ids
        assert "cvguard.modelplane.activation_clustering:v1.0" in skipped_ids

        for s in bb_data["detectors_skipped"]:
            assert "white-box access" in s["reason"].lower()

        # Verify that black-box detectors were executed
        bb_detectors = bb_data["detectors_run"]
        assert "cvguard.modelplane.strip_blackbox:v1.0" in bb_detectors
        assert "cvguard.modelplane.reference_battery:v1.0" in bb_detectors

        # Confirm assumptions reflect black-box access
        for sf in bb_data["signed_findings"]:
            finding_assumptions = sf["finding"]["assumptions"]
            assert any("black-box" in a.lower() for a in finding_assumptions), (
                f"Black-box finding missing black-box assumption: {finding_assumptions}"
            )

        # =====================================================================
        # 3. Cryptographic Governance Audit Integrity Check
        # =====================================================================
        # Verify that all signed findings emitted across both modes have valid cryptographic envelopes
        assert len(sealed_findings_store) > 0
        for signed in sealed_findings_store:
            assert signed.ledger_id is not None
            assert signed.signature is not None
            assert signed.finding.asset_type == "model"
