"""Unit tests for CVGuard Canonical Pydantic v2 Schemas (v0.2.0)."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from cvguard_schemas import (
    AssetType,
    CoverageStatement,
    Disposition,
    Finding,
    Report,
    Severity,
    SignedFinding,
)


def test_valid_finding_creation() -> None:
    """Ensure a Finding record can be constructed with mandatory and default fields."""
    finding = Finding(
        asset_type=AssetType.MODEL,
        asset_ref="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        detector="cvguard.detector.model_integrity:v1.0",
        reason="Model checkpoint contains anomalous weight magnitude deviation in conv2d_4 layer.",
        evidence=["layer:conv2d_4", "max_abs_val:184.2", "threshold:25.0"],
        confidence=0.985,
        severity=Severity.HIGH,
        disposition=Disposition.QUARANTINE,
        assumptions=["Standard Gaussian weight distribution expected in FP32 format."],
        limitations=["Only layer weights evaluated, not graph topology."],
    )

    assert finding.asset_type == AssetType.MODEL
    assert finding.confidence == 0.985
    assert finding.severity == Severity.HIGH
    assert finding.disposition == Disposition.QUARANTINE
    assert finding.created_at <= datetime.now(timezone.utc)
    assert not hasattr(finding, "signature")
    assert len(finding.evidence) == 3


def test_finding_asset_type_locked_enum() -> None:
    """Ensure free-form string asset types are rejected in favor of locked AssetType enum."""
    # Invalid free-form string must fail
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            asset_type="unknown_free_form_asset",  # type: ignore[arg-type]
            asset_ref="img_001.png",
            detector="test_det",
            reason="test reason",
            confidence=0.5,
            severity=Severity.LOW,
            disposition=Disposition.ACCEPT,
        )
    assert "asset_type" in str(exc_info.value)

    # Valid enum values succeed
    for valid_asset in [AssetType.SAMPLE, AssetType.SOURCE, AssetType.MODEL, AssetType.INFERENCE_RECORD, AssetType.BATCH]:
        f = Finding(
            asset_type=valid_asset,
            asset_ref="asset_ref_test",
            detector="test_det",
            reason="valid asset enum test",
            confidence=0.8,
            severity=Severity.INFO,
            disposition=Disposition.ACCEPT,
        )
        assert f.asset_type == valid_asset


def test_signed_finding_envelope() -> None:
    """Ensure SignedFinding cleanly encapsulates Finding, cryptographic hashes, signature, and ledger ID."""
    finding = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="sample_001.png",
        detector="test_detector:v1",
        reason="Test anomaly detection",
        confidence=0.75,
        severity=Severity.MEDIUM,
        disposition=Disposition.REVIEW,
    )

    signed_finding = SignedFinding(
        finding=finding,
        entry_hash="a" * 64,
        prev_hash="0" * 64,
        signature="b" * 128,
        ledger_id=1,
    )

    assert signed_finding.finding.asset_ref == "sample_001.png"
    assert signed_finding.entry_hash == "a" * 64
    assert signed_finding.prev_hash == "0" * 64
    assert signed_finding.ledger_id == 1

    # Immutability check
    with pytest.raises(ValidationError):
        signed_finding.ledger_id = 2  # type: ignore[misc]


def test_finding_confidence_bounds() -> None:
    """Verify that confidence must strictly be within [0.0, 1.0]."""
    # Over 1.0 should fail
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            asset_type=AssetType.SAMPLE,
            asset_ref="img_001.png",
            detector="test_det",
            reason="test reason",
            confidence=1.05,
            severity=Severity.LOW,
            disposition=Disposition.ACCEPT,
        )
    assert "confidence" in str(exc_info.value)

    # Below 0.0 should fail
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            asset_type=AssetType.SAMPLE,
            asset_ref="img_001.png",
            detector="test_det",
            reason="test reason",
            confidence=-0.1,
            severity=Severity.LOW,
            disposition=Disposition.ACCEPT,
        )
    assert "confidence" in str(exc_info.value)


def test_finding_enum_validation() -> None:
    """Ensure invalid severity and disposition strings raise validation errors."""
    with pytest.raises(ValidationError):
        Finding(
            asset_type=AssetType.SAMPLE,
            asset_ref="img_001.png",
            detector="test_det",
            reason="test reason",
            confidence=0.5,
            severity="unknown_severity",  # type: ignore[arg-type]
            disposition=Disposition.ACCEPT,
        )

    with pytest.raises(ValidationError):
        Finding(
            asset_type=AssetType.SAMPLE,
            asset_ref="img_001.png",
            detector="test_det",
            reason="test reason",
            confidence=0.5,
            severity=Severity.MEDIUM,
            disposition="unknown_disposition",  # type: ignore[arg-type]
        )


def test_finding_immutability() -> None:
    """Ensure Finding instances are frozen/immutable to prevent accidental tamper."""
    finding = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="img_001.png",
        detector="test_det",
        reason="test reason",
        confidence=0.5,
        severity=Severity.INFO,
        disposition=Disposition.ACCEPT,
    )
    with pytest.raises(ValidationError):
        finding.confidence = 0.99  # type: ignore[misc]


def test_report_creation_and_serialization() -> None:
    """Verify Report aggregation with Findings, CoverageStatement, and ReproducibilityManifest."""
    finding = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="img_100.png",
        detector="cvguard.detector.adversarial_patch:v1",
        reason="Adversarial high-frequency perturbation detected in top-left bounding quadrant.",
        evidence=["fft_high_freq_spike:3.4db", "bbox:[0,0,128,128]"],
        confidence=0.91,
        severity=Severity.CRITICAL,
        disposition=Disposition.QUARANTINE,
    )

    coverage = CoverageStatement(
        total_assets_scanned=1000,
        passed_count=999,
        flagged_count=1,
        skipped_count=0,
        scope_description="Air-gapped verification run against ImageNet evaluation subset.",
    )

    reproducibility_manifest = {
        "engine_version": "0.2.0",
        "python_version": "3.12.9",
        "git_commit": "e53bcc29440cbdf9da14668c1718b60",
        "config_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
        "offline_manifest_timestamp": "2026-09-08T00:00:00Z",
    }

    report = Report(
        title="Offline CV Model Integrity Scan #481",
        findings=[finding],
        coverage=coverage,
        reproducibility=reproducibility_manifest,
    )

    assert report.coverage.total_assets_scanned == 1000
    assert len(report.findings) == 1
    assert report.reproducibility["engine_version"] == "0.2.0"

    # Test roundtrip JSON dump and load
    json_data = report.model_dump_json()
    reconstructed = Report.model_validate_json(json_data)

    assert reconstructed.report_id == report.report_id
    assert reconstructed.findings[0].asset_ref == finding.asset_ref
    assert reconstructed.findings[0].severity == Severity.CRITICAL
