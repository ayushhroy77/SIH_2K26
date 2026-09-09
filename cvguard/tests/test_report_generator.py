"""Unit tests for CVGuard Report Generator, HTML/PDF Renderers, and Triage Queue Sorting (Phase 7).

Verifies:
1. Report generation against hand-constructed test Findings spanning multiple planes and severities.
2. Report validates against canonical Pydantic v2 Report schema.
3. Coverage statement accurately lists supported and non-supported classes merged from coverage_manifest.json.
4. Calling report generation twice over the same range produces byte-identical reproducibility data.
5. HTML and PDF rendering succeed without exception and reflect the expected finding count.
6. Triage queue default sort logic orders findings by severity descending, then confidence descending.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure governance and schemas are on sys.path
_BASE_DIR = Path(__file__).resolve().parent.parent
_GOVERNANCE_DIR = _BASE_DIR / "services" / "governance"
_SCHEMAS_DIR = _BASE_DIR / "libs" / "schemas"

for p in [_GOVERNANCE_DIR, _SCHEMAS_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cvguard_schemas import (
    AssetType,
    Disposition,
    Finding,
    Report,
    Severity,
    SignedFinding,
)
from ledger import AuditLedger
from reports import (
    generate_report,
    load_coverage_manifest,
    render_report_html,
    render_report_pdf,
    sort_findings_triage,
)
from signer import LocalFileSigner


@pytest.fixture
def test_ledger() -> AuditLedger:
    """Create an isolated test AuditLedger with signed findings spanning all planes."""
    signer = LocalFileSigner()
    ledger = AuditLedger(signer=signer)

    # 1. Data Plane finding (High severity)
    f1 = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="sha256:d41d8cd98f00b204e9800998ecf8427e",
        detector="cvguard.dataplane.ood_mahalanobis:v1.0",
        reason="Out-of-distribution sample with Mahalanobis distance exceeding threshold.",
        evidence=["mahalanobis_distance:14.8", "threshold:12.0"],
        confidence=0.94,
        severity=Severity.HIGH,
        disposition=Disposition.REVIEW,
        assumptions=["Reference class distribution is unimodal Gaussian."],
        limitations=["May exhibit elevated false positive rates on multi-modal classes."],
    )

    # 2. Model Plane finding (Critical severity)
    f2 = Finding(
        asset_type=AssetType.MODEL,
        asset_ref="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        detector="cvguard.modelplane.safetensors_structural:v1.0",
        reason="Arbitrary code execution risk: model checkpoint contains unauthorized pickle stream.",
        evidence=["header_magic:PK0304", "embedded_python_opcode:GLOBAL"],
        confidence=0.99,
        severity=Severity.CRITICAL,
        disposition=Disposition.QUARANTINE,
        assumptions=["SafeTensors format strictly enforced."],
        limitations=["Only format envelope and byte signatures validated."],
    )

    # 3. Inference Plane finding (Medium severity)
    f3 = Finding(
        asset_type=AssetType.INFERENCE_RECORD,
        asset_ref="inf-record-alpha-492",
        detector="cvguard.inferenceplane.adversarial_perturbation:v1.0",
        reason="High-frequency DCT spectral spike indicates possible adversarial perturbation.",
        evidence=["spectral_energy_ratio:0.042", "baseline_max:0.015"],
        confidence=0.85,
        severity=Severity.MEDIUM,
        disposition=Disposition.REVIEW,
        assumptions=["Input normalized to standard ImageNet space."],
        limitations=["Spatial patch attacks without spectral anomalies may evade detection."],
    )

    # 4. Drift Plane finding (Low severity)
    f4 = Finding(
        asset_type=AssetType.BATCH,
        asset_ref="batch-daily-2026-09-09",
        detector="cvguard.driftplane.distribution_verifier:v1.0",
        reason="Mild distribution shift detected correlated with sensor_id change.",
        evidence=["mmd_divergence:0.038", "p_value:0.041", "correlated_metadata:sensor_id"],
        confidence=0.72,
        severity=Severity.LOW,
        disposition=Disposition.REVIEW,
        assumptions=["Reference profile representative of operational baseline."],
        limitations=["Cannot distinguish sensor recalibration from physical lighting changes."],
    )

    # 5. Clean Data Plane finding (Info severity)
    f5 = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="sha256:9f83c65a4c9b3a521e904b23b5d1209b",
        detector="cvguard.detector.phash_near_duplicate:v1.0",
        reason="Sample verified distinct from all registered historical assets.",
        evidence=["min_hamming_distance:28"],
        confidence=0.50,
        severity=Severity.INFO,
        disposition=Disposition.ACCEPT,
        assumptions=["pHash 64-bit DCT evaluated."],
        limitations=["Does not evaluate high-level semantic variations."],
    )

    ledger.append_entry(f1)
    ledger.append_entry(f2)
    ledger.append_entry(f3)
    ledger.append_entry(f4)
    ledger.append_entry(f5)

    return ledger


def test_report_generation_and_schema_validation(test_ledger: AuditLedger) -> None:
    """Assert the Report generates successfully and validates against canonical Pydantic schema."""
    report = generate_report(ledger=test_ledger, since_id=1)

    assert isinstance(report, Report)
    assert report.report_id.startswith("rep_")
    assert len(report.findings) == 5

    # Verify Pydantic v2 serialization and reconstruction
    report_dict = report.model_dump(mode="json")
    reconstructed = Report.model_validate(report_dict)

    assert reconstructed.report_id == report.report_id
    assert len(reconstructed.findings) == 5
    assert reconstructed.coverage.total_assets_scanned == 5
    assert reconstructed.coverage.flagged_count == 4
    assert reconstructed.coverage.passed_count == 1


def test_report_coverage_merges_supported_and_uncovered_classes(test_ledger: AuditLedger) -> None:
    """Assert the coverage section lists exactly the expected supported and non-supported classes from the manifest."""
    manifest_data, manifest_hash = load_coverage_manifest()
    report = generate_report(ledger=test_ledger, since_id=1)

    # Verify coverage contains supported classes from all planes
    supported = report.coverage.supported_attack_classes
    uncovered = report.coverage.uncovered_attack_classes

    assert len(supported) > 0
    assert len(uncovered) > 0

    supported_names = {item["name"] for item in supported}
    uncovered_names = {item["name"] for item in uncovered}

    # Verify presence of representative covered classes
    assert "Exact and Near-Duplicate Image Injection" in supported_names
    assert "Covariate Shift and Out-of-Domain Contamination" in supported_names
    assert "Safetensors Header Manipulation" in supported_names
    assert "Atomic Inference Replay and Checkpoint Mismatch" in supported_names

    # Verify presence of representative uncovered classes
    assert "Large Spatial Transformations and Non-Affine Crops" in uncovered_names
    assert "Adversarial Embedding Invariance (Manifold-Constrained Perturbations)" in uncovered_names

    # Verify detector versions mapping is populated
    assert "cvguard.dataplane.ood_mahalanobis:v1.0" in report.coverage.detector_versions
    assert report.coverage.detector_versions["cvguard.dataplane.ood_mahalanobis:v1.0"] == "v1.0"


def test_report_reproducibility_is_deterministic(test_ledger: AuditLedger) -> None:
    """Assert calling report generation twice over the same ledger range produces byte-identical reproducibility data."""
    report_a = generate_report(ledger=test_ledger, since_id=1)
    report_b = generate_report(ledger=test_ledger, since_id=1)

    # The reproducibility dictionary must be identical
    assert report_a.reproducibility == report_b.reproducibility

    repro_json_a = json.dumps(report_a.reproducibility, sort_keys=True)
    repro_json_b = json.dumps(report_b.reproducibility, sort_keys=True)

    assert repro_json_a == repro_json_b
    assert "coverage_manifest_hash" in report_a.reproducibility
    assert "generation_timestamp" in report_a.reproducibility
    assert "detector_versions" in report_a.reproducibility

    # Verify detectors present in findings are in detector_versions
    assert "cvguard.dataplane.ood_mahalanobis:v1.0" in report_a.reproducibility["detector_versions"]
    assert "cvguard.modelplane.safetensors_structural:v1.0" in report_a.reproducibility["detector_versions"]


def test_report_rendering_html_and_pdf(test_ledger: AuditLedger) -> None:
    """Assert HTML and PDF rendering succeed without exception and reflect expected finding counts."""
    report = generate_report(ledger=test_ledger, since_id=1)

    # 1. HTML Rendering
    html_output = render_report_html(report)
    assert isinstance(html_output, str)
    assert len(html_output) > 500
    assert "CVGuard Integrity Assurance Report" in html_output
    assert "Executive Risk Summary" in html_output
    assert "Coverage &amp; Limitations Statement" in html_output
    assert f"Findings Count:</strong> 5" in html_output

    # 2. PDF Rendering
    pdf_output = render_report_pdf(report)
    assert isinstance(pdf_output, bytes)
    assert len(pdf_output) > 100
    assert pdf_output.startswith(b"%PDF-")


def test_triage_queue_default_sort_logic() -> None:
    """Assert pure sorting function orders findings by severity descending, then confidence descending."""
    f_low_highconf = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f1",
        detector="d1",
        reason="low sev, high conf",
        confidence=0.95,
        severity=Severity.LOW,
        disposition=Disposition.REVIEW,
    )
    f_crit_lowconf = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f2",
        detector="d2",
        reason="crit sev, lower conf",
        confidence=0.80,
        severity=Severity.CRITICAL,
        disposition=Disposition.QUARANTINE,
    )
    f_crit_highconf = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f3",
        detector="d3",
        reason="crit sev, highest conf",
        confidence=0.99,
        severity=Severity.CRITICAL,
        disposition=Disposition.QUARANTINE,
    )
    f_high_midconf = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f4",
        detector="d4",
        reason="high sev, mid conf",
        confidence=0.90,
        severity=Severity.HIGH,
        disposition=Disposition.REVIEW,
    )
    f_high_highconf = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f5",
        detector="d5",
        reason="high sev, high conf",
        confidence=0.95,
        severity=Severity.HIGH,
        disposition=Disposition.REVIEW,
    )
    f_med = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f6",
        detector="d6",
        reason="med sev",
        confidence=0.85,
        severity=Severity.MEDIUM,
        disposition=Disposition.REVIEW,
    )
    f_info = Finding(
        asset_type=AssetType.SAMPLE,
        asset_ref="f7",
        detector="d7",
        reason="info sev",
        confidence=0.50,
        severity=Severity.INFO,
        disposition=Disposition.ACCEPT,
    )

    scrambled = [
        f_low_highconf,
        f_crit_lowconf,
        f_info,
        f_high_midconf,
        f_crit_highconf,
        f_med,
        f_high_highconf,
    ]

    sorted_results = sort_findings_triage(scrambled)

    # Expected order:
    # 1. Critical (0.99)
    # 2. Critical (0.80)
    # 3. High (0.95)
    # 4. High (0.90)
    # 5. Medium (0.85)
    # 6. Low (0.95)
    # 7. Info (0.50)
    assert sorted_results[0].reason == "crit sev, highest conf"
    assert sorted_results[1].reason == "crit sev, lower conf"
    assert sorted_results[2].reason == "high sev, high conf"
    assert sorted_results[3].reason == "high sev, mid conf"
    assert sorted_results[4].reason == "med sev"
    assert sorted_results[5].reason == "low sev, high conf"
    assert sorted_results[6].reason == "info sev"
