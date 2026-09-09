"""Unit Test Suite for CVGuard Model Plane Detectors.

Tests:
1. WeightFingerprintDetector on clean parameters, NaN/Inf corrupted parameters, and norm outlier weights.
2. ActivationClusteringDetector on clean unimodal features vs backdoored bimodal activation sub-clusters.
3. STRIPDetector on clean high-entropy perturbation responses vs backdoored low-entropy responses.
4. ReferenceBatteryDetector on compliant models vs models with collapsed output distributions.
"""

from __future__ import annotations

import math
import pytest

from cvguard_schemas import AssetType, Disposition, Severity
from detectors.activation_clustering import ActivationClusteringDetector
from detectors.reference_battery import ReferenceBatteryDetector
from detectors.strip_detector import STRIPDetector
from detectors.weight_fingerprint import WeightFingerprintDetector


# =============================================================================
# 1. Weight Fingerprint Detector Unit Tests
# =============================================================================
def test_weight_fingerprint_clean_weights_produce_no_findings() -> None:
    """Verify that a model whose weights lie strictly within expected architecture bounds passes cleanly."""
    detector = WeightFingerprintDetector()
    clean_stats = {
        "declared_architecture": "resnet50",
        "total_parameters": 25557032,
        "nan_count": 0,
        "inf_count": 0,
        "overall_sparsity": 0.05,
        "layers": [
            {"name": "conv1.weight", "mean": 0.001, "std": 0.045, "l2_norm": 2.14, "sparsity": 0.0},
            {"name": "layer1.0.conv1.weight", "mean": -0.002, "std": 0.038, "l2_norm": 1.85, "sparsity": 0.0},
            {"name": "fc.weight", "mean": 0.0005, "std": 0.025, "l2_norm": 3.42, "sparsity": 0.02},
        ],
    }

    findings = detector.analyze_stats(clean_stats, asset_ref="model:clean_resnet50:abc123")
    assert len(findings) == 0, f"Expected 0 findings for clean weights, got {len(findings)}"


def test_weight_fingerprint_nan_weights_produce_critical_quarantine_finding() -> None:
    """Verify that NaN values in weight tensors trigger immediate CRITICAL quarantine."""
    detector = WeightFingerprintDetector()
    corrupted_stats = {
        "declared_architecture": "resnet50",
        "total_parameters": 1000,
        "nan_count": 14,
        "inf_count": 0,
        "overall_sparsity": 0.0,
        "layers": [
            {"name": "conv1.weight", "mean": float("nan"), "std": 0.05, "l2_norm": 1.0, "sparsity": 0.0},
        ],
    }

    findings = detector.analyze_stats(corrupted_stats, asset_ref="model:nan_corrupted:bad456")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == Severity.CRITICAL
    assert finding.disposition == Disposition.QUARANTINE
    assert finding.confidence >= 0.95
    assert "nan_weights_detected: 14" in finding.evidence[0]


def test_weight_fingerprint_norm_spike_flags_statistical_outlier() -> None:
    """Verify that a layer with an abnormal L2 norm spike triggers a HIGH review finding."""
    detector = WeightFingerprintDetector()
    spiked_stats = {
        "declared_architecture": "resnet50",
        "total_parameters": 50000,
        "nan_count": 0,
        "inf_count": 0,
        "overall_sparsity": 0.0,
        "layers": [
            # Max permitted norm for resnet50 is 50.0
            {"name": "suspicious_trigger_filter.weight", "mean": 0.85, "std": 0.45, "l2_norm": 184.2, "sparsity": 0.0},
        ],
    }

    findings = detector.analyze_stats(spiked_stats, asset_ref="model:spiked_norm:xyz789")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == Severity.HIGH
    assert finding.disposition == Disposition.REVIEW
    assert "L2 norm 184.20 exceeds ceiling 50.0" in finding.evidence[2]


# =============================================================================
# 2. Activation Clustering Detector Unit Tests
# =============================================================================
def test_activation_clustering_clean_unimodal_features_pass() -> None:
    """Verify that normally distributed activations forming a single unimodal cloud produce 0 findings."""
    detector = ActivationClusteringDetector(min_samples_per_class=6, separation_threshold=2.2)

    # 10 samples in class 0 tightly centered around [1.0, 1.0, ...]
    clean_samples = [[1.0 + (i * 0.02) for _ in range(8)] for i in range(10)]
    activation_data = {
        "class_activations": {
            "0": clean_samples,
        }
    }

    findings = detector.analyze_activations(activation_data, asset_ref="model:clean_act:123")
    assert len(findings) == 0, f"Expected 0 findings for unimodal features, got {len(findings)}"


def test_activation_clustering_backdoored_subcluster_flags_anomaly() -> None:
    """Verify that a class with a separate minority sub-cluster (poisoned samples) is detected."""
    detector = ActivationClusteringDetector(min_samples_per_class=6, separation_threshold=2.0)

    # 8 clean samples around [0.0, ...]
    clean_samples = [[0.05 * i for _ in range(8)] for i in range(8)]
    # 2 backdoored samples strongly separated around [15.0, ...] (20% of class)
    backdoor_samples = [[15.0 + 0.1 * i for _ in range(8)] for i in range(2)]

    mixed_class_samples = clean_samples + backdoor_samples
    activation_data = {
        "class_activations": {
            "target_class_4": mixed_class_samples,
        }
    }

    findings = detector.analyze_activations(activation_data, asset_ref="model:backdoored_act:789")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == Severity.HIGH
    assert finding.disposition == Disposition.REVIEW
    assert "primary_flagged_class: target_class_4" in finding.evidence[1]
    assert "minority_sub_population_ratio: 20.0%" in finding.evidence[5]


# =============================================================================
# 3. STRIP Black-Box Detector Unit Tests
# =============================================================================
def test_strip_clean_inputs_high_entropy_passes() -> None:
    """Verify that clean probe inputs whose predictions exhibit high entropy across perturbations pass."""
    detector = STRIPDetector(entropy_threshold=0.45)
    clean_profiles = [
        {"sample_id": "probe_01", "entropies": [1.85, 2.10, 1.95, 2.05, 1.78], "dominant_class": 0},
        {"sample_id": "probe_02", "entropies": [1.90, 1.82, 2.15, 2.00, 1.92], "dominant_class": 1},
    ]

    findings = detector.evaluate_entropy_profiles(clean_profiles, asset_ref="model:clean_strip:111")
    assert len(findings) == 0, f"Expected 0 findings for high entropy inputs, got {len(findings)}"


def test_strip_backdoored_low_entropy_flags_quarantine() -> None:
    """Verify that triggered inputs showing entropy collapse (e.g. 0.08 bits) are flagged for quarantine."""
    detector = STRIPDetector(entropy_threshold=0.45)
    poisoned_profiles = [
        {"sample_id": "clean_probe", "entropies": [1.90, 2.00, 1.85], "dominant_class": 0},
        # Trigger-bearing sample: prediction never wavers, entropy stays near 0.08 bits
        {"sample_id": "triggered_probe", "entropies": [0.08, 0.09, 0.07, 0.10, 0.08], "dominant_class": 7},
    ]

    findings = detector.evaluate_entropy_profiles(poisoned_profiles, asset_ref="model:poisoned_strip:222")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == Severity.HIGH
    assert finding.disposition == Disposition.QUARANTINE  # entropy < 0.20 triggers QUARANTINE
    assert "target_compromise_classes: [7]" in finding.evidence[2]
    assert finding.confidence >= 0.75


# =============================================================================
# 4. Reference Battery Detector Unit Tests
# =============================================================================
def test_reference_battery_compliant_predictions_pass() -> None:
    """Verify that model outputs complying with all canonical test envelopes produce no findings."""
    detector = ReferenceBatteryDetector()
    compliant_preds = [
        {"item_id": "canon_001_natural_contrast", "probabilities": [0.70, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]},
        {"item_id": "canon_002_geometric_primitives", "probabilities": [0.05, 0.65, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]},
        {"item_id": "canon_003_smooth_gradient", "probabilities": [0.05, 0.05, 0.60, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.05]},
        {"item_id": "canon_004_textured_surface", "probabilities": [0.05, 0.05, 0.05, 0.62, 0.05, 0.04, 0.04, 0.04, 0.04, 0.07]},
        {"item_id": "canon_005_ambiguous_noise", "probabilities": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]},
    ]

    findings, summary = detector.evaluate_model_outputs(compliant_preds, asset_ref="model:compliant_battery:333")
    assert len(findings) == 0
    assert summary.failed_tests == 0
    assert summary.passed_tests == 5


def test_reference_battery_degenerate_distribution_flags_deviation() -> None:
    """Verify that a model exhibiting class collapse (predicting class 0 on everything including noise) fails."""
    detector = ReferenceBatteryDetector()
    # Degenerate: always class 0 with 99.9% probability, entropy near 0 on ambiguous noise
    collapsed_preds = [
        {"item_id": "canon_001_natural_contrast", "probabilities": [0.999] + [0.0001] * 9},
        {"item_id": "canon_002_geometric_primitives", "probabilities": [0.999] + [0.0001] * 9},
        {"item_id": "canon_003_smooth_gradient", "probabilities": [0.999] + [0.0001] * 9},
        {"item_id": "canon_004_textured_surface", "probabilities": [0.999] + [0.0001] * 9},
        {"item_id": "canon_005_ambiguous_noise", "probabilities": [0.999] + [0.0001] * 9},  # Fails min_entropy requirement
    ]

    findings, summary = detector.evaluate_model_outputs(collapsed_preds, asset_ref="model:collapsed_battery:444")
    assert len(findings) == 1
    assert summary.failed_tests >= 1
    finding = findings[0]
    assert finding.severity in (Severity.HIGH, Severity.CRITICAL)
    assert "canon_005_ambiguous_noise" in finding.evidence[3]
