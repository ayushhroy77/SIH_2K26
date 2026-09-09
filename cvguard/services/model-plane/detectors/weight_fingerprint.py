"""CVGuard White-Box Weight and Parameter Fingerprint Detector.

Analyzes neural network serialized weight matrices to compute layer-wise statistical
fingerprints (mean, standard deviation, L2 norm, sparsity, and extreme values).
Flags anomalies against expected parameter distribution envelopes for declared architectures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cvguard_schemas import AssetType, Disposition, Finding, Severity

# Expected weight baseline envelopes for common vision architecture families
ARCHITECTURE_ENVELOPES: dict[str, dict[str, Any]] = {
    "resnet50": {
        "expected_mean_range": (-0.05, 0.05),
        "expected_std_range": (0.001, 0.15),
        "max_permitted_layer_norm": 50.0,
        "max_permitted_sparsity": 0.85,
    },
    "vit_base": {
        "expected_mean_range": (-0.08, 0.08),
        "expected_std_range": (0.005, 0.20),
        "max_permitted_layer_norm": 80.0,
        "max_permitted_sparsity": 0.90,
    },
    "convnet": {
        "expected_mean_range": (-0.06, 0.06),
        "expected_std_range": (0.002, 0.18),
        "max_permitted_layer_norm": 60.0,
        "max_permitted_sparsity": 0.85,
    },
    "generic_cnn": {
        "expected_mean_range": (-0.10, 0.10),
        "expected_std_range": (0.001, 0.25),
        "max_permitted_layer_norm": 100.0,
        "max_permitted_sparsity": 0.92,
    },
}


@dataclass
class WeightAnomalyReport:
    """Detailed telemetry on observed weight anomalies."""

    has_critical_anomaly: bool
    has_statistical_outlier: bool
    anomalous_layers: list[dict[str, Any]]
    confidence: float
    summary: str


class WeightFingerprintDetector:
    """White-box detector evaluating layer-wise parameter distributions for tampering."""

    DETECTOR_ID = "cvguard.modelplane.weight_fingerprint:v1.0"

    def __init__(self, architecture_envelopes: dict[str, dict[str, Any]] | None = None) -> None:
        self.envelopes = architecture_envelopes or ARCHITECTURE_ENVELOPES

    def analyze_stats(
        self,
        weight_stats: dict[str, Any],
        asset_ref: str,
        access_level: str = "white-box",
    ) -> list[Finding]:
        """Evaluate weight statistics and return findings if anomalies exceed safety envelopes."""
        findings: list[Finding] = []
        declared_arch = weight_stats.get("declared_architecture", "generic_cnn").lower()
        envelope = self.envelopes.get(declared_arch, self.envelopes["generic_cnn"])

        nan_count = weight_stats.get("nan_count", 0)
        inf_count = weight_stats.get("inf_count", 0)
        layers = weight_stats.get("layers", [])
        total_params = weight_stats.get("total_parameters", 0)

        anomalous_layers: list[dict[str, Any]] = []

        # 1. Check for fatal arithmetic anomalies (NaN/Inf in weights)
        if nan_count > 0 or inf_count > 0:
            evidence = [
                f"nan_weights_detected: {nan_count}",
                f"inf_weights_detected: {inf_count}",
                f"total_parameters_inspected: {total_params}",
                f"declared_architecture: {declared_arch}",
            ]
            findings.append(
                Finding(
                    asset_type=AssetType.MODEL,
                    asset_ref=asset_ref,
                    detector=self.DETECTOR_ID,
                    reason=(
                        f"Corrupted or adversarial weights detected: model contains {nan_count} NaN "
                        f"and {inf_count} Inf weight values causing numerical breakdown or denial of service."
                    ),
                    evidence=evidence,
                    confidence=0.99,
                    severity=Severity.CRITICAL,
                    disposition=Disposition.QUARANTINE,
                    assumptions=[
                        f"Model access level: {access_level} (full serialized weights available for inspection).",
                        f"Declared architecture baseline: {declared_arch}.",
                    ],
                    limitations=[
                        "Does not determine root cause of corruption (disk defect vs deliberate gradient explosion).",
                    ],
                )
            )

        # 2. Check per-layer statistical deviations
        mean_min, mean_max = envelope["expected_mean_range"]
        std_min, std_max = envelope["expected_std_range"]
        max_norm = envelope["max_permitted_layer_norm"]
        max_sparsity = envelope["max_permitted_sparsity"]

        for layer in layers:
            name = layer.get("name", "unknown")
            l_mean = layer.get("mean", 0.0)
            l_std = layer.get("std", 0.0)
            l_norm = layer.get("l2_norm", 0.0)
            l_sparsity = layer.get("sparsity", 0.0)

            issues = []
            if l_mean < mean_min or l_mean > mean_max:
                issues.append(f"mean {l_mean:.4f} outside [{mean_min}, {mean_max}]")
            if l_std < std_min or l_std > std_max:
                issues.append(f"std {l_std:.4f} outside [{std_min}, {std_max}]")
            if l_norm > max_norm:
                issues.append(f"L2 norm {l_norm:.2f} exceeds ceiling {max_norm}")
            if l_sparsity > max_sparsity:
                issues.append(f"sparsity {l_sparsity:.2%} exceeds ceiling {max_sparsity:.0%}")

            if issues:
                anomalous_layers.append({
                    "layer": name,
                    "issues": issues,
                    "l2_norm": l_norm,
                    "mean": l_mean,
                    "std": l_std,
                })

        if anomalous_layers:
            evidence = [
                f"anomalous_layer_count: {len(anomalous_layers)} / {len(layers)}",
                f"declared_architecture: {declared_arch}",
                f"sample_outlier_layer: {anomalous_layers[0]['layer']} ({'; '.join(anomalous_layers[0]['issues'])})",
                f"max_observed_layer_norm: {max(l['l2_norm'] for l in anomalous_layers):.2f} (threshold {max_norm})",
            ]
            # Honest, conservative confidence score based on fraction of anomalous layers
            fraction_anomalous = len(anomalous_layers) / max(len(layers), 1)
            confidence = round(min(0.60 + fraction_anomalous * 0.35, 0.92), 4)

            findings.append(
                Finding(
                    asset_type=AssetType.MODEL,
                    asset_ref=asset_ref,
                    detector=self.DETECTOR_ID,
                    reason=(
                        f"Statistical weight distribution anomaly detected in {len(anomalous_layers)} layers "
                        f"deviating significantly from the expected {declared_arch} envelope."
                    ),
                    evidence=evidence,
                    confidence=confidence,
                    severity=Severity.HIGH if fraction_anomalous > 0.3 else Severity.MEDIUM,
                    disposition=Disposition.REVIEW,
                    assumptions=[
                        f"Model access level: {access_level} (full serialized weights available for inspection).",
                        f"Declared architecture envelope: {declared_arch}.",
                        "Weights in standard pre-trained or fine-tuned vision models follow regularized distributions.",
                    ],
                    limitations=[
                        "Subtle, manifold-constrained backdoors that preserve global weight distribution statistics "
                        "will not be flagged by layer-wide moments.",
                        "Custom or non-standard regularization schemes during training may trigger false positives.",
                    ],
                )
            )

        return findings
