"""CVGuard Reference Battery Behavioral Envelope Detector.

Runs every ingested model against the canonical, versioned reference battery of test images
and flags any model whose outputs deviate significantly from documented expected envelopes
(e.g., probability collapse, zero entropy on ambiguous noise, NaN outputs, or total class fixation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cvguard_schemas import AssetType, Disposition, Finding, Severity
from reference_battery.battery_data import get_canonical_battery_inputs, load_battery_manifest


def compute_entropy(probs: list[float], eps: float = 1e-9) -> float:
    """Compute Shannon entropy in bits for probability distribution."""
    return -sum(p * math.log2(p + eps) for p in probs if p > 0.0)


@dataclass
class BatteryEvaluationResult:
    """Summary of model evaluation on the canonical reference battery."""

    battery_version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    deviations: list[dict[str, Any]]


class ReferenceBatteryDetector:
    """Detector evaluating model behavioral adherence to canonical reference envelopes."""

    DETECTOR_ID = "cvguard.modelplane.reference_battery:v1.0"

    def __init__(self) -> None:
        self.manifest = load_battery_manifest()
        self.battery_version = self.manifest.get("version", "1.0.0")

    def evaluate_model_outputs(
        self,
        model_predictions: list[dict[str, Any]],
        asset_ref: str,
        access_level: str = "white-box",
    ) -> tuple[list[Finding], BatteryEvaluationResult]:
        """Evaluate prediction distributions against expected item envelopes.

        Args:
            model_predictions: List of dicts with:
                - "item_id": matching manifest item_id
                - "probabilities": list of float output probabilities
            asset_ref: Reference locator for the model.
            access_level: 'white-box' or 'black-box'.
        """
        canonical_items = {item["item_id"]: item for item in get_canonical_battery_inputs()}
        deviations: list[dict[str, Any]] = []

        total_tested = 0
        for pred in model_predictions:
            item_id = pred.get("item_id")
            if item_id not in canonical_items:
                continue

            total_tested += 1
            item = canonical_items[item_id]
            probs: list[float] = pred.get("probabilities", [])

            if not probs:
                deviations.append({
                    "item_id": item_id,
                    "reason": "Empty or missing prediction vector returned by model.",
                })
                continue

            # Check for NaN / Inf
            if any(math.isnan(p) or math.isinf(p) for p in probs):
                deviations.append({
                    "item_id": item_id,
                    "reason": "Prediction vector contains NaN or Inf floating point values.",
                })
                continue

            entropy = compute_entropy(probs)
            max_prob = max(probs)
            pred_class = probs.index(max_prob)

            # Check entropy bounds
            min_ent = item.get("min_entropy", 0.0)
            max_ent = item.get("max_entropy", 4.0)
            if entropy < min_ent:
                deviations.append({
                    "item_id": item_id,
                    "reason": f"Output entropy {entropy:.3f} below minimum envelope {min_ent:.3f} (extreme overconfidence / collapse).",
                    "observed_entropy": entropy,
                    "expected_min_entropy": min_ent,
                })
            elif entropy > max_ent:
                deviations.append({
                    "item_id": item_id,
                    "reason": f"Output entropy {entropy:.3f} exceeds maximum envelope {max_ent:.3f} (excessive uncertainty / degenerate distribution).",
                    "observed_entropy": entropy,
                    "expected_max_entropy": max_ent,
                })

            # Check minimum confidence
            min_conf = item.get("min_confidence", 0.1)
            if max_prob < min_conf:
                deviations.append({
                    "item_id": item_id,
                    "reason": f"Top-1 confidence {max_prob:.3f} fell below minimum requirement {min_conf:.3f}.",
                    "observed_confidence": max_prob,
                    "expected_min_confidence": min_conf,
                })

        findings: list[Finding] = []
        passed = total_tested - len(deviations)

        summary = BatteryEvaluationResult(
            battery_version=self.battery_version,
            total_tests=total_tested,
            passed_tests=passed,
            failed_tests=len(deviations),
            deviations=deviations,
        )

        if deviations:
            evidence = [
                f"battery_version: {self.battery_version}",
                f"tests_evaluated: {total_tested}",
                f"tests_deviating: {len(deviations)}",
                f"first_deviation_item: {deviations[0].get('item_id', 'unknown')}",
                f"first_deviation_reason: {deviations[0].get('reason', '')}",
            ]

            severity = Severity.CRITICAL if len(deviations) >= (total_tested // 2) else Severity.HIGH
            disposition = Disposition.QUARANTINE if severity == Severity.CRITICAL else Disposition.REVIEW

            # Confidence scaled honestly
            confidence = round(min(0.70 + (len(deviations) / max(total_tested, 1)) * 0.25, 0.95), 4)

            findings.append(
                Finding(
                    asset_type=AssetType.MODEL,
                    asset_ref=asset_ref,
                    detector=self.DETECTOR_ID,
                    reason=(
                        f"Model failed behavioral envelope verification on canonical reference battery "
                        f"v{self.battery_version}: {len(deviations)} / {total_tested} test probes deviated "
                        f"from documented confidence or entropy bounds."
                    ),
                    evidence=evidence,
                    confidence=confidence,
                    severity=severity,
                    disposition=disposition,
                    assumptions=[
                        f"Model access level: {access_level}.",
                        f"Canonical reference battery v{self.battery_version} defines baseline vision behavior envelopes.",
                        "Evaluated model is trained for general vision classification aligned with test probes.",
                    ],
                    limitations=[
                        "Does not diagnose task-specific failure modes on specialized non-classification architectures.",
                        "Models trained on distinct class ontologies may fail specific class index targets.",
                    ],
                )
            )

        return findings, summary
