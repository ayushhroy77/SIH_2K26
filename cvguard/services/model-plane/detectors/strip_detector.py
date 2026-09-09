"""CVGuard Black-Box STRIP (STRong Intentional Perturbation) Backdoor Detector.

Evaluates neural network prediction invariance under intentional input blending perturbations.
For clean models, blending an input with another image increases prediction uncertainty (HIGH entropy).
When a backdoor trigger dominates the input, the model persistently predicts the attacker's
target class regardless of the overlaid background image (abnormally LOW entropy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cvguard_schemas import AssetType, Disposition, Finding, Severity


def compute_shannon_entropy(probs: list[float], eps: float = 1e-9) -> float:
    """Compute Shannon entropy in bits for a discrete probability distribution."""
    return -sum(p * math.log2(p + eps) for p in probs if p > 0.0)


@dataclass
class STRIPSampleResult:
    """Telemetry for a single probe sample evaluated under STRIP perturbation."""

    sample_id: str
    average_entropy: float
    entropy_variance: float
    dominant_class: int
    is_trigger_suspect: bool


class STRIPDetector:
    """Black-box detector measuring output entropy collapse under input superposition."""

    DETECTOR_ID = "cvguard.modelplane.strip_blackbox:v1.0"

    def __init__(
        self,
        entropy_threshold: float = 0.45,
        num_perturbations: int = 16,
        blend_alpha: float = 0.50,
    ) -> None:
        self.entropy_threshold = entropy_threshold
        self.num_perturbations = num_perturbations
        self.blend_alpha = blend_alpha

    def evaluate_entropy_profiles(
        self,
        sample_entropy_profiles: list[dict[str, Any]],
        asset_ref: str,
        access_level: str = "black-box",
    ) -> list[Finding]:
        """Assess entropy scores across probe samples and emit findings on entropy collapse.

        Args:
            sample_entropy_profiles: List of dicts containing:
                - "sample_id": identifier
                - "entropies": list of Shannon entropies across N blended perturbations
                - "dominant_class": predicted class index
            asset_ref: Locator for the assessed model.
            access_level: Access classification ('black-box' or 'white-box').
        """
        findings: list[Finding] = []
        flagged_samples: list[STRIPSampleResult] = []

        for sample in sample_entropy_profiles:
            entropies: list[float] = sample.get("entropies", [])
            sample_id = sample.get("sample_id", "sample_unknown")
            dominant_class = sample.get("dominant_class", 0)

            if not entropies:
                continue

            avg_entropy = sum(entropies) / len(entropies)
            var_entropy = sum((e - avg_entropy) ** 2 for e in entropies) / len(entropies)

            if avg_entropy < self.entropy_threshold:
                flagged_samples.append(
                    STRIPSampleResult(
                        sample_id=sample_id,
                        average_entropy=round(avg_entropy, 4),
                        entropy_variance=round(var_entropy, 5),
                        dominant_class=dominant_class,
                        is_trigger_suspect=True,
                    )
                )

        if flagged_samples:
            min_entropy = min(s.average_entropy for s in flagged_samples)
            target_classes = sorted({s.dominant_class for s in flagged_samples})

            evidence = [
                f"flagged_trigger_suspect_count: {len(flagged_samples)} / {len(sample_entropy_profiles)}",
                f"minimum_observed_entropy: {min_entropy:.4f} bits (threshold {self.entropy_threshold:.2f} bits)",
                f"target_compromise_classes: {target_classes}",
                f"sample_culprit_id: {flagged_samples[0].sample_id}",
                f"perturbation_count_per_sample: {self.num_perturbations}",
            ]

            # Confidence scaled by how far entropy dropped below the threshold
            entropy_margin = self.entropy_threshold - min_entropy
            confidence = round(min(0.68 + entropy_margin * 0.45, 0.94), 4)

            findings.append(
                Finding(
                    asset_type=AssetType.MODEL,
                    asset_ref=asset_ref,
                    detector=self.DETECTOR_ID,
                    reason=(
                        f"STRIP backdoor trigger signature detected: {len(flagged_samples)} probe inputs "
                        f"exhibited persistent entropy collapse (minimum {min_entropy:.4f} bits < "
                        f"threshold {self.entropy_threshold:.2f} bits) despite heavy image superposition, "
                        f"indicating a dominant backdoor shortcut to class {target_classes}."
                    ),
                    evidence=evidence,
                    confidence=confidence,
                    severity=Severity.HIGH,
                    disposition=Disposition.QUARANTINE if min_entropy < 0.20 else Disposition.REVIEW,
                    assumptions=[
                        f"Model access level: {access_level} (evaluated via query API).",
                        "Input blending perturbations introduce sufficient semantic ambiguity in clean models.",
                        "Trigger-bearing inputs persistently activate the target label across background mixes.",
                    ],
                    limitations=[
                        "Does not detect dynamic, sample-specific backdoors where the trigger only activates "
                        "on specific unmixed input geometries.",
                        "Models with severe overconfidence or sharp logits across all inputs may exhibit "
                        "elevated false-positive rates.",
                    ],
                )
            )

        return findings
