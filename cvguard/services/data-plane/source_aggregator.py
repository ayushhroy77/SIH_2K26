"""Unified Source-Level Aggregator for Cross-Detector Anomaly Concentration.

Aggregates per-sample anomaly signals from all active detectors (near-duplicate,
out-of-distribution, label-flip, and spectral trigger) to detect coordinated data poisoning
or suspicious contributor behavior via weighted multi-signal z-scoring.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvguard_schemas import AssetType, Disposition, Finding, Severity

logger = logging.getLogger("cvguard.dataplane.source_aggregator")

DEFAULT_SOURCE_SCORE_THRESHOLD = float(os.getenv("SOURCE_SCORE_THRESHOLD", "1.50"))
DEFAULT_SOURCE_Z_THRESHOLD = float(os.getenv("SOURCE_Z_THRESHOLD", "1.50"))


@dataclass(frozen=True)
class AnomalySignal:
    """Individual detector signal emitted for a specific sample."""

    image_id: int
    contributor_id: str
    minio_key: str
    detector_name: str
    score: float  # [0.0, 1.0] confidence or normalized metric
    is_anomaly: bool
    weight: float = 1.0


class SourceAggregator:
    """Unified cross-detector source concentration engine."""

    def __init__(
        self,
        score_threshold: float = DEFAULT_SOURCE_SCORE_THRESHOLD,
        z_threshold: float = DEFAULT_SOURCE_Z_THRESHOLD,
    ) -> None:
        self.score_threshold = score_threshold
        self.z_threshold = z_threshold

    def aggregate(
        self,
        signals: list[AnomalySignal],
        all_contributors: list[str] | None = None,
    ) -> list[Finding]:
        """Aggregate all detector signals across contributors and emit AssetType.SOURCE findings."""
        if not signals:
            return []

        # Group signals by contributor
        contrib_signals: dict[str, list[AnomalySignal]] = defaultdict(list)
        for sig in signals:
            if sig.contributor_id:
                contrib_signals[sig.contributor_id].append(sig)

        # Include any known non-flagged contributors in the denominator for accurate baseline
        if all_contributors:
            for c in all_contributors:
                if c not in contrib_signals:
                    contrib_signals[c] = []

        # Compute weighted anomaly score per contributor
        # S_c = sum(weight * score for anomalous signals)
        contrib_scores: dict[str, float] = {}
        contrib_anom_counts: dict[str, int] = {}
        contrib_detectors: dict[str, set[str]] = defaultdict(set)
        contrib_evidence_keys: dict[str, set[str]] = defaultdict(set)

        for c, sig_list in contrib_signals.items():
            score_sum = 0.0
            anom_count = 0
            for s in sig_list:
                if s.is_anomaly:
                    score_sum += s.weight * s.score
                    anom_count += 1
                    contrib_detectors[c].add(s.detector_name)
                    if s.minio_key:
                        contrib_evidence_keys[c].add(s.minio_key)
            contrib_scores[c] = score_sum
            contrib_anom_counts[c] = anom_count

        # Compute population mean and standard deviation across all contributors
        scores_arr = np.array(list(contrib_scores.values()), dtype=np.float64)
        mean_score = float(np.mean(scores_arr)) if len(scores_arr) > 0 else 0.0
        std_score = float(np.std(scores_arr)) if len(scores_arr) > 0 else 0.0
        eps = 1e-6

        source_findings: list[Finding] = []

        for c, score in contrib_scores.items():
            anom_count = contrib_anom_counts[c]
            detectors_involved = sorted(list(contrib_detectors[c]))
            z_score = (score - mean_score) / (std_score + eps)

            # Threshold Decision Policy:
            # 1. Must cross absolute score threshold (score >= score_threshold).
            # 2. Must either:
            #    a) Have multiple anomalies (anom_count >= 2) OR cross z-score threshold (z_score >= z_threshold).
            #    b) Cross high-confidence compound threshold (score >= 2.0).
            # TRUE-NEGATIVE GUARANTEE: An isolated, weak single signal (e.g. score < 1.0, count == 1)
            # will NOT cross both thresholds and will NOT be flagged.
            is_source_anomaly = (
                score >= self.score_threshold
                and anom_count >= 2
                and (z_score >= self.z_threshold or score >= 2.0)
            )

            if not is_source_anomaly:
                continue

            evidence_keys = sorted(list(contrib_evidence_keys[c]))
            severity = Severity.HIGH if (score >= 3.0 or z_score >= 3.0) else Severity.MEDIUM
            disposition = Disposition.QUARANTINE if (score >= 3.0 or z_score >= 3.0) else Disposition.REVIEW
            confidence = float(min(0.99, max(0.60, 0.50 + 0.12 * score)))

            finding = Finding(
                asset_type=AssetType.SOURCE,
                asset_ref=f"source:{c}",
                detector="cvguard.dataplane.source_aggregator:v1.0",
                reason=(
                    f"Combined multi-detector anomaly concentration flagged from contributor '{c}': "
                    f"cumulative score {score:.2f} (z-score +{z_score:.2f}) across "
                    f"{len(detectors_involved)} detector(s) ({', '.join(detectors_involved)})."
                ),
                evidence=[
                    f"contributor_id:{c}",
                    f"cumulative_score:{score:.4f}",
                    f"z_score:{z_score:.3f}",
                    f"flagged_samples_count:{anom_count}",
                    f"detectors_involved:{','.join(detectors_involved)}",
                ]
                + [f"sample:{k}" for k in evidence_keys],
                confidence=round(confidence, 4),
                severity=severity,
                disposition=disposition,
                assumptions=[
                    "Contributor identifiers truthfully identify distinct data submission origins.",
                    "Co-occurring multi-detector signals from the same source indicate deliberate poisoning or systematic contamination.",
                ],
                limitations=[
                    "Cannot correlate distributed Sybil attacks where an adversary submits from multiple rotating IDs.",
                    "High-volume benign contributors may accumulate minor false-positive scores unless normalized by total volume.",
                ],
            )
            source_findings.append(finding)

        return source_findings
