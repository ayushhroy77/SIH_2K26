"""Out-of-Distribution (OOD) Vision Detector using Mahalanobis Distance in Embedding Space.

Computes the Mahalanobis distance of ingested image embeddings against empirical reference
centroids and regularized precision matrices (inverse covariance) of declared classes.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvguard_schemas import AssetType, Disposition, Finding, Severity

logger = logging.getLogger("cvguard.dataplane.ood_detector")

DEFAULT_OOD_THRESHOLD = float(os.getenv("OOD_MAHALANOBIS_THRESHOLD", "12.0"))
COVARIANCE_REGULARIZATION = 1e-4


@dataclass(frozen=True)
class OODSampleEvaluation:
    """Evaluation result for an individual sample against reference distribution."""

    image_id: int
    filename: str
    contributor_id: str
    minio_key: str
    class_name: str
    distance: float
    threshold: float
    is_ood: bool
    confidence: float
    finding: Finding | None = None


def fit_reference_distribution(
    embeddings: list[list[float]] | np.ndarray,
    reg_eps: float = COVARIANCE_REGULARIZATION,
) -> tuple[list[float], list[list[float]]]:
    """Fit centroid and regularized inverse covariance (precision matrix) from reference embeddings."""
    X = np.array(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {X.shape}")

    n_samples, dim = X.shape
    if n_samples < 2:
        raise ValueError("At least 2 reference samples required to compute empirical covariance.")

    centroid = np.mean(X, axis=0)
    diff = X - centroid
    cov = (diff.T @ diff) / (n_samples - 1)

    # Regularize with epsilon on diagonal to ensure well-conditioned positive-definite inverse
    cov_reg = cov + np.eye(dim) * reg_eps
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov_reg)

    return centroid.tolist(), cov_inv.tolist()


def compute_mahalanobis_distance(
    embedding: np.ndarray,
    centroid: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Compute Mahalanobis distance: sqrt((x - mu)^T * Sigma^{-1} * (x - mu))."""
    delta = embedding - centroid
    dist_sq = float(delta.T @ cov_inv @ delta)
    return math.sqrt(max(0.0, dist_sq))


class OODDetector:
    """Detects out-of-distribution visual samples against registered dataset class distributions."""

    def __init__(self, threshold: float = DEFAULT_OOD_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate_sample(
        self,
        image_id: int,
        filename: str,
        contributor_id: str,
        minio_key: str,
        embedding: np.ndarray,
        class_name: str,
        ref_distribution: dict[str, Any],
    ) -> OODSampleEvaluation:
        """Evaluate single image embedding against class reference distribution."""
        centroid = np.array(ref_distribution["centroid"], dtype=np.float64)
        cov_inv = np.array(ref_distribution["covariance_inv"], dtype=np.float64)

        distance = compute_mahalanobis_distance(embedding, centroid, cov_inv)
        is_ood = distance > self.threshold

        if not is_ood:
            return OODSampleEvaluation(
                image_id=image_id,
                filename=filename,
                contributor_id=contributor_id,
                minio_key=minio_key,
                class_name=class_name,
                distance=distance,
                threshold=self.threshold,
                is_ood=False,
                confidence=0.0,
                finding=None,
            )

        # Confidence scales monotonically with distance-outside-distribution
        excess = distance - self.threshold
        scale_ratio = excess / max(1.0, self.threshold)
        # Scaled smoothly between 0.50 (at threshold boundary) and 0.99 (extreme outlier)
        confidence = float(min(0.99, 0.50 + 0.49 * (1.0 - math.exp(-0.75 * scale_ratio))))

        # High severity if distance is substantially outside (> 1.6x threshold)
        severity = Severity.HIGH if distance > (self.threshold * 1.6) else Severity.MEDIUM

        finding = Finding(
            asset_type=AssetType.SAMPLE,
            asset_ref=f"sample:{minio_key}",
            detector="cvguard.dataplane.ood_mahalanobis:v1.0",
            reason=(
                f"Out-of-distribution sample detected for declared class '{class_name}': "
                f"Mahalanobis distance {distance:.2f} exceeds calibrated threshold {self.threshold:.2f}."
            ),
            evidence=[
                f"minio_key:{minio_key}",
                f"declared_class:{class_name}",
                f"mahalanobis_distance:{distance:.4f}",
                f"threshold:{self.threshold:.4f}",
                f"reference_samples:{ref_distribution.get('num_samples', 0)}",
            ],
            confidence=round(confidence, 4),
            severity=severity,
            disposition=Disposition.REVIEW,
            assumptions=[
                "Reference class distribution exhibits unimodal Gaussian density in feature space.",
                "Feature backbone embeddings are invariant to benign illumination and scale fluctuations.",
            ],
            limitations=[
                "Multimodal class semantics (e.g. diverse sub-breeds or viewpoints) can trigger false positives unless mixture models are used.",
                "Adversarial perturbations specifically crafted to stay within reference covariance ellipsoids will not be detected by pure Mahalanobis metrics.",
            ],
        )

        return OODSampleEvaluation(
            image_id=image_id,
            filename=filename,
            contributor_id=contributor_id,
            minio_key=minio_key,
            class_name=class_name,
            distance=distance,
            threshold=self.threshold,
            is_ood=True,
            confidence=confidence,
            finding=finding,
        )
