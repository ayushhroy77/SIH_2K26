"""CVGuard White-Box Activation Clustering Detector.

Extracts penultimate-layer feature activations across an internal probe dataset and
evaluates whether any class's activations form an anomalous split sub-cluster.
A tight, segregated sub-cluster within a single semantic class strongly indicates
a targeted backdoor where triggered samples activate distinct latent pathways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cvguard_schemas import AssetType, Disposition, Finding, Severity


def _euclidean_dist(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two numeric vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans_k2(vectors: list[list[float]], max_iter: int = 25) -> tuple[list[int], list[list[float]], float]:
    """Execute KMeans with k=2 on a list of vectors.

    Returns:
        (assignments, centroids, silhouette_score)
    """
    n = len(vectors)
    if n < 4:
        return [0] * n, [[0.0] * len(vectors[0]), [0.0] * len(vectors[0])], 0.0

    dim = len(vectors[0])

    # Initial centroids: choose first vector and the vector furthest from it
    c0 = list(vectors[0])
    c1 = max(vectors, key=lambda v: _euclidean_dist(v, c0))
    c1 = list(c1)

    assignments = [0] * n

    for _ in range(max_iter):
        new_assignments = []
        for v in vectors:
            d0 = _euclidean_dist(v, c0)
            d1 = _euclidean_dist(v, c1)
            new_assignments.append(0 if d0 <= d1 else 1)

        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Recompute centroids
        cluster_0 = [vectors[i] for i in range(n) if assignments[i] == 0]
        cluster_1 = [vectors[i] for i in range(n) if assignments[i] == 1]

        if not cluster_0 or not cluster_1:
            # Degenerate single cluster
            return assignments, [c0, c1], 0.0

        c0 = [sum(v[d] for v in cluster_0) / len(cluster_0) for d in range(dim)]
        c1 = [sum(v[d] for v in cluster_1) / len(cluster_1) for d in range(dim)]

    # Compute silhouette score approximation
    inter_cluster_dist = _euclidean_dist(c0, c1)
    intra_cluster_0 = sum(_euclidean_dist(v, c0) for v in cluster_0) / len(cluster_0)
    intra_cluster_1 = sum(_euclidean_dist(v, c1) for v in cluster_1) / len(cluster_1)
    avg_intra = (intra_cluster_0 + intra_cluster_1) / 2.0

    separation_score = inter_cluster_dist / (avg_intra + 1e-6)
    return assignments, [c0, c1], separation_score


@dataclass
class ClusterAnomaly:
    """Summary of a suspicious activation sub-cluster within a target class."""

    class_label: str
    sample_count: int
    cluster_0_size: int
    cluster_1_size: int
    separation_ratio: float
    minority_cluster_ratio: float


class ActivationClusteringDetector:
    """White-box detector evaluating activation manifold splitting for backdoor signatures."""

    DETECTOR_ID = "cvguard.modelplane.activation_clustering:v1.0"

    def __init__(
        self,
        min_samples_per_class: int = 6,
        separation_threshold: float = 2.2,
        max_minority_ratio: float = 0.40,
        min_minority_ratio: float = 0.05,
    ) -> None:
        self.min_samples_per_class = min_samples_per_class
        self.separation_threshold = separation_threshold
        self.max_minority_ratio = max_minority_ratio
        self.min_minority_ratio = min_minority_ratio

    def analyze_activations(
        self,
        activation_data: dict[str, Any],
        asset_ref: str,
        access_level: str = "white-box",
    ) -> list[Finding]:
        """Perform KMeans k=2 activation clustering per class and emit findings on suspicious splits."""
        findings: list[Finding] = []
        class_activations: dict[str, list[list[float]]] = activation_data.get("class_activations", {})

        suspicious_classes: list[ClusterAnomaly] = []

        for class_label, samples in class_activations.items():
            if len(samples) < self.min_samples_per_class:
                continue

            assignments, _, separation_ratio = _kmeans_k2(samples)
            c0_count = sum(1 for a in assignments if a == 0)
            c1_count = len(samples) - c0_count

            if c0_count == 0 or c1_count == 0:
                continue

            minority_count = min(c0_count, c1_count)
            minority_ratio = minority_count / len(samples)

            # A suspicious backdoor sub-cluster is characterized by:
            # 1. High separation ratio (clusters clearly distinct in feature space)
            # 2. Minority cluster between 5% and 40% of class samples (indicates poisoned sub-population)
            if (
                separation_ratio >= self.separation_threshold
                and self.min_minority_ratio <= minority_ratio <= self.max_minority_ratio
            ):
                suspicious_classes.append(
                    ClusterAnomaly(
                        class_label=class_label,
                        sample_count=len(samples),
                        cluster_0_size=c0_count,
                        cluster_1_size=c1_count,
                        separation_ratio=round(separation_ratio, 3),
                        minority_cluster_ratio=round(minority_ratio, 3),
                    )
                )

        if suspicious_classes:
            primary_anomaly = suspicious_classes[0]
            evidence = [
                f"flagged_class_count: {len(suspicious_classes)}",
                f"primary_flagged_class: {primary_anomaly.class_label}",
                f"class_sample_count: {primary_anomaly.sample_count}",
                f"sub_cluster_distribution: [{primary_anomaly.cluster_0_size}, {primary_anomaly.cluster_1_size}]",
                f"separation_ratio: {primary_anomaly.separation_ratio} (threshold {self.separation_threshold})",
                f"minority_sub_population_ratio: {primary_anomaly.minority_cluster_ratio:.1%}",
            ]

            # Confidence scaled conservatively by separation strength
            confidence = round(
                min(0.65 + (primary_anomaly.separation_ratio - self.separation_threshold) * 0.08, 0.88),
                4,
            )

            findings.append(
                Finding(
                    asset_type=AssetType.MODEL,
                    asset_ref=asset_ref,
                    detector=self.DETECTOR_ID,
                    reason=(
                        f"Bimodal activation sub-clustering detected in class '{primary_anomaly.class_label}': "
                        f"penultimate features split into distinct clusters with separation ratio "
                        f"{primary_anomaly.separation_ratio} (suggestive of a latent backdoor payload)."
                    ),
                    evidence=evidence,
                    confidence=confidence,
                    severity=Severity.HIGH,
                    disposition=Disposition.REVIEW,
                    assumptions=[
                        f"Model access level: {access_level} (internal layer activations accessible).",
                        "Internal probe dataset contains both representative clean samples and triggered/poisoned samples.",
                        "Clean samples in a single semantic class follow a unimodal connected manifold in activation space.",
                    ],
                    limitations=[
                        "Does not detect backdoors if the probe dataset contains zero triggered instances.",
                        "Naturally multimodal visual classes (e.g. 'sports car' vs 'pickup truck' under 'vehicle') "
                        "can induce legitimate activation clustering.",
                    ],
                )
            )

        return findings
