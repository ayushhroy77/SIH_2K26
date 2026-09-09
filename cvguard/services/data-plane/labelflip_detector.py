"""Label-Flip and Systematic Mislabelling Detector using k-Nearest-Neighbors in Embedding Space.

Identifies samples whose declared annotation contradicts the consensus label of its
k nearest semantic neighbors in the dense representation manifold.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvguard_schemas import AssetType, Disposition, Finding, Severity

logger = logging.getLogger("cvguard.dataplane.labelflip_detector")

DEFAULT_K_NEIGHBORS = int(os.getenv("LABEL_FLIP_K_NEIGHBORS", "10"))
DEFAULT_DISCORDANCE_THRESHOLD = float(os.getenv("LABEL_FLIP_THRESHOLD", "0.70"))


@dataclass(frozen=True)
class LabeledSample:
    """Image representation with feature embedding and declared class annotation."""

    image_id: int
    filename: str
    contributor_id: str
    minio_key: str
    embedding: np.ndarray
    label: str


@dataclass(frozen=True)
class LabelFlipEvaluation:
    """Outcome of kNN label neighborhood analysis for a sample."""

    image_id: int
    filename: str
    contributor_id: str
    minio_key: str
    declared_label: str
    dominant_neighbor_label: str
    discordance_rate: float
    k: int
    is_mislabeled: bool
    finding: Finding | None = None


class LabelFlipDetector:
    """Detects systematic mislabelling by inspecting semantic neighborhood label consensus."""

    def __init__(
        self,
        k: int = DEFAULT_K_NEIGHBORS,
        threshold: float = DEFAULT_DISCORDANCE_THRESHOLD,
    ) -> None:
        self.k = k
        self.threshold = threshold

    def evaluate_batch(
        self,
        samples: list[LabeledSample],
    ) -> list[LabelFlipEvaluation]:
        """Evaluate all labeled samples in the batch against their nearest neighbors."""
        n = len(samples)
        if n < 2:
            return []

        # Effective k cannot exceed n - 1
        effective_k = min(self.k, n - 1)

        # Stack embeddings matrix (n, dim)
        embeddings_matrix = np.vstack([s.embedding for s in samples])
        # Compute pairwise cosine similarity: dot product of L2-normalized embeddings
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normed_matrix = embeddings_matrix / norms
        sim_matrix = normed_matrix @ normed_matrix.T  # (n, n)

        evaluations: list[LabelFlipEvaluation] = []

        for i in range(n):
            current = samples[i]
            if not current.label:
                continue

            # Sort neighbors by descending cosine similarity, excluding self (index i)
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf  # Mask self
            neighbor_indices = np.argsort(sims)[::-1][:effective_k]

            neighbor_labels = [samples[idx].label for idx in neighbor_indices if samples[idx].label]
            if not neighbor_labels:
                continue

            label_counts = Counter(neighbor_labels)
            dominant_label, dominant_count = label_counts.most_common(1)[0]

            # Discordance: fraction of neighbors that do NOT match the declared label
            discordant_count = sum(1 for l in neighbor_labels if l != current.label)
            discordance_rate = float(discordant_count / len(neighbor_labels))

            is_flipped = discordance_rate >= self.threshold and dominant_label != current.label

            if not is_flipped:
                evaluations.append(
                    LabelFlipEvaluation(
                        image_id=current.image_id,
                        filename=current.filename,
                        contributor_id=current.contributor_id,
                        minio_key=current.minio_key,
                        declared_label=current.label,
                        dominant_neighbor_label=dominant_label,
                        discordance_rate=discordance_rate,
                        k=effective_k,
                        is_mislabeled=False,
                        finding=None,
                    )
                )
                continue

            severity = Severity.HIGH if discordance_rate >= 0.85 else Severity.MEDIUM

            finding = Finding(
                asset_type=AssetType.SAMPLE,
                asset_ref=f"sample:{current.minio_key}",
                detector="cvguard.dataplane.label_flip_knn:v1.0",
                reason=(
                    f"Possible label-flip: sample annotated as '{current.label}' has "
                    f"{discordant_count}/{len(neighbor_labels)} ({discordance_rate * 100:.0f}%) nearest neighbors "
                    f"labeled as '{dominant_label}' (threshold {self.threshold * 100:.0f}%)."
                ),
                evidence=[
                    f"minio_key:{current.minio_key}",
                    f"declared_label:{current.label}",
                    f"dominant_neighbor_label:{dominant_label}",
                    f"discordance_rate:{discordance_rate:.4f}",
                    f"k_neighbors:{effective_k}",
                    f"neighbor_consensus_breakdown:{dict(label_counts)}",
                ],
                confidence=round(discordance_rate, 4),
                severity=severity,
                disposition=Disposition.REVIEW,
                assumptions=[
                    "Semantic visual closeness in representation space corresponds to ground-truth category identity.",
                    "The surrounding neighborhood contains predominantly genuine, uncorrupted labels.",
                ],
                limitations=[
                    "Cannot detect coordinated symmetric label inversion (e.g. 100% of 'airplane' and 'bird' swapped simultaneously).",
                    "Fine-grained categories with ambiguous visual boundaries may trigger false-positive discordance.",
                ],
            )

            evaluations.append(
                LabelFlipEvaluation(
                    image_id=current.image_id,
                    filename=current.filename,
                    contributor_id=current.contributor_id,
                    minio_key=current.minio_key,
                    declared_label=current.label,
                    dominant_neighbor_label=dominant_label,
                    discordance_rate=discordance_rate,
                    k=effective_k,
                    is_mislabeled=True,
                    finding=finding,
                )
            )

        return evaluations
