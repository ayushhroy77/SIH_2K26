"""Near-duplicate vision detector using perceptual hashing (pHash) and anomaly signal emission."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any

import imagehash
from PIL import Image

from cvguard_schemas import AssetType, Disposition, Finding, Severity
from source_aggregator import AnomalySignal

logger = logging.getLogger("cvguard.dataplane.detector")

DEFAULT_HAMMING_THRESHOLD = int(os.getenv("HAMMING_DISTANCE_THRESHOLD", "10"))


@dataclass(frozen=True)
class IngestedImage:
    """Internal representation of an ingested image for detector processing."""

    id: int
    filename: str
    sha256: str
    minio_key: str
    contributor_id: str
    phash: str
    label: str | None = None
    dataset_id: str = "default"


def compute_image_phash(image_bytes: bytes) -> str:
    """Compute 64-bit DCT perceptual hash (pHash) for raw image bytes."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        hash_val = imagehash.phash(img)
        return str(hash_val)


class NearDuplicateDetector:
    """Detects near-duplicate image artifacts using 64-bit DCT perceptual hashing."""

    def __init__(self, threshold: int = DEFAULT_HAMMING_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate_with_signals(
        self,
        images: list[IngestedImage],
    ) -> tuple[list[Finding], list[AnomalySignal]]:
        """Perform pairwise pHash evaluation, returning SAMPLE findings and AnomalySignals for SourceAggregator."""
        if len(images) < 2:
            return [], []

        parsed_hashes = [imagehash.hex_to_hash(img.phash) for img in images]
        sample_findings: list[Finding] = []
        signals: list[AnomalySignal] = []

        n = len(images)
        for i in range(n):
            for j in range(i + 1, n):
                img_a = images[i]
                img_b = images[j]

                distance = parsed_hashes[i] - parsed_hashes[j]

                if distance <= self.threshold:
                    confidence = round(max(0.0, min(1.0, 1.0 - (distance / 64.0))), 4)
                    severity = Severity.HIGH if distance <= 2 else Severity.MEDIUM

                    reason = (
                        f"Near-duplicate image pair detected: '{img_a.filename}' and "
                        f"'{img_b.filename}' with Hamming distance {distance} "
                        f"(threshold <= {self.threshold})."
                    )

                    sample_finding = Finding(
                        asset_type=AssetType.SAMPLE,
                        asset_ref=f"sample:{img_a.sha256[:16]}_{img_b.sha256[:16]}",
                        detector="cvguard.detector.phash_near_duplicate:v1.0",
                        reason=reason,
                        evidence=[img_a.minio_key, img_b.minio_key],
                        confidence=confidence,
                        severity=severity,
                        disposition=Disposition.QUARANTINE if distance <= 1 else Disposition.REVIEW,
                        assumptions=[
                            "Evaluates 64-bit DCT perceptual hash (pHash) invariance to scale, compression, and slight tint."
                        ],
                        limitations=[
                            "pHash cannot detect semantic/embedding-level near duplicates or crop/rotations exceeding frequency threshold."
                        ],
                    )
                    sample_findings.append(sample_finding)

                    # Emit per-sample anomaly signals for shared SourceAggregator
                    signals.append(
                        AnomalySignal(
                            image_id=img_a.id,
                            contributor_id=img_a.contributor_id,
                            minio_key=img_a.minio_key,
                            detector_name="phash_near_duplicate",
                            score=confidence,
                            is_anomaly=True,
                            weight=1.0,
                        )
                    )
                    signals.append(
                        AnomalySignal(
                            image_id=img_b.id,
                            contributor_id=img_b.contributor_id,
                            minio_key=img_b.minio_key,
                            detector_name="phash_near_duplicate",
                            score=confidence,
                            is_anomaly=True,
                            weight=1.0,
                        )
                    )

        logger.info(
            "Near-duplicate evaluation completed: %d sample findings, %d anomaly signals",
            len(sample_findings),
            len(signals),
        )
        return sample_findings, signals

    def evaluate_batch(
        self,
        images: list[IngestedImage],
    ) -> list[Finding]:
        """Legacy compatibility wrapper returning sample findings."""
        findings, _ = self.evaluate_with_signals(images)
        return findings
