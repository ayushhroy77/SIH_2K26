"""Frequency-Domain Trigger and Backdoor Anomaly Detector with Spatial Saliency Localization.

Converts images to 2D Fourier frequency space to detect statistical outliers in high-frequency
energy distribution, and isolates the spatial localization bounding box of the anomalous signal.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from cvguard_schemas import AssetType, Disposition, Finding, Severity

logger = logging.getLogger("cvguard.dataplane.trigger_detector")

DEFAULT_Z_THRESHOLD = float(os.getenv("TRIGGER_Z_THRESHOLD", "2.5"))
DEFAULT_HF_CUTOFF_RATIO = float(os.getenv("TRIGGER_HF_CUTOFF", "0.50"))


@dataclass(frozen=True)
class TriggerEvaluation:
    """Outcome of frequency analysis and spatial trigger localization for an image."""

    image_id: int
    filename: str
    contributor_id: str
    minio_key: str
    hf_energy_ratio: float
    z_score: float
    saliency_bbox: list[float]  # [ymin, xmin, ymax, xmax] normalized [0.0, 1.0]
    is_trigger_anomaly: bool
    confidence: float
    finding: Finding | None = None


def compute_image_frequency_features(
    image_bytes: bytes,
    cutoff_ratio: float = DEFAULT_HF_CUTOFF_RATIO,
) -> tuple[float, list[float]]:
    """Compute 2D FFT magnitude spectrum, high-frequency energy ratio, and localized saliency bbox."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img_gray = img.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img_gray, dtype=np.float64) / 255.0

    H, W = arr.shape
    # 2D Fast Fourier Transform and shift DC component to center
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F)
    mag_spectrum = np.abs(F_shifted)
    power_spectrum = mag_spectrum**2

    total_energy = float(np.sum(power_spectrum))
    if total_energy < 1e-12:
        return 0.0, [0.0, 0.0, 1.0, 1.0]

    # Create high-pass radial frequency mask
    cy, cx = H // 2, W // 2
    y_coords, x_coords = np.ogrid[:H, :W]
    normalized_dist = np.sqrt(((y_coords - cy) / cy) ** 2 + ((x_coords - cx) / cx) ** 2)
    hf_mask = normalized_dist >= cutoff_ratio

    hf_energy = float(np.sum(power_spectrum[hf_mask]))
    hf_ratio = hf_energy / total_energy

    # Spatial saliency localization: Inverse FFT on filtered high frequencies
    F_high = F_shifted * hf_mask
    F_high_unshifted = np.fft.ifftshift(F_high)
    spatial_hf_map = np.abs(np.fft.ifft2(F_high_unshifted))

    # Grid search across 8x8 patches to locate localized energy concentration
    grid_size = 8
    patch_h, patch_w = H // grid_size, W // grid_size
    max_energy = -1.0
    best_grid = (0, 0)

    for gy in range(grid_size):
        for gx in range(grid_size):
            patch = spatial_hf_map[gy * patch_h : (gy + 1) * patch_h, gx * patch_w : (gx + 1) * patch_w]
            energy = float(np.sum(patch**2))
            if energy > max_energy:
                max_energy = energy
                best_grid = (gy, gx)

    gy, gx = best_grid
    # Bounding box normalized coordinates [ymin, xmin, ymax, xmax]
    saliency_bbox = [
        round(gy / grid_size, 3),
        round(gx / grid_size, 3),
        round((gy + 1) / grid_size, 3),
        round((gx + 1) / grid_size, 3),
    ]

    return hf_ratio, saliency_bbox


class TriggerDetector:
    """Baseline backdoor and trigger pattern detector based on spectral frequency anomalies."""

    def __init__(
        self,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        cutoff_ratio: float = DEFAULT_HF_CUTOFF_RATIO,
    ) -> None:
        self.z_threshold = z_threshold
        self.cutoff_ratio = cutoff_ratio

    def evaluate_batch(
        self,
        images: list[dict[str, Any]],
        image_bytes_map: dict[str, bytes],
    ) -> list[TriggerEvaluation]:
        """Evaluate frequency outliers across the ingested batch."""
        if not images:
            return []

        hf_ratios: list[float] = []
        bboxes: list[list[float]] = []

        for img in images:
            raw_bytes = image_bytes_map.get(img["minio_key"], b"")
            if raw_bytes:
                ratio, bbox = compute_image_frequency_features(raw_bytes, self.cutoff_ratio)
            else:
                ratio, bbox = 0.0, [0.0, 0.0, 1.0, 1.0]
            hf_ratios.append(ratio)
            bboxes.append(bbox)

        ratios_arr = np.array(hf_ratios, dtype=np.float64)
        mean_ratio = float(np.mean(ratios_arr))
        std_ratio = float(np.std(ratios_arr))
        eps = 1e-6

        evaluations: list[TriggerEvaluation] = []

        for i, img in enumerate(images):
            ratio = hf_ratios[i]
            bbox = bboxes[i]
            z_score = (ratio - mean_ratio) / (std_ratio + eps)

            is_anomaly = z_score >= self.z_threshold

            if not is_anomaly:
                evaluations.append(
                    TriggerEvaluation(
                        image_id=img["id"],
                        filename=img["filename"],
                        contributor_id=img["contributor_id"],
                        minio_key=img["minio_key"],
                        hf_energy_ratio=ratio,
                        z_score=z_score,
                        saliency_bbox=bbox,
                        is_trigger_anomaly=False,
                        confidence=0.0,
                        finding=None,
                    )
                )
                continue

            # Policy: Noticeably cautious confidence (capped between 0.40 and 0.70)
            # Never set disposition beyond REVIEW — backdoor claims are high-stakes and costly if false
            scaled_conf = 0.40 + 0.30 * min(1.0, max(0.0, (z_score - self.z_threshold) / 2.0))
            confidence = round(float(scaled_conf), 4)

            finding = Finding(
                asset_type=AssetType.SAMPLE,
                asset_ref=f"sample:{img['minio_key']}",
                detector="cvguard.dataplane.trigger_fft_spectral:v1.0",
                reason=(
                    f"Suspicious spectral trigger pattern detected: 2D FFT high-frequency energy ratio "
                    f"({ratio:.4f}) is a statistical outlier (z-score +{z_score:.2f} > threshold {self.z_threshold:.2f})."
                ),
                evidence=[
                    f"minio_key:{img['minio_key']}",
                    f"hf_energy_ratio:{ratio:.5f}",
                    f"batch_mean_hf_ratio:{mean_ratio:.5f}",
                    f"z_score:{z_score:.3f}",
                    f"saliency_bbox_norm:{bbox}",
                    f"cutoff_ratio:{self.cutoff_ratio}",
                ],
                confidence=confidence,
                severity=Severity.MEDIUM,
                disposition=Disposition.REVIEW,  # Cautious policy: NEVER beyond REVIEW
                assumptions=[
                    "Backdoor injection triggers (e.g. checkerboards, high-pass noise) produce sharp high-frequency spectral spikes.",
                    "Natural images in the dataset have typical smooth 1/f power law spectral decay.",
                ],
                limitations=[
                    "Will NOT detect clean-label natural trigger blends, low-frequency subtle color shifts, or imperceptible warping triggers.",
                    "Natural high-frequency textures (e.g. wire fences, gravel, textile weaves) may generate false-positive spectral anomalies.",
                ],
            )

            evaluations.append(
                TriggerEvaluation(
                    image_id=img["id"],
                    filename=img["filename"],
                    contributor_id=img["contributor_id"],
                    minio_key=img["minio_key"],
                    hf_energy_ratio=ratio,
                    z_score=z_score,
                    saliency_bbox=bbox,
                    is_trigger_anomaly=True,
                    confidence=confidence,
                    finding=finding,
                )
            )

        return evaluations
