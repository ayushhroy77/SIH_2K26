"""Unit test suite for CVGuard Phase 3 Data Plane detectors and unified source aggregator.

Tests:
1. OOD Detector against handcrafted in-distribution and out-of-distribution fixtures.
2. Label-Flip Detector against a deliberately mislabeled semantic fixture.
3. Trigger Detector against a synthetic high-frequency checkerboard trigger patch.
4. SourceAggregator True-Positive (multi-signal adversary) and True-Negative (scattered/low signal).
"""

from __future__ import annotations

import io
import math
import numpy as np
import pytest
from PIL import Image, ImageDraw

from cvguard_schemas import AssetType, Disposition, Finding, Severity
from labelflip_detector import LabeledSample, LabelFlipDetector
from ood_detector import OODDetector, fit_reference_distribution
from source_aggregator import AnomalySignal, SourceAggregator
from trigger_detector import TriggerDetector, compute_image_frequency_features


def create_smooth_test_image(color: tuple[int, int, int]) -> bytes:
    """Generate a smooth low-frequency gradient/circle image."""
    img = Image.new("RGB", (128, 128), color=color)
    draw = ImageDraw.Draw(img)
    # Smooth broad circular shapes (low spatial frequency)
    draw.ellipse([20, 20, 108, 108], fill=(color[0] // 2, color[1] // 2, color[2] // 2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def create_high_frequency_trigger_image() -> bytes:
    """Generate an image with an injected sharp high-frequency checkerboard trigger patch."""
    img = Image.new("RGB", (128, 128), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    # Draw smooth background
    draw.ellipse([10, 10, 80, 80], fill=(70, 70, 70))
    # Inject high-frequency alternating 2x2 checkerboard trigger in top-right quadrant [0:32, 96:128]
    for y in range(0, 32, 2):
        for x in range(96, 128, 2):
            draw.point((x, y), fill=(255, 255, 255))
            draw.point((x + 1, y + 1), fill=(255, 255, 255))
            draw.point((x + 1, y), fill=(0, 0, 0))
            draw.point((x, y + 1), fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# =============================================================================
# 1. Unit Test: Out-of-Distribution (OOD) Detector
# =============================================================================
def test_ood_detector_with_handcrafted_fixtures():
    """Verify Mahalanobis distance correctly flags OOD samples and passes in-distribution samples."""
    dim = 8
    rng = np.random.RandomState(42)

    # 1. Construct empirical reference distribution: 20 samples around centroid
    base_centroid = np.full(dim, 0.5, dtype=np.float64)
    ref_samples = [base_centroid + rng.normal(0, 0.05, size=dim) for _ in range(20)]
    centroid, cov_inv = fit_reference_distribution(ref_samples)

    ref_dist = {
        "dataset_id": "test_dataset",
        "class_name": "class_vehicle",
        "centroid": centroid,
        "covariance_inv": cov_inv,
        "num_samples": len(ref_samples),
    }

    detector = OODDetector(threshold=8.0)

    # 2. In-distribution sample: very close to centroid
    in_dist_vec = base_centroid + np.full(dim, 0.02)
    in_eval = detector.evaluate_sample(
        image_id=1,
        filename="in_dist.jpg",
        contributor_id="contrib_benign",
        minio_key="k_in_dist",
        embedding=in_dist_vec,
        class_name="class_vehicle",
        ref_distribution=ref_dist,
    )
    assert not in_eval.is_ood
    assert in_eval.distance < detector.threshold
    assert in_eval.finding is None

    # 3. Hand-crafted OOD sample: heavily displaced along feature dimensions
    ood_vec = base_centroid + np.full(dim, 1.5)
    ood_eval = detector.evaluate_sample(
        image_id=2,
        filename="ood_sample.jpg",
        contributor_id="contrib_poisoner",
        minio_key="k_ood",
        embedding=ood_vec,
        class_name="class_vehicle",
        ref_distribution=ref_dist,
    )
    assert ood_eval.is_ood
    assert ood_eval.distance > detector.threshold
    assert ood_eval.finding is not None
    assert ood_eval.finding.asset_type == AssetType.SAMPLE
    assert ood_eval.finding.disposition == Disposition.REVIEW
    # Confidence scales monotonically with distance outside threshold
    assert 0.50 <= ood_eval.confidence <= 0.99
    assert "class_vehicle" in ood_eval.finding.reason


# =============================================================================
# 2. Unit Test: Label-Flip / Systematic Mislabelling Detector
# =============================================================================
def test_labelflip_detector_with_mislabeled_fixture():
    """Verify kNN consensus flags a sample whose neighbors have a discordant majority label."""
    dim = 8
    # Cluster A: centered around [1, 0, 0, 0, ...]
    cluster_a = np.zeros(dim)
    cluster_a[0] = 1.0
    # Cluster B: centered around [0, 1, 0, 0, ...]
    cluster_b = np.zeros(dim)
    cluster_b[1] = 1.0

    samples: list[LabeledSample] = []
    # 8 clean samples in Cluster A labeled as "cat"
    for i in range(8):
        emb = cluster_a + np.random.normal(0, 0.05, size=dim)
        emb /= np.linalg.norm(emb)
        samples.append(
            LabeledSample(
                image_id=i + 1,
                filename=f"cat_{i}.jpg",
                contributor_id="contrib_clean",
                minio_key=f"key_cat_{i}",
                embedding=emb,
                label="cat",
            )
        )

    # 1 deliberately mislabeled sample: located inside Cluster A, but labeled as "dog"
    mislabeled_emb = cluster_a + np.random.normal(0, 0.02, size=dim)
    mislabeled_emb /= np.linalg.norm(mislabeled_emb)
    mislabeled_sample = LabeledSample(
        image_id=99,
        filename="poison_dog.jpg",
        contributor_id="contrib_adversary",
        minio_key="key_poison_dog",
        embedding=mislabeled_emb,
        label="dog",
    )
    samples.append(mislabeled_sample)

    detector = LabelFlipDetector(k=5, threshold=0.70)
    evaluations = detector.evaluate_batch(samples)

    # The mislabeled sample should be flagged: its 5 nearest neighbors are all "cat" (100% discordance)
    target_eval = next((e for e in evaluations if e.image_id == 99), None)
    assert target_eval is not None
    assert target_eval.is_mislabeled
    assert target_eval.discordance_rate >= 0.70
    assert target_eval.dominant_neighbor_label == "cat"
    assert target_eval.finding is not None
    assert target_eval.finding.asset_type == AssetType.SAMPLE
    assert "Suspected label-flip" in target_eval.finding.reason or "label" in target_eval.finding.reason.lower()

    # Clean cat samples should NOT be flagged
    clean_evals = [e for e in evaluations if e.image_id != 99]
    for ce in clean_evals:
        assert not ce.is_mislabeled


# =============================================================================
# 3. Unit Test: Trigger / Backdoor Detector (FFT Frequency Space)
# =============================================================================
def test_trigger_detector_with_synthetic_high_frequency_patch():
    """Verify 2D FFT spectral analysis flags high-frequency checkerboard outlier."""
    # Create batch: 7 smooth images + 1 trigger patch image
    images_meta = []
    image_bytes_map = {}

    colors = [
        (100, 150, 200),
        (50, 120, 80),
        (180, 80, 50),
        (220, 200, 100),
        (90, 90, 110),
        (140, 110, 140),
        (80, 160, 160),
    ]

    for idx, c in enumerate(colors):
        b = create_smooth_test_image(c)
        k = f"smooth_{idx}.jpg"
        images_meta.append(
            {"id": idx + 1, "filename": k, "contributor_id": "contrib_normal", "minio_key": k}
        )
        image_bytes_map[k] = b

    # Add trigger image
    trigger_bytes = create_high_frequency_trigger_image()
    trigger_key = "trigger_image.jpg"
    images_meta.append(
        {"id": 88, "filename": trigger_key, "contributor_id": "contrib_backdoor", "minio_key": trigger_key}
    )
    image_bytes_map[trigger_key] = trigger_bytes

    detector = TriggerDetector(z_threshold=2.0)
    evaluations = detector.evaluate_batch(images_meta, image_bytes_map)

    trigger_eval = next((e for e in evaluations if e.image_id == 88), None)
    assert trigger_eval is not None
    assert trigger_eval.is_trigger_anomaly
    assert trigger_eval.z_score >= detector.z_threshold
    assert trigger_eval.finding is not None
    assert trigger_eval.finding.asset_type == AssetType.SAMPLE
    # Cautious policy checks
    assert trigger_eval.finding.confidence <= 0.70
    assert trigger_eval.finding.disposition == Disposition.REVIEW
    assert "limitations" in trigger_eval.finding.model_dump()

    # Clean smooth images must not be flagged
    for e in evaluations:
        if e.image_id != 88:
            assert not e.is_trigger_anomaly


# =============================================================================
# 4. Unit Test: Unified SourceAggregator (True-Positive & True-Negative)
# =============================================================================
def test_source_aggregator_true_positive_and_true_negative():
    """Verify SourceAggregator flags combined-weak multi-signal adversary and DOES NOT flag scattered single-signal contributor."""
    signals: list[AnomalySignal] = []

    # Adversary "contrib_adversary": 2 individually moderate signals across 2 detectors
    # Signal 1: OOD (score 0.85, weight 1.0)
    signals.append(
        AnomalySignal(
            image_id=101,
            contributor_id="contrib_adversary",
            minio_key="adv_sample_1",
            detector_name="ood_mahalanobis",
            score=0.85,
            is_anomaly=True,
            weight=1.0,
        )
    )
    # Signal 2: Label-flip (score 0.80, weight 1.2) -> weighted 0.96
    signals.append(
        AnomalySignal(
            image_id=102,
            contributor_id="contrib_adversary",
            minio_key="adv_sample_2",
            detector_name="label_flip_knn",
            score=0.80,
            is_anomaly=True,
            weight=1.2,
        )
    )
    # Total score for contrib_adversary = 0.85 + 0.96 = 1.81 (crosses threshold 1.50)

    # Benign User "contrib_benign": ONLY 1 isolated weak anomaly (e.g. slight border blur)
    signals.append(
        AnomalySignal(
            image_id=201,
            contributor_id="contrib_benign",
            minio_key="benign_sample_1",
            detector_name="trigger_fft_spectral",
            score=0.45,
            is_anomaly=True,
            weight=0.8,
        )
    )
    # Total score for contrib_benign = 0.36 (does NOT cross threshold 1.50)

    # Clean User "contrib_clean": 0 anomalies
    all_contributors = ["contrib_adversary", "contrib_benign", "contrib_clean", "contrib_clean_2"]

    aggregator = SourceAggregator(score_threshold=1.50, z_threshold=1.20)
    source_findings = aggregator.aggregate(signals, all_contributors=all_contributors)

    flagged_contributors = [f.asset_ref.replace("source:", "") for f in source_findings]

    # True-Positive Assertion:
    assert "contrib_adversary" in flagged_contributors
    adv_finding = next(f for f in source_findings if "contrib_adversary" in f.asset_ref)
    assert adv_finding.asset_type == AssetType.SOURCE
    assert "ood_mahalanobis" in adv_finding.reason or "ood_mahalanobis" in str(adv_finding.evidence)
    assert "label_flip_knn" in adv_finding.reason or "label_flip_knn" in str(adv_finding.evidence)

    # True-Negative Assertion (CRITICAL):
    # Isolated weak signals MUST NOT produce an AssetType.SOURCE finding
    assert "contrib_benign" not in flagged_contributors
    assert "contrib_clean" not in flagged_contributors
    assert len(source_findings) == 1
