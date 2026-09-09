"""CVGuard Model Plane Detector Suite."""

from __future__ import annotations

from detectors.activation_clustering import ActivationClusteringDetector
from detectors.reference_battery import ReferenceBatteryDetector
from detectors.strip_detector import STRIPDetector
from detectors.weight_fingerprint import WeightFingerprintDetector

__all__ = [
    "WeightFingerprintDetector",
    "ActivationClusteringDetector",
    "STRIPDetector",
    "ReferenceBatteryDetector",
]
