"""CVGuard Reference Battery Canonical Fixtures and Test Inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def load_battery_manifest() -> dict[str, Any]:
    """Load the canonical reference battery manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_canonical_battery_inputs() -> list[dict[str, Any]]:
    """Return canonical test input vectors corresponding to manifest items.

    Each item includes:
      - item_id
      - tensor_data (flattened list of floats)
      - expected_envelope
    """
    manifest = load_battery_manifest()
    items = manifest.get("items", [])
    battery_inputs = []

    for item in items:
        # Generate deterministic synthetic probe vectors (16 values representing downscaled feature maps)
        idx = int(item["item_id"].split("_")[1])
        base_val = float(idx) * 0.15
        vector = [(base_val + i * 0.05) % 1.0 for i in range(16)]

        battery_inputs.append({
            "item_id": item["item_id"],
            "description": item["description"],
            "expected_top_class": item.get("expected_top_class"),
            "min_confidence": item.get("min_confidence", 0.2),
            "min_entropy": item.get("min_entropy", 0.3),
            "max_entropy": item.get("max_entropy", 3.2),
            "vector": vector,
        })

    return battery_inputs
