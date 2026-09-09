"""Shared Vision Backbone Feature Extractor for CVGuard Drift Plane.

Reuses the self-hosted embedding extractor from CVGuard Data Plane
to avoid duplicating ONNX Runtime model loading and preprocessing code.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Dynamically ensure services/data-plane is in sys.path
_DATA_PLANE_DIR = Path(__file__).resolve().parent.parent / "data-plane"
if _DATA_PLANE_DIR.exists() and str(_DATA_PLANE_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_PLANE_DIR))

logger = logging.getLogger("cvguard.driftplane.extractor")

try:
    from embeddings import EmbeddingExtractor, get_embedding_extractor
except ImportError:
    try:
        from cvguard.services.data_plane.embeddings import (  # type: ignore
            EmbeddingExtractor,
            get_embedding_extractor,
        )
    except ImportError:
        logger.warning(
            "Could not load EmbeddingExtractor from data-plane. "
            "Ensure services/data-plane/embeddings.py is accessible."
        )
        raise

__all__ = ["EmbeddingExtractor", "get_embedding_extractor"]
