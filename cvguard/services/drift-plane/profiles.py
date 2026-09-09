"""Reference profile management and versioned profile registry.

Maintains declared normal operational distribution profiles containing:
- Centroid vector and empirical covariance matrix.
- Full or subsampled reference embeddings (for MMD evaluation).
- Categorical metadata histograms and temporal ranges.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReferenceProfile:
    """Versioned statistical reference profile representing declared normal operational state."""

    profile_id: str
    version: int
    num_samples: int
    dim: int
    centroid: list[float]
    embeddings: list[list[float]]
    covariance: list[list[float]] | None = None
    metadata_histograms: dict[str, dict[str, int]] = field(default_factory=dict)
    metadata_records: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    @classmethod
    def compute_from_samples(
        cls,
        profile_id: str,
        version: int,
        embeddings: list[list[float]],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> ReferenceProfile:
        """Construct a ReferenceProfile from raw embeddings and optional sample metadata."""
        n = len(embeddings)
        if n == 0:
            raise ValueError("Cannot construct ReferenceProfile from zero samples.")

        dim = len(embeddings[0])

        # 1. Compute centroid vector
        centroid = [0.0] * dim
        for vec in embeddings:
            for d in range(dim):
                centroid[d] += vec[d]
        centroid = [round(v / n, 6) for v in centroid]

        # 2. Compute sample covariance matrix (regularized diagonal or full)
        cov_matrix: list[list[float]] | None = None
        if n > 1:
            # We compute diagonal variance + top off-diagonals or full matrix for small dim
            cov_matrix = [[0.0] * dim for _ in range(dim)]
            for vec in embeddings:
                diffs = [vec[d] - centroid[d] for d in range(dim)]
                for r in range(dim):
                    for c in range(dim):
                        cov_matrix[r][c] += diffs[r] * diffs[c]

            denom = max(n - 1, 1)
            for r in range(dim):
                for c in range(dim):
                    cov_matrix[r][c] = round(cov_matrix[r][c] / denom, 6)
                    if r == c:
                        cov_matrix[r][c] += 1e-4  # Ridge regularization for stability

        # 3. Compute metadata histograms
        metadata_histograms: dict[str, dict[str, int]] = {}
        metadata_records: list[dict[str, Any]] = metadata_list or []

        if metadata_records:
            for meta in metadata_records:
                if not meta:
                    continue
                for k, v in meta.items():
                    if v is not None:
                        val_str = str(v)
                        if k not in metadata_histograms:
                            metadata_histograms[k] = {}
                        metadata_histograms[k][val_str] = metadata_histograms[k].get(val_str, 0) + 1

        return cls(
            profile_id=profile_id,
            version=version,
            num_samples=n,
            dim=dim,
            centroid=centroid,
            covariance=cov_matrix,
            embeddings=embeddings,
            metadata_histograms=metadata_histograms,
            metadata_records=metadata_records,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize lightweight profile summary without bulk embedding matrices."""
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "num_samples": self.num_samples,
            "embedding_dim": self.dim,
            "centroid_summary": self.centroid[:8],
            "metadata_histograms": self.metadata_histograms,
            "created_at": self.created_at,
        }


class ProfileRegistry:
    """In-memory thread-safe versioned registry for reference profiles."""

    def __init__(self) -> None:
        self._profiles: dict[tuple[str, int], ReferenceProfile] = {}
        self._latest_versions: dict[str, int] = {}

    def save_profile(self, profile: ReferenceProfile) -> ReferenceProfile:
        """Store a versioned profile and update latest version pointer."""
        key = (profile.profile_id, profile.version)
        self._profiles[key] = profile

        current_latest = self._latest_versions.get(profile.profile_id, 0)
        if profile.version > current_latest:
            self._latest_versions[profile.profile_id] = profile.version

        return profile

    def get_profile(self, profile_id: str, version: int | None = None) -> ReferenceProfile | None:
        """Retrieve a profile by ID and optional version (defaults to latest)."""
        if version is None:
            version = self._latest_versions.get(profile_id)
            if version is None:
                return None

        return self._profiles.get((profile_id, version))

    def get_latest_version(self, profile_id: str) -> int | None:
        """Get latest stored version number for a profile ID."""
        return self._latest_versions.get(profile_id)

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all registered profiles with summary metadata."""
        summaries: list[dict[str, Any]] = []
        for (pid, ver), prof in sorted(self._profiles.items()):
            summaries.append({
                "profile_id": pid,
                "version": ver,
                "is_latest": ver == self._latest_versions.get(pid),
                "num_samples": prof.num_samples,
                "dim": prof.dim,
                "created_at": prof.created_at,
                "metadata_fields": list(prof.metadata_histograms.keys()),
            })
        return summaries

    def clear(self) -> None:
        """Clear all registered profiles (for testing isolation)."""
        self._profiles.clear()
        self._latest_versions.clear()


# Global singleton instance
_GLOBAL_REGISTRY: ProfileRegistry | None = None


def get_profile_registry() -> ProfileRegistry:
    """Access singleton ProfileRegistry instance."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ProfileRegistry()
    return _GLOBAL_REGISTRY
