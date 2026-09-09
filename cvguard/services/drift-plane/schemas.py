"""Pydantic V2 schemas for CVGuard Drift Plane Service."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class SampleInput(BaseModel):
    """Single input sample for reference profiling or batch assessment."""

    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded raw image file bytes (PNG, JPEG, etc.).",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Pre-extracted visual feature embedding vector (optional for direct testing).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Operational metadata key-values (e.g. sensor_id, season, illumination, timestamp).",
    )


class CreateProfileRequest(BaseModel):
    """Payload for POST /reference-profile."""

    profile_id: str | None = Field(
        default=None,
        description="Unique identifier for the reference profile (auto-generated if omitted).",
    )
    version: int | None = Field(
        default=None,
        description="Version number for the profile (auto-increments if omitted).",
    )
    samples: list[SampleInput] = Field(
        ...,
        min_length=1,
        description="Batch of reference image samples representing declared normal operational state.",
    )


class CreateProfileResponse(BaseModel):
    """Response returned by POST /reference-profile."""

    profile_id: str = Field(description="Assigned profile identifier.")
    version: int = Field(description="Profile version number.")
    num_samples: int = Field(description="Number of ingested reference samples.")
    embedding_dim: int = Field(description="Dimensionality of visual embedding representations.")
    centroid_summary: list[float] = Field(description="First 8 dimensions of the computed reference centroid.")
    metadata_histograms: dict[str, dict[str, int]] = Field(
        description="Frequency counts per categorical metadata attribute.",
    )
    created_at: str = Field(description="ISO-8601 UTC creation timestamp.")


class AssessBatchRequest(BaseModel):
    """Payload for POST /assess-batch."""

    profile_id: str = Field(..., description="Target reference profile ID to compare against.")
    version: int | None = Field(
        default=None,
        description="Target profile version (defaults to latest if omitted).",
    )
    batch_id: str | None = Field(
        default=None,
        description="Optional unique identifier for the incoming assessment batch.",
    )
    samples: list[SampleInput] = Field(
        ...,
        min_length=1,
        description="Batch of query image samples to evaluate for distribution shift.",
    )


class AssessBatchResponse(BaseModel):
    """Response returned by POST /assess-batch."""

    batch_id: str = Field(description="Assessment batch identifier.")
    profile_id: str = Field(description="Reference profile ID used for baseline comparison.")
    profile_version: int = Field(description="Version number of reference profile.")
    sample_count: int = Field(description="Number of samples evaluated in this batch.")
    mmd_statistic: float = Field(description="Maximum Mean Discrepancy (MMD) statistical divergence.")
    mmd_standard_error: float = Field(description="Standard error of the MMD divergence estimate.")
    ks_shifted_dimension_ratio: float = Field(
        description="Fraction of embedding dimensions exhibiting statistically significant shift (p < 0.05).",
    )
    ks_mean_statistic: float = Field(description="Mean Kolmogorov-Smirnov statistic across all dimensions.")
    risk_score: float = Field(description="Calibrated distribution shift risk score in [0.0, 1.0].")
    confidence_interval: list[float] = Field(
        description="95% confidence interval [lower_bound, upper_bound] for the calibrated risk score.",
    )
    standard_error: float = Field(description="Standard error of the risk estimate.")
    classification: str = Field(
        description="Statistical classification: 'no_drift', 'probable_operational_drift', 'suspicious_manipulation', or 'indeterminate'.",
    )
    reason: str = Field(description="Plain-language explanation of findings and evidence.")
    evidence: list[str] = Field(description="Detailed statistical indicators and metric strings.")
    correlated_metadata_field: str | None = Field(
        default=None,
        description="Identified operational metadata field explaining shift (if operational drift).",
    )
    governance_ledger_id: int | None = Field(
        default=None,
        description="Ledger entry ID sealed into Governance Spine audit trail.",
    )
    signed_finding: dict[str, Any] | None = Field(
        default=None,
        description="Full cryptographic SignedFinding returned by Governance Spine.",
    )
    limitations: list[str] = Field(
        description="Explicit caveats regarding metadata dependency and threat boundary assumptions.",
    )
