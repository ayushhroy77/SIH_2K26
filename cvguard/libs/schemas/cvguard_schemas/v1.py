"""Version 1 of CVGuard Canonical Pydantic v2 domain schemas.

Designed for air-gapped, zero-network runtime verification.
Models are immutable (frozen) by default to prevent accidental runtime tampering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """Normalized vulnerability or anomaly severity classification."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssetType(str, Enum):
    """Locked classification enum for computer vision evaluation assets."""

    SAMPLE = "sample"
    SOURCE = "source"
    MODEL = "model"
    INFERENCE_RECORD = "inference_record"
    BATCH = "batch"


class Disposition(str, Enum):
    """Triaged status and operational routing disposition."""

    ACCEPT = "accept"
    REVIEW = "review"
    QUARANTINE = "quarantine"


class Finding(BaseModel):
    """Atomic assurance record representing a single vision artifact evaluation result.

    Complies with CVGuard Phase 1 specification.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    finding_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the finding record (UUIDv4).",
    )
    asset_type: AssetType = Field(
        ...,
        description="Asset category enum: sample, source, model, inference_record, or batch.",
        examples=[AssetType.MODEL, AssetType.SAMPLE],
    )
    asset_ref: str = Field(
        ...,
        min_length=1,
        description="Deterministic local asset locator or cryptographic digest.",
        examples=["sha256:4a35b.../resnet50.onnx", "dataset/v2/shard_004.tar"],
    )
    detector: str = Field(
        ...,
        min_length=1,
        description="Identifier and version of the detector module that produced this finding.",
        examples=["cvguard.detector.model_integrity:v1.0"],
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Concise human-readable rationale summarizing the anomaly or failure.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="List of corroborating evidence pointers, metrics, or telemetry strings.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Statistical confidence score strictly constrained between 0.0 and 1.0.",
    )
    severity: Severity = Field(
        ...,
        description="Risk severity level: info, low, medium, high, or critical.",
    )
    disposition: Disposition = Field(
        ...,
        description="Target routing disposition: accept, review, or quarantine.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions made by the detector during evaluation.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known algorithmic limitations or boundary condition caveats.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO 8601 UTC timestamp of finding creation.",
    )


class SignedFinding(BaseModel):
    """Cryptographically signed finding envelope bound to the tamper-evident audit ledger.

    Explicitly separates immutable domain findings from cryptographic ledger metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding: Finding
    entry_hash: str
    prev_hash: str
    signature: str
    ledger_id: int | None = None  # set once persisted


class CoverageStatement(BaseModel):
    """Summary of inspection surface area and test boundary completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_assets_scanned: int = Field(..., ge=0)
    passed_count: int = Field(..., ge=0)
    flagged_count: int = Field(..., ge=0)
    skipped_count: int = Field(default=0, ge=0)
    scope_description: str = Field(
        default="Full offline plane inspection",
        description="Narrative scope of the coverage run.",
    )


class Report(BaseModel):
    """Comprehensive governance inspection container aggregating findings and provenance."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    report_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the report instance.",
    )
    title: str = Field(
        default="CVGuard Integrity Assurance Report",
        description="Human-readable title for the report.",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="List of all detected findings across evaluated planes.",
    )
    coverage: CoverageStatement = Field(
        ...,
        description="Coverage statement outlining evaluated surface area.",
    )
    reproducibility: dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic manifest of hashes, environment parameters, and detector configs.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO 8601 UTC timestamp when the report was sealed.",
    )
