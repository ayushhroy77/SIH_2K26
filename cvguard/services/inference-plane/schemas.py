"""Pydantic v2 domain schemas for CVGuard Inference Plane API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from merkle import MerkleProofStep


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = Field(default="ok", description="Operational status flag.")
    service: str = Field(default="inference-plane", description="Name of the reporting service.")
    version: str = Field(default="0.2.0", description="Service semantic version.")
    schemas_version: str = Field(..., description="Version of cvguard_schemas linked to runtime.")
    db_connected: bool = Field(default=True, description="Postgres connection readiness.")
    pending_records_in_buffer: int = Field(default=0, description="Records currently pending batch seal.")


class InferJsonRequest(BaseModel):
    """JSON payload for inference execution when not using multipart/form-data."""

    image_base64: str = Field(..., description="Base64-encoded raw image bytes.")
    model_id: str = Field(..., description="Deterministic model identity reference or weight digest.")
    config: dict[str, Any] | str = Field(
        default_factory=dict,
        description="Preprocessing, normalization, and runtime inference parameters.",
    )


class InferResponse(BaseModel):
    """Response returned upon successful inference and cryptographic binding computation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., description="Unique UUIDv4 identifier of the inference record.")
    model_id: str = Field(..., description="Model identifier reference.")
    sequence_number: int = Field(
        ...,
        ge=1,
        description="Strictly monotonic, atomically enforced sequence number preventing replay.",
    )
    input_hash: str = Field(..., description="SHA-256 hex digest of the raw input image bytes.")
    config_hash: str = Field(..., description="SHA-256 hex digest of the canonical JSON config.")
    output_hash: str = Field(..., description="SHA-256 hex digest of the canonical JSON output.")
    record_hash: str = Field(
        ...,
        description="Cryptographic leaf binding hash sealing sequence_number, model_id, inputs, config, and output.",
    )
    output: dict[str, Any] = Field(..., description="Inference prediction output dictionary.")
    batch_id: str | None = Field(
        default=None,
        description="Assigned batch ID if immediately flushed into a Merkle batch.",
    )
    status: str = Field(
        default="pending_batch",
        description="Status flag: 'pending_batch' or 'sealed'.",
    )


class VerifyResponse(BaseModel):
    """Comprehensive verification result for an inference record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., description="Inference record UUID.")
    model_id: str = Field(..., description="Model identifier reference.")
    sequence_number: int = Field(..., description="Sequence number of the record.")
    input_hash: str = Field(..., description="Computed/stored input hash.")
    config_hash: str = Field(..., description="Computed/stored config hash.")
    output_hash: str = Field(..., description="Computed/stored output hash.")
    record_hash: str = Field(..., description="Cryptographic record binding hash.")
    batch_id: str | None = Field(default=None, description="Batch identifier containing this record.")
    merkle_root: str | None = Field(default=None, description="Merkle root of the containing batch.")
    merkle_proof: list[MerkleProofStep] = Field(
        default_factory=list,
        description="Sibling hash path required to prove leaf inclusion in the Merkle root.",
    )
    proof_valid: bool = Field(
        ...,
        description="True if the Merkle proof mathematically computes to the batch Merkle root.",
    )
    governance_sealed: bool = Field(
        ...,
        description="True if the batch is sealed with a valid signature and recorded in the Governance Spine.",
    )
    ledger_id: int | None = Field(
        default=None,
        description="PostgreSQL primary key in Governance Spine audit ledger.",
    )
    governance_signature: str | None = Field(
        default=None,
        description="Hex-encoded Ed25519 digital signature from Governance Spine.",
    )
    tampered: bool = Field(
        ...,
        description="True if any post-hoc alteration or integrity violation was detected.",
    )
    status: str = Field(
        ...,
        description="Overall status summary: 'VERIFIED', 'PENDING_BATCH', or 'TAMPERED'.",
    )
    reason: str | None = Field(default=None, description="Explanation if tampered or pending.")


class BatchFlushResponse(BaseModel):
    """Response returned when an inference record batch is sealed."""

    batch_id: str = Field(..., description="Unique batch identifier.")
    size: int = Field(..., description="Total number of records bundled in this batch.")
    first_sequence: int = Field(..., description="Sequence number of the first record.")
    last_sequence: int = Field(..., description="Sequence number of the last record.")
    merkle_root: str = Field(..., description="64-character SHA-256 Merkle root.")
    finding_id: str = Field(..., description="Governance Finding UUID.")
    ledger_id: int | None = Field(default=None, description="Governance ledger entry ID.")
    signature: str | None = Field(default=None, description="Governance Ed25519 signature.")
    status: str = Field(default="sealed", description="Batch status.")
