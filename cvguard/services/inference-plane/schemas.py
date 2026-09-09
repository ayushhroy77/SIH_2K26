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
    model_id: str = Field(..., description="Model identifier reference or weight digest.")
    config: dict[str, Any] | str = Field(
        default_factory=dict,
        description="Preprocessing, normalization, and runtime inference parameters.",
    )


class InferResponse(BaseModel):
    """Response returned upon successful inference and cryptographic binding computation."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    record_id: str = Field(..., description="Unique UUIDv4 identifier of the inference record.")
    model_id: str = Field(..., description="Model weight digest.")
    monotonic_sequence_no: int = Field(
        ...,
        ge=1,
        description="Strictly monotonic, atomically enforced sequence number preventing replay.",
    )
    sequence_number: int = Field(
        ...,
        ge=1,
        description="Alias for monotonic_sequence_no for compatibility.",
    )
    input_hash: str = Field(..., description="SHA-256 hex digest of the raw input image bytes.")
    config_hash: str = Field(..., description="SHA-256 hex digest of the canonical JSON config.")
    output_hash: str = Field(..., description="SHA-256 hex digest of the canonical JSON output.")
    record_hash: str = Field(
        ...,
        description="Cryptographic leaf binding hash sealing sequence_no, model_id, inputs, config, nonce, and output.",
    )
    nonce: str = Field(..., description="Unique cryptographic entropy nonce.")
    timestamp: str = Field(..., description="ISO-8601 or float timestamp.")
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
    """Verification result for an inference record.

    Per Phase 5 task:
    GET /verify/{record_id} returns {valid: bool, reason: str}.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    valid: bool = Field(
        ...,
        description="True if record binding and Merkle proof against published root are valid; False otherwise.",
    )
    reason: str = Field(
        ...,
        description="Detailed verification outcome or tamper explanation.",
    )
    record_id: str = Field(..., description="Inference record UUID.")
    model_id: str = Field(..., description="Model weight digest.")
    monotonic_sequence_no: int = Field(..., description="Monotonic sequence number of the record.")
    sequence_number: int = Field(default=0, description="Alias for monotonic_sequence_no.")
    input_hash: str = Field(..., description="Computed/stored input hash.")
    config_hash: str = Field(..., description="Computed/stored config hash.")
    output_hash: str = Field(..., description="Computed/stored output hash.")
    record_hash: str = Field(..., description="Cryptographic record binding hash.")
    nonce: str | None = Field(default=None, description="Cryptographic nonce.")
    timestamp: str | None = Field(default=None, description="Record creation timestamp.")
    batch_id: str | None = Field(default=None, description="Batch identifier containing this record.")
    merkle_root: str | None = Field(default=None, description="Merkle root of the containing batch.")
    merkle_proof: list[MerkleProofStep] = Field(
        default_factory=list,
        description="Sibling hash path required to prove leaf inclusion in the Merkle root.",
    )
    proof_valid: bool = Field(
        default=False,
        description="True if the Merkle proof mathematically computes to the batch Merkle root.",
    )
    governance_sealed: bool = Field(
        default=False,
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
        default=False,
        description="True if any post-hoc alteration or integrity violation was detected.",
    )
    status: str = Field(
        default="PENDING_BATCH",
        description="Status: 'VERIFIED', 'PENDING_BATCH', or 'TAMPERED'.",
    )


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
