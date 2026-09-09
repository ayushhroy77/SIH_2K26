-- ==============================================================================
-- CVGuard Inference Plane: Sequence Numbers & Cryptographic Binding Records
-- Phase 5: Replay-Resistant Atomic Sequences & Merkle Batch Storage
-- ==============================================================================

-- 1. Atomic sequence counters per model identity
CREATE TABLE IF NOT EXISTS inference_sequences (
    model_id TEXT PRIMARY KEY,
    current_sequence BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- 2. Immutable inference records with strict (model_id, monotonic_sequence_no) uniqueness
CREATE TABLE IF NOT EXISTS inference_records (
    record_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    monotonic_sequence_no BIGINT NOT NULL,
    sequence_number BIGINT NOT NULL,
    input_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    nonce TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    config_json JSONB NOT NULL,
    output_json JSONB NOT NULL,
    batch_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_model_sequence UNIQUE (model_id, monotonic_sequence_no)
);

CREATE INDEX IF NOT EXISTS idx_records_model_seq ON inference_records (model_id, monotonic_sequence_no);
CREATE INDEX IF NOT EXISTS idx_records_batch_id ON inference_records (batch_id);
CREATE INDEX IF NOT EXISTS idx_records_record_hash ON inference_records (record_hash);
CREATE INDEX IF NOT EXISTS idx_records_created_at ON inference_records (created_at);

-- 3. Batches sealed via Merkle root and signed findings in Governance Spine
CREATE TABLE IF NOT EXISTS inference_batches (
    batch_id TEXT PRIMARY KEY,
    merkle_root TEXT NOT NULL,
    size INT NOT NULL,
    first_sequence BIGINT NOT NULL,
    last_sequence BIGINT NOT NULL,
    model_id TEXT,
    finding_id TEXT NOT NULL,
    ledger_id BIGINT,
    signed_finding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS idx_batches_merkle_root ON inference_batches (merkle_root);
CREATE INDEX IF NOT EXISTS idx_batches_created_at ON inference_batches (created_at);
