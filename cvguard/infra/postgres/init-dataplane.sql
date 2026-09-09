-- ==============================================================================
-- CVGuard Data Plane: Image Catalog & Detection Traceability Initialization
-- Phase 2: Perceptual Hashing & Near-Duplicate Detection Storage
-- ==============================================================================

-- 1. Image metadata repository
CREATE TABLE IF NOT EXISTS images (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    minio_key TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    contributor_id TEXT NOT NULL,
    phash TEXT NOT NULL,
    label TEXT,
    dataset_id TEXT DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images (sha256);
CREATE INDEX IF NOT EXISTS idx_images_contributor_id ON images (contributor_id);
CREATE INDEX IF NOT EXISTS idx_images_uploaded_at ON images (uploaded_at);
CREATE INDEX IF NOT EXISTS idx_images_label ON images (label);

-- 2. Reference distributions for Out-of-Distribution (OOD) detection
CREATE TABLE IF NOT EXISTS reference_distributions (
    id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    centroid JSONB NOT NULL,
    covariance_inv JSONB NOT NULL,
    num_samples INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_dataset_class UNIQUE (dataset_id, class_name)
);

CREATE INDEX IF NOT EXISTS idx_ref_dist_dataset_class ON reference_distributions (dataset_id, class_name);

-- 3. Traceability record linking local detections to the Governance Spine ledger
CREATE TABLE IF NOT EXISTS detections (
    id BIGSERIAL PRIMARY KEY,
    finding_id TEXT NOT NULL,
    ledger_id BIGINT,
    asset_type TEXT NOT NULL,
    contributor_id TEXT,
    evidence JSONB NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS idx_detections_finding_id ON detections (finding_id);
CREATE INDEX IF NOT EXISTS idx_detections_ledger_id ON detections (ledger_id);
