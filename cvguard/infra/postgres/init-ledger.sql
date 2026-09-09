-- ==============================================================================
-- CVGuard Governance Spine: Audit Ledger Initialization Script
-- Phase 1: Tamper-Evident Hash Chain Audit Log & Append-Only Database Role
-- ==============================================================================

-- 1. Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    entry_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    signature TEXT NOT NULL
);

-- Index for entry lookups, chain validation, and verification queries
CREATE INDEX IF NOT EXISTS idx_audit_log_entry_hash ON audit_log (entry_hash);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log (created_at);

-- 2. Provision dedicated restricted role for append-only audit ledger operations
-- NOTE: In production, password is provided via secure secret management / environment.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'cvguard_ledger') THEN
        CREATE ROLE cvguard_ledger WITH LOGIN PASSWORD 'cvguard_ledger_secret_change_me_in_prod';
    END IF;
END
$$;

-- 3. Grant connection and schema usage
GRANT CONNECT ON DATABASE cvguard TO cvguard_ledger;
GRANT USAGE ON SCHEMA public TO cvguard_ledger;

-- 4. Grant STRICTLY INSERT and SELECT privileges on the audit_log table
GRANT SELECT, INSERT ON TABLE audit_log TO cvguard_ledger;
GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO cvguard_ledger;

-- 5. Explicitly REVOKE UPDATE, DELETE, and TRUNCATE privileges
-- This guarantees at the database storage engine layer that even if application code
-- or credentials were compromised, existing ledger history cannot be silently modified or deleted.
REVOKE UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLE audit_log FROM cvguard_ledger;
