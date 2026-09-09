# CVGuard: Offline Computer Vision Integrity Assurance Platform

CVGuard is a production-grade, offline-first integrity assurance platform engineered to rigorously validate computer vision models, datasets, and inference runtimes in high-security, air-gapped environments. The system orchestrates four independent evaluation planes (Data, Model, Inference, and Drift) that feed a unified, tamper-evident Governance spine to produce reproducible, auditable assurance reports. Every micro-service is stateless, zero-trust ready, and designed without any runtime internet dependency.

> **Offline Architecture Notice:**  
> This system is designed to run fully offline/air-gapped in production; local dev uses standard package registries for convenience only.

> **Project Phase Notice:**  
> This repository is currently at **Phase 1 of a 9-phase build plan: The Governance Spine**. This phase establishes the cryptographic Ed25519 signer, the tamper-evident PostgreSQL hash-chained audit ledger, and the security verification test suite. No detector logic or ML models exist yet; all planes feed findings into this immutable assurance spine.

> **Sandbox Generation Notice & Verification Mandate:**  
> This implementation was generated in a sandbox without execution capability and MUST be verified locally before being considered complete. Do NOT assume container builds or test runs have occurred. Execute the verification commands detailed below in a real Linux/WSL2 Docker and Python 3.12 environment.

---

## Governance Spine: Cryptographic Architecture & Threat Model

The Governance Spine (`services/governance`) provides the cryptographic backbone of CVGuard. It guarantees that any assurance finding emitted by an evaluation plane is signed, immutably recorded into a hash chain, and mathematically verifiable by external air-gapped auditors.

### 1. Hash-Chain Cryptographic Construction
- **Envelope Model (`SignedFinding`)**: To preserve immutability, `Finding` remains a frozen, content-only Pydantic model (`frozen=True`, `extra="forbid"`). The database and API wrap findings inside a `SignedFinding` envelope containing:
  - `ledger_id`: Monotonically increasing database primary key (`BIGSERIAL`)
  - `entry_hash`: Deterministic SHA256 digest of `prev_hash || canonical_json(finding)`
  - `prev_hash`: The `entry_hash` of the immediately preceding record (or `0` * 64 for genesis)
  - `signature`: Ed25519 detached signature sealing the canonical JSON serialization of `{"entry_hash": ..., "prev_hash": ...}`
- **Single Source of Truth for Signing**: The Ed25519 signer signs strictly `canonical_chain_payload(entry_hash, prev_hash)` in lexicographically sorted, whitespace-free JSON. This cryptographically binds the finding payload's hash directly to its position in the chronological chain.

### 2. Threat Model: Protections & Current Phase Limitations

| Threat / Attack Vector | Current Status in Phase 1 | Mitigation Mechanism |
| :--- | :--- | :--- |
| **Payload Tampering** (altering confidence, severity, or findings in the DB) | **PROTECTED** | Detected immediately. Any modification to a finding invalidates that row's `entry_hash`, breaking the SHA256 chain and failing Ed25519 signature verification. |
| **Row Deletion / Record Omission** (silently dropping a critical finding) | **PROTECTED** | Detected immediately. Deleting row $N$ causes row $N+1$'s `prev_hash` to point to a nonexistent hash, breaking chain continuity during `verify_chain()`. |
| **Signature Forgery** (injecting findings without the governance private key) | **PROTECTED** | Detected immediately. Ed25519 digital signatures verified with the official public key reject signatures generated from any unauthorized key. |
| **Direct DB Modification by Application Code** | **PROTECTED** | The `governance` service connects via the dedicated `cvguard_ledger` role, which is explicitly denied `UPDATE`, `DELETE`, and `TRUNCATE` at the PostgreSQL engine level. |
| **Compromised Host Filesystem / Key Extraction** | **NOT YET PROTECTED (Phase 7)** | Currently, `LocalFileSigner` persists the private key on the local filesystem (with `0600` permissions in a gitignored `secrets/` directory). Phase 7 replaces this with Hardware Security Modules (HSMs), Vault Transit, and TEE (Confidential Computing enclaves). |
| **History Rewriting by Full DB Superuser** | **PARTIALLY PROTECTED** | A rogue PostgreSQL superuser could rewrite all hashes and re-sign the entire ledger *if and only if* they also steal the Ed25519 private key. Phase 7 and Phase 8 introduce external timestamping, multi-party consensus, and append-only WORM media to eliminate this. |

### 3. Role of the Restricted Database User (`cvguard_ledger`)
Security in depth requires that the application layer itself cannot rewrite history even in the event of an SQL injection or code execution vulnerability:
- The database schema provisions a dedicated `cvguard_ledger` user.
- Privileges are strictly limited to `INSERT` and `SELECT` on `audit_log`, plus sequence usage on `audit_log_id_seq`.
- `UPDATE`, `DELETE`, `TRUNCATE`, `REFERENCES`, and `TRIGGER` permissions are explicitly revoked.
- Any attempt to modify or delete rows produces an uncatchable `42501 InsufficientPrivilege` error from PostgreSQL's storage engine.

### 4. How to Verify the Audit Ledger Offline (Conceptual Auditor Guide)
An independent auditor in an air-gapped facility needs only:
1. The **32-byte Ed25519 Public Key** (retrieved once during setup via `GET /health` or from out-of-band policy manifests).
2. A raw database dump (`SELECT id, entry_hash, prev_hash, payload, signature FROM audit_log ORDER BY id ASC`).

**Verification Steps:**
1. **Initialize State**: Set `expected_prev_hash` to the fixed 64-character genesis hash (`0000000000000000000000000000000000000000000000000000000000000000`).
2. **Iterate Entries in Sequential Order ($id = 1, 2, \dots, N$)**:
   - **Step A (Chain Continuity)**: Assert that the entry's `prev_hash` exactly equals `expected_prev_hash`. If it does not match, a row was deleted, reordered, or spliced.
   - **Step B (Payload Integrity)**: Canonicalize the `payload` JSON (lexicographical key sorting, compact delimiters, UTF-8 encoding) and compute `SHA256(prev_hash || canonical_payload)`. Assert that this digest matches `entry_hash` exactly. If it does not match, the payload content was altered.
   - **Step C (Cryptographic Authenticity)**: Construct the canonical envelope `{"entry_hash": entry_hash, "prev_hash": prev_hash}` and verify the row's `signature` against this payload using the Ed25519 public key. If verification fails, the record was forged.
   - **Step D (Advance State)**: Set `expected_prev_hash = entry_hash` and proceed to the next record.
3. If all records pass steps A through C without error, the entire historical ledger is verified mathematically clean and authentic.

---

## Architecture Overview

```
                           +----------------------+
                           |       Frontend       |
                           |  (React/TS/Tailwind) |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |   Gateway Service    | :8000
                           +----------+-----------+
                                      |
           +--------------------------+--------------------------+
           |                          |                          |
           v                          v                          v
+--------------------+      +--------------------+      +--------------------+
|     Data Plane     | :8001|    Model Plane     | :8002|  Inference Plane   | :8003
+----------+---------+      +----------+---------+      +----------+---------+
           |                           |                           |
           +---------------------------+---------------------------+
           |                           |
           v                           v
+--------------------+      +--------------------+
|    Drift Plane     | :8004|     Governance     | :8005
+--------------------+      +----------+---------+
                                       |
                   +-------------------+-------------------+
                   |                   |                   |
                   v                   v                   v
              PostgreSQL 16         Redis 7              MinIO
              (:5432)               (:6379)         (:9000 / :9001)
```

### Shared Domain Package: `libs/schemas` (`cvguard-schemas`)
All six plane services share canonical Pydantic v2 domain schemas exported by `cvguard_schemas`:
- **`Finding`**: Atomic unit of assurance reporting comprising `finding_id`, `asset_type`, `asset_ref`, `detector`, `reason`, `evidence`, `confidence` [0.0 - 1.0], `severity` (`info` | `low` | `medium` | `high` | `critical`), `disposition` (`accept` | `review` | `quarantine`), `assumptions`, `limitations`, `created_at`, and `signature`.
- **`Report`**: Comprehensive governance container aggregating findings, a `CoverageStatement`, and an immutable `reproducibility` manifest of artifact hashes, environment parameters, and detector configurations.

---

## Local Verification Commands

To verify this scaffold locally on your Linux or WSL2 workstation:

### 1. Environment Configuration
Copy the infrastructure environment template:
```bash
cp infra/.env.example infra/.env
```

### 2. Run All Services with Docker Compose
Build and launch all six FastAPI services alongside PostgreSQL 16, Redis 7, and MinIO:
```bash
make up
```
*Or directly via Docker Compose:*
```bash
docker compose -f infra/docker-compose.yml --env-file infra/.env up --build
```

Verify service health probes:
```bash
curl -f http://localhost:8000/health   # Gateway
curl -f http://localhost:8001/health   # Data Plane
curl -f http://localhost:8002/health   # Model Plane
curl -f http://localhost:8003/health   # Inference Plane
curl -f http://localhost:8004/health   # Drift Plane
curl -f http://localhost:8005/health   # Governance
```

### 3. Run Static Analysis and Type Checking
Validate PEP 8 formatting and mypy strict typing across all packages:
```bash
make lint
```

### 4. Verify Shared Schemas Import Cleanliness
Check that `cvguard-schemas` installs and imports without circular dependencies across all services:
```bash
make schema-check
```

### 5. Execute Test Suite
Run pytest across all schema serialization, bounds validation, and endpoint tests:
```bash
make test
```

### 6. Tear Down Containers
Stop all background containers and remove network bridges:
```bash
make down
```

---

## Port Allocations

| Service / Container | Host Port | Protocol | Purpose |
|---------------------|-----------|----------|---------|
| `gateway`           | 8000      | HTTP     | External REST API & Router |
| `data-plane`        | 8001      | HTTP     | Dataset Validation Plane |
| `model-plane`       | 8002      | HTTP     | Model Weights & Checkpoints Plane |
| `inference-plane`   | 8003      | HTTP     | Runtime Inference Integrity Plane |
| `drift-plane`       | 8004      | HTTP     | Statistical Distribution Drift Plane |
| `governance`        | 8005      | HTTP     | Governance Spine & Report Generator |
| `postgres`          | 5432      | TCP      | Relational audit log (PG 16) |
| `redis`             | 6379      | TCP      | Stateless message cache (Redis 7) |
| `minio` (API)       | 9000      | S3/HTTP  | Air-gapped object store |
| `minio` (Console)   | 9001      | HTTP     | MinIO Web Console |
