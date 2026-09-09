# CVGuard: Offline Computer Vision Integrity Assurance Platform

CVGuard is a production-grade, offline-first integrity assurance platform engineered to rigorously validate computer vision models, datasets, and inference runtimes in high-security, air-gapped environments. The system orchestrates four independent evaluation planes (Data, Model, Inference, and Drift) that feed a unified, tamper-evident Governance spine to produce reproducible, auditable assurance reports. Every micro-service is stateless, zero-trust ready, and designed without any runtime internet dependency.

> **Offline Architecture Notice:**  
> This system is designed to run fully offline/air-gapped in production; local dev uses standard package registries for convenience only.

> **Project Phase Notice:**  
> This repository represents **Phase 0 of a 9-phase build plan**. No detection logic, machine learning models, or domain-specific heuristic engines exist yet. This phase establishes the monorepo architecture, Docker containerization, canonical Pydantic v2 schemas, infrastructure datastores, and development verification tooling.

> **Sandbox Generation Notice & Verification Mandate:**  
> This scaffold was generated in a sandbox without execution capability and MUST be verified locally before being considered complete. Do NOT assume container builds or test runs have occurred. Execute the verification commands detailed below in a real Linux/WSL2 Docker and Python 3.12 environment.

---

## Local Verification Commands

To verify this scaffold locally on your Linux or WSL2 workstation:

### 1. Environment Configuration
Copy the infrastructure environment template:
```bash
cp cvguard/infra/.env.example cvguard/infra/.env
```

### 2. Run All Services with Docker Compose
Build and launch all six FastAPI services alongside PostgreSQL 16, Redis 7, and MinIO:
```bash
make up
```
*Or directly via Docker Compose from within the cvguard directory:*
```bash
cd cvguard && make up
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
```bash
make lint
```

### 4. Verify Shared Schemas Import Cleanliness
```bash
make schema-check
```

### 5. Execute Test Suite
```bash
make test
```

### 6. Tear Down Containers
```bash
make down
```

---

## Port Allocations

| Service / Container | Host Port | Protocol | Purpose |
|---------------------|-----------|----------|---------|
| `gateway`           | 8000      | HTTP     | External REST API & Router |
| `data-plane`        | 8001      | HTTP     | Dataset Validation Plane |
| `model-plane`       | 8002      | HTTP     | Model Checkpoints & Weights Plane |
| `inference-plane`   | 8003      | HTTP     | Runtime Inference Integrity Plane |
| `drift-plane`       | 8004      | HTTP     | Covariate Shift & Drift Plane |
| `governance`        | 8005      | HTTP     | Governance Spine & Report Generator |
| `postgres`          | 5432      | TCP      | Relational audit log (PG 16) |
| `redis`             | 6379      | TCP      | Stateless message cache (Redis 7) |
| `minio` (API)       | 9000      | S3/HTTP  | Air-gapped object store |
| `minio` (Console)   | 9001      | HTTP     | MinIO Web Console |
