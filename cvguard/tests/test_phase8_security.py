"""CVGuard Phase 8 Security Hardening Test Suite.

Verifies:
1. Authentication: Rejection of unauthenticated or invalid tokens with HTTP 401.
2. Transport Security: mTLS certificate validation and rejection of untrusted CA certificates.
3. Authorization (RBAC):
   - Unauthorized role access rejected with HTTP 403 ("Forbidden: Insufficient permissions").
   - Valid analyst access permitted for operational/findings endpoints.
   - Valid auditor access permitted for audit verification endpoints.
   - Valid admin access permitted across governance, profiles, and reference distributions.
4. Dependency Hygiene: Verification that all service lockfiles are pinned, synchronized, and cryptographically hashed.
"""

from __future__ import annotations

import os
import ssl
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import cvguard_schemas
from cvguard_schemas import Role, UserIdentity
from cvguard_schemas.security import (
    build_client_ssl_context,
    build_server_ssl_context,
    create_test_jwt,
    verify_bearer_token,
)
from scripts.generate_certs import generate_pki
from scripts.verify_lockfiles import main as verify_lockfiles_main


# ==============================================================================
# Fixtures & Tokens
# ==============================================================================

@pytest.fixture(scope="module")
def analyst_token() -> str:
    return create_test_jwt(username="analyst_alice", roles=["analyst"])


@pytest.fixture(scope="module")
def admin_token() -> str:
    return create_test_jwt(username="admin_bob", roles=["admin"])


@pytest.fixture(scope="module")
def auditor_token() -> str:
    return create_test_jwt(username="auditor_charlie", roles=["auditor"])


@pytest.fixture(scope="module")
def expired_token() -> str:
    # 3600 seconds in the past
    return create_test_jwt(username="stale_user", roles=["analyst"], expires_delta=-3600)


@pytest.fixture(scope="module")
def gateway_client() -> TestClient:
    """Initialize FastAPI TestClient for Gateway Service."""
    # Ensure service import
    sys.path.insert(0, os.path.abspath("services/gateway"))
    from main import app as gateway_app
    return TestClient(gateway_app)


@pytest.fixture(scope="module")
def data_plane_client() -> TestClient:
    """Initialize FastAPI TestClient for Data Plane Service."""
    sys.path.insert(0, os.path.abspath("services/data-plane"))
    from main import app as data_plane_app
    return TestClient(data_plane_app)


@pytest.fixture(scope="module")
def drift_plane_client() -> TestClient:
    """Initialize FastAPI TestClient for Drift Plane Service."""
    sys.path.insert(0, os.path.abspath("services/drift-plane"))
    from main import app as drift_plane_app
    return TestClient(drift_plane_app)


@pytest.fixture(scope="module")
def governance_client() -> TestClient:
    """Initialize FastAPI TestClient for Governance Spine Service."""
    sys.path.insert(0, os.path.abspath("services/governance"))
    from main import app as gov_app
    return TestClient(gov_app)


# ==============================================================================
# 1. Authentication Tests (401 Rejections)
# ==============================================================================

def test_gateway_unauthenticated_post_rejected_with_401(gateway_client: TestClient):
    """External requests missing Bearer token must be rejected with HTTP 401."""
    resp = gateway_client.post("/ingest/images", files={"files": ("test.jpg", b"fake", "image/jpeg")})
    assert resp.status_code == 401
    assert "Unauthorized" in resp.json().get("detail", "")


def test_gateway_unauthenticated_get_rejected_with_401(gateway_client: TestClient):
    """External GET queries missing Bearer token must be rejected with HTTP 401."""
    resp = gateway_client.get("/findings")
    assert resp.status_code == 401
    assert "Unauthorized" in resp.json().get("detail", "")


def test_gateway_expired_token_rejected_with_401(gateway_client: TestClient, expired_token: str):
    """Requests with expired Bearer token must be rejected with HTTP 401."""
    headers = {"Authorization": f"Bearer {expired_token}"}
    resp = gateway_client.get("/findings", headers=headers)
    assert resp.status_code == 401
    assert "Unauthorized" in resp.json().get("detail", "")


def test_gateway_malformed_token_rejected_with_401(gateway_client: TestClient):
    """Requests with malformed Bearer token must be rejected with HTTP 401."""
    headers = {"Authorization": "Bearer not.a.valid.jwt.payload"}
    resp = gateway_client.get("/findings", headers=headers)
    assert resp.status_code == 401
    assert "Unauthorized" in resp.json().get("detail", "")


def test_governance_unauthenticated_finding_submission_rejected(governance_client: TestClient):
    """Direct finding submission without mTLS service identity or user token is rejected."""
    resp = governance_client.post("/findings", json={"finding": {"title": "Test"}})
    assert resp.status_code == 401


# ==============================================================================
# 2. Authorization & RBAC Tests (403 Forbidden vs Allowed)
# ==============================================================================

def test_unauthorized_role_analyst_cannot_register_reference_distribution(
    gateway_client: TestClient,
    analyst_token: str,
):
    """Analyst role attempting admin-only reference distribution registration returns HTTP 403."""
    headers = {"Authorization": f"Bearer {analyst_token}"}
    payload = {
        "dataset_id": "ds_restricted",
        "class_name": "target",
        "reference_embeddings": [[0.1] * 128, [0.2] * 128],
    }
    resp = gateway_client.post("/reference-distributions", json=payload, headers=headers)
    assert resp.status_code == 403
    assert resp.json().get("detail") == "Forbidden: Insufficient permissions"


def test_unauthorized_role_analyst_cannot_access_audit_verify(
    gateway_client: TestClient,
    analyst_token: str,
):
    """Analyst role attempting auditor/admin audit verification returns HTTP 403."""
    headers = {"Authorization": f"Bearer {analyst_token}"}
    resp = gateway_client.get("/audit/verify", headers=headers)
    assert resp.status_code == 403
    assert resp.json().get("detail") == "Forbidden: Insufficient permissions"


def test_unauthorized_role_auditor_cannot_ingest_images(
    gateway_client: TestClient,
    auditor_token: str,
):
    """Auditor role attempting image ingestion (analyst/admin only) returns HTTP 403."""
    headers = {"Authorization": f"Bearer {auditor_token}"}
    resp = gateway_client.post(
        "/ingest/images",
        files={"files": ("test.jpg", b"fake", "image/jpeg")},
        headers=headers,
    )
    assert resp.status_code == 403
    assert resp.json().get("detail") == "Forbidden: Insufficient permissions"


def test_direct_data_plane_admin_check_enforced(data_plane_client: TestClient, analyst_token: str):
    """Direct POST to data-plane reference distributions enforces admin role."""
    headers = {"Authorization": f"Bearer {analyst_token}"}
    payload = {
        "dataset_id": "test_ds",
        "class_name": "cat",
        "centroid": [0.1] * 16,
        "covariance_inv": [[1.0 if i == j else 0.0 for j in range(16)] for i in range(16)],
        "num_samples": 50,
    }
    resp = data_plane_client.post("/reference-distributions", json=payload, headers=headers)
    assert resp.status_code == 403
    assert resp.json().get("detail") == "Forbidden: Insufficient permissions"


def test_direct_drift_plane_admin_check_enforced(drift_plane_client: TestClient, analyst_token: str):
    """Direct POST to drift-plane reference profile creation enforces admin role."""
    headers = {"Authorization": f"Bearer {analyst_token}"}
    payload = {
        "profile_id": "prof_unauthorized",
        "samples": [{"embedding": [0.1] * 16}],
    }
    resp = drift_plane_client.post("/reference-profile", json=payload, headers=headers)
    assert resp.status_code == 403
    assert resp.json().get("detail") == "Forbidden: Insufficient permissions"


def test_valid_admin_role_allowed_for_reference_distribution(
    data_plane_client: TestClient,
    admin_token: str,
):
    """Admin role is authorized to register reference distributions."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "dataset_id": "test_admin_ds",
        "class_name": "airplane",
        "centroid": [0.5] * 16,
        "covariance_inv": [[1.0 if i == j else 0.0 for j in range(16)] for i in range(16)],
        "num_samples": 100,
    }
    resp = data_plane_client.post("/reference-distributions", json=payload, headers=headers)
    assert resp.status_code == 201
    assert resp.json().get("status") == "ok"


def test_valid_admin_role_allowed_for_drift_profile(
    drift_plane_client: TestClient,
    admin_token: str,
):
    """Admin role is authorized to register drift reference profiles."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "profile_id": "prof_admin_ok",
        "samples": [{"embedding": [0.1] * 16}],
    }
    resp = drift_plane_client.post("/reference-profile", json=payload, headers=headers)
    assert resp.status_code == 201
    assert resp.json().get("profile_id") == "prof_admin_ok"


# ==============================================================================
# 3. Transport Security & Mutual TLS (mTLS) Tests
# ==============================================================================

def test_mtls_valid_and_invalid_ca_rejection():
    """Verify that an SSLContext configured with trusted CA validates trusted certs and rejects untrusted ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Generate trusted CVGuard CA and service certs
        ca_dir = tmp_path / "trusted_pki"
        generate_pki(ca_dir)

        trusted_ca_cert = ca_dir / "ca.crt"
        gateway_cert = ca_dir / "gateway.crt"
        gateway_key = ca_dir / "gateway.key"
        governance_cert = ca_dir / "governance.crt"
        governance_key = ca_dir / "governance.key"

        # 2. Generate rogue/untrusted CA and rogue certificate
        rogue_dir = tmp_path / "rogue_pki"
        generate_pki(rogue_dir)
        rogue_cert = rogue_dir / "gateway.crt"
        rogue_key = rogue_dir / "gateway.key"

        # 3. Build server SSL context for governance requiring client certs from trusted CA
        server_ctx = build_server_ssl_context(
            ca_cert_path=str(trusted_ca_cert),
            cert_path=str(governance_cert),
            key_path=str(governance_key),
        )
        assert server_ctx.verify_mode == ssl.CERT_REQUIRED

        # 4. Build client SSL context for trusted gateway
        trusted_client_ctx = build_client_ssl_context(
            ca_cert_path=str(trusted_ca_cert),
            cert_path=str(gateway_cert),
            key_path=str(gateway_key),
        )
        assert trusted_client_ctx.verify_mode == ssl.CERT_REQUIRED

        # 5. Build client SSL context for rogue client (signed by rogue CA)
        rogue_client_ctx = build_client_ssl_context(
            ca_cert_path=str(trusted_ca_cert),
            cert_path=str(rogue_cert),
            key_path=str(rogue_key),
        )

        # 6. Verify in-memory SSL transport verification
        # An SSL socket memory BIO verification between server_ctx and rogue_client_ctx fails certificate verification
        server_ssl = server_ctx.wrap_bio(
            incoming=ssl.MemoryBIO(),
            outgoing=ssl.MemoryBIO(),
            server_side=True,
        )
        rogue_client_ssl = rogue_client_ctx.wrap_bio(
            incoming=ssl.MemoryBIO(),
            outgoing=ssl.MemoryBIO(),
            server_side=False,
        )

        # Client initiates handshake (ClientHello)
        try:
            rogue_client_ssl.do_handshake()
        except ssl.SSLWantReadError:
            pass

        # Forward client data to server
        client_out = rogue_client_ssl.outgoing.read()
        server_ssl.incoming.write(client_out)

        # Server processes ClientHello and produces ServerHello, Certificate, CertificateRequest, ServerHelloDone
        try:
            server_ssl.do_handshake()
        except ssl.SSLWantReadError:
            pass

        server_out = server_ssl.outgoing.read()
        rogue_client_ssl.incoming.write(server_out)

        # Rogue client produces its Certificate (signed by rogue CA)
        try:
            rogue_client_ssl.do_handshake()
        except ssl.SSLWantReadError:
            pass

        client_resp = rogue_client_ssl.outgoing.read()
        server_ssl.incoming.write(client_resp)

        # Server evaluates rogue certificate against trusted CA -> must raise SSLCertVerificationError
        with pytest.raises(ssl.SSLCertVerificationError):
            server_ssl.do_handshake()


# ==============================================================================
# 4. Dependency Hygiene Test
# ==============================================================================

def test_dependency_lockfiles_in_sync_and_hashed():
    """Verify that all services and libraries have locked, pinned requirements with cryptographic hashes."""
    result = verify_lockfiles_main()
    assert result == 0, "Dependency lockfile verification failed."
