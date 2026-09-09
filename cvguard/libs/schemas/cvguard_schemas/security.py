"""CVGuard Security, Mutual TLS (mTLS), and Role-Based Access Control (RBAC) Module.

Phase 8 Hardening:
- Shared mTLS SSLContext factories for strict service-to-service authentication.
- Cryptographic JWT parsing, claim verification, and role extraction for Keycloak OIDC.
- Role-Based Access Control policies for 'analyst', 'admin', and 'auditor'.
- Internal service-to-service caller validation to prevent direct user fabrication of findings.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("cvguard.security")


class Role(str, Enum):
    """CVGuard RBAC Canonical Roles."""

    ANALYST = "analyst"
    ADMIN = "admin"
    AUDITOR = "auditor"


# Official microservice DNS identities for internal mTLS
INTERNAL_SERVICE_IDENTITIES = frozenset(
    [
        "gateway",
        "data-plane",
        "model-plane",
        "inference-plane",
        "drift-plane",
        "governance",
    ]
)

# Services authorized to seal findings into governance
FINDING_PRODUCER_SERVICES = frozenset(
    [
        "data-plane",
        "model-plane",
        "inference-plane",
        "drift-plane",
    ]
)


@dataclass(frozen=True)
class UserIdentity:
    """Authenticated user context extracted from validated JWT bearer token."""

    user_id: str
    username: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    email: str | None = None

    def has_role(self, role: str | Role) -> bool:
        """Check if user holds a specific role."""
        target = role.value if isinstance(role, Role) else role
        return target in self.roles or Role.ADMIN.value in self.roles

    def has_any_role(self, roles: Sequence[str | Role]) -> bool:
        """Check if user holds at least one of the specified roles."""
        for r in roles:
            if self.has_role(r):
                return True
        return False


@dataclass(frozen=True)
class ServiceIdentity:
    """Authenticated microservice caller verified via mutual TLS client certificate."""

    service_name: str
    san: str
    is_internal: bool = True


# ==============================================================================
# Mutual TLS (mTLS) Context Factories
# ==============================================================================

def is_mtls_enabled() -> bool:
    """Check if mutual TLS is globally enabled via environment."""
    val = os.getenv("CVGUARD_MTLS_ENABLED", "false").strip().lower()
    return val in ("true", "1", "yes", "on")


def get_default_cert_paths() -> tuple[Path | None, Path | None, Path | None]:
    """Retrieve configured or default paths for CA cert, service cert, and service key."""
    ca = os.getenv("CVGUARD_CA_CERT_PATH")
    cert = os.getenv("CVGUARD_CERT_PATH")
    key = os.getenv("CVGUARD_KEY_PATH")

    ca_path = Path(ca) if ca and Path(ca).is_file() else None
    cert_path = Path(cert) if cert and Path(cert).is_file() else None
    key_path = Path(key) if key and Path(key).is_file() else None

    return ca_path, cert_path, key_path


def create_client_ssl_context(
    ca_cert_path: str | Path | None = None,
    client_cert_path: str | Path | None = None,
    client_key_path: str | Path | None = None,
) -> ssl.SSLContext | None:
    """Create an SSLContext for an outgoing mTLS client connection.

    Verifies the server's certificate against the CVGuard Root CA and presents
    the calling service's client certificate and private key.
    """
    ca_p, cert_p, key_p = get_default_cert_paths()
    ca_file = str(ca_cert_path or ca_p or "")
    cert_file = str(client_cert_path or cert_p or "")
    key_file = str(client_key_path or key_p or "")

    if not ca_file or not os.path.isfile(ca_file):
        if not is_mtls_enabled():
            return None
        raise FileNotFoundError(f"CVGuard Root CA certificate not found at: {ca_file}")

    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False  # Allows connection to internal Docker bridge service names

    if cert_file and key_file and os.path.isfile(cert_file) and os.path.isfile(key_file):
        ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    elif is_mtls_enabled():
        raise FileNotFoundError(
            f"Client certificate or key missing for mTLS: cert={cert_file}, key={key_file}"
        )

    return ctx


def create_server_ssl_context(
    ca_cert_path: str | Path | None = None,
    server_cert_path: str | Path | None = None,
    server_key_path: str | Path | None = None,
    require_client_cert: bool = True,
) -> ssl.SSLContext | None:
    """Create an SSLContext for an incoming mTLS service listener.

    Presents the service's certificate to the caller and requires a valid client
    certificate signed by the CVGuard Root CA (rejecting non-CA certs at handshake).
    """
    ca_p, cert_p, key_p = get_default_cert_paths()
    ca_file = str(ca_cert_path or ca_p or "")
    cert_file = str(server_cert_path or cert_p or "")
    key_file = str(server_key_path or key_p or "")

    if not cert_file or not key_file or not os.path.isfile(cert_file) or not os.path.isfile(key_file):
        if not is_mtls_enabled():
            return None
        raise FileNotFoundError(
            f"Server certificate or key missing for mTLS: cert={cert_file}, key={key_file}"
        )

    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)

    if require_client_cert:
        if not ca_file or not os.path.isfile(ca_file):
            raise FileNotFoundError(f"CVGuard Root CA certificate missing for mTLS client verification: {ca_file}")
        ctx.load_verify_locations(cafile=ca_file)
        ctx.verify_mode = ssl.CERT_REQUIRED

    return ctx


def get_httpx_mtls_kwargs(
    service_name: str | None = None,
) -> dict[str, Any]:
    """Build kwargs for httpx.AsyncClient or httpx.Client configuring mTLS."""
    ca_p, cert_p, key_p = get_default_cert_paths()

    if not is_mtls_enabled() and (not ca_p or not cert_p):
        return {}

    kwargs: dict[str, Any] = {}
    if ca_p and ca_p.is_file():
        kwargs["verify"] = str(ca_p)
    if cert_p and key_p and cert_p.is_file() and key_p.is_file():
        kwargs["cert"] = (str(cert_p), str(key_p))

    # Add default client identity header to assist in reverse-proxy routing
    if service_name:
        kwargs.setdefault("headers", {})["X-Client-Identity"] = service_name

    return kwargs


# ==============================================================================
# JWT Authentication & Verification (Keycloak OIDC)
# ==============================================================================

def parse_jwt_unverified(token: str) -> dict[str, Any]:
    """Parse the unverified payload of a standard JWT (for offline header/payload inspection)."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format: token must contain 3 dot-separated segments.")

    payload_b64 = parts[1]
    # Add padding if needed
    rem = len(payload_b64) % 4
    if rem > 0:
        payload_b64 += "=" * (4 - rem)

    try:
        decoded_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        payload: dict[str, Any] = json.loads(decoded_json)
        return payload
    except Exception as exc:
        raise ValueError(f"Failed to decode JWT payload: {exc}") from exc


def extract_user_from_payload(payload: dict[str, Any]) -> UserIdentity:
    """Extract standard user identity and roles from Keycloak JWT payload."""
    user_id = str(payload.get("sub", ""))
    username = str(payload.get("preferred_username", payload.get("username", user_id)))
    email = payload.get("email")

    roles: set[str] = set()

    # 1. Realm-level roles
    realm_access = payload.get("realm_access", {})
    if isinstance(realm_access, dict):
        for r in realm_access.get("roles", []):
            roles.add(str(r))

    # 2. Client-level roles (e.g., cvguard-gateway)
    resource_access = payload.get("resource_access", {})
    if isinstance(resource_access, dict):
        for client_data in resource_access.values():
            if isinstance(client_data, dict):
                for r in client_data.get("roles", []):
                    roles.add(str(r))

    # 3. Direct roles field (for custom or test tokens)
    direct_roles = payload.get("roles", [])
    if isinstance(direct_roles, list):
        for r in direct_roles:
            roles.add(str(r))

    return UserIdentity(
        user_id=user_id,
        username=username,
        roles=tuple(sorted(roles)),
        email=email,
    )


def validate_jwt_claims(payload: dict[str, Any], clock_skew_seconds: int = 60) -> None:
    """Validate standard temporal claims (exp, nbf)."""
    now = time.time()

    exp = payload.get("exp")
    if exp is not None:
        try:
            if float(exp) < (now - clock_skew_seconds):
                raise ValueError("Token has expired.")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid 'exp' claim in token: {exc}") from exc

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            if float(nbf) > (now + clock_skew_seconds):
                raise ValueError("Token is not yet valid (nbf).")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid 'nbf' claim in token: {exc}") from exc


def verify_bearer_token(token: str, public_key_pem: str | None = None) -> UserIdentity:
    """Verify a Bearer JWT token and return authenticated UserIdentity.

    If public_key_pem is supplied, validates RSA signature using cryptography.
    Validates expiration and temporal validity.
    """
    if not token or not isinstance(token, str):
        raise ValueError("Missing or empty token string.")

    token_clean = token.replace("Bearer ", "").strip()
    payload = parse_jwt_unverified(token_clean)
    validate_jwt_claims(payload)

    if public_key_pem:
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives.serialization import load_pem_public_key

            parts = token_clean.split(".")
            header_and_payload = f"{parts[0]}.{parts[1]}".encode("ascii")
            sig_b64 = parts[2]
            rem = len(sig_b64) % 4
            if rem > 0:
                sig_b64 += "=" * (4 - rem)
            sig_bytes = base64.urlsafe_b64decode(sig_b64.encode("ascii"))

            pub_key = load_pem_public_key(public_key_pem.encode("utf-8"))
            if hasattr(pub_key, "verify"):
                pub_key.verify(
                    sig_bytes,
                    header_and_payload,
                    padding.PKCS1v15(),
                    hashes.SHA256(),
                )
        except Exception as exc:
            raise ValueError(f"Cryptographic signature verification failed: {exc}") from exc

    return extract_user_from_payload(payload)
