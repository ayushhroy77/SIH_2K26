"""CVGuard canonical shared Pydantic v2 schemas package.

Exports current v1 domain entities for findings, reports, and governance.
"""

from cvguard_schemas.security import (
    FINDING_PRODUCER_SERVICES,
    INTERNAL_SERVICE_IDENTITIES,
    Role,
    ServiceIdentity,
    UserIdentity,
    create_client_ssl_context,
    create_server_ssl_context,
    get_httpx_mtls_kwargs,
    verify_bearer_token,
)
from cvguard_schemas.v1 import (
    AssetType,
    CoverageStatement,
    Disposition,
    Finding,
    Report,
    Severity,
    SignedFinding,
)

__version__ = "0.2.0"

__all__ = [
    "AssetType",
    "CoverageStatement",
    "Disposition",
    "FINDING_PRODUCER_SERVICES",
    "Finding",
    "INTERNAL_SERVICE_IDENTITIES",
    "Report",
    "Role",
    "ServiceIdentity",
    "Severity",
    "SignedFinding",
    "UserIdentity",
    "__version__",
    "create_client_ssl_context",
    "create_server_ssl_context",
    "get_httpx_mtls_kwargs",
    "verify_bearer_token",
]
