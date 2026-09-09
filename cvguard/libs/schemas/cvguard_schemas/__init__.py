"""CVGuard canonical shared Pydantic v2 schemas package.

Exports current v1 domain entities for findings, reports, and governance.
"""

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
    "Finding",
    "Report",
    "Severity",
    "SignedFinding",
    "__version__",
]
