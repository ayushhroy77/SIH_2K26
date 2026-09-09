"""Validation test ensuring cvguard_schemas package imports cleanly across environments."""

import sys
from pathlib import Path


def test_cvguard_schemas_imports() -> None:
    """Ensure cvguard_schemas and canonical models import without circular dependencies."""
    schemas_path = Path(__file__).resolve().parent.parent / "libs" / "schemas"
    if str(schemas_path) not in sys.path:
        sys.path.insert(0, str(schemas_path))

    import cvguard_schemas
    from cvguard_schemas import (
        AssetType,
        CoverageStatement,
        Disposition,
        Finding,
        Report,
        Severity,
        SignedFinding,
    )
    from cvguard_schemas.v1 import (
        AssetType as V1AssetType,
        Finding as V1Finding,
        Report as V1Report,
        SignedFinding as V1SignedFinding,
    )

    assert cvguard_schemas.__version__ == "0.2.0"
    assert AssetType is V1AssetType
    assert Finding is V1Finding
    assert SignedFinding is V1SignedFinding
    assert Report is V1Report
    assert issubclass(Severity, str)
    assert issubclass(Disposition, str)
    assert issubclass(AssetType, str)
