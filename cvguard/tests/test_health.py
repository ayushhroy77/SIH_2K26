"""Integration tests for the /health endpoints of all six CVGuard services."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent

SERVICES = [
    ("gateway", ROOT_DIR / "services" / "gateway" / "main.py"),
    ("data-plane", ROOT_DIR / "services" / "data-plane" / "main.py"),
    ("model-plane", ROOT_DIR / "services" / "model-plane" / "main.py"),
    ("inference-plane", ROOT_DIR / "services" / "inference-plane" / "main.py"),
    ("drift-plane", ROOT_DIR / "services" / "drift-plane" / "main.py"),
    ("governance", ROOT_DIR / "services" / "governance" / "main.py"),
]


def load_service_app(file_path: Path):
    """Dynamically load the FastAPI application from a service's main.py file."""
    module_name = f"cvguard_test_{file_path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "app")


@pytest.mark.parametrize("service_name,main_path", SERVICES)
def test_service_health_endpoint(service_name: str, main_path: Path) -> None:
    """Validate that every plane exposes a valid /health probe with expected schema."""
    assert main_path.is_file(), f"Service file does not exist: {main_path}"
    
    app = load_service_app(main_path)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200 from {service_name} /health, got {response.status_code}"

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == service_name
    assert "version" in data
    assert "schemas_version" in data
