"""CVGuard Drift Plane Service.

Phase 0 Scaffolding: Visual domain shift detection, feature distribution drift, and calibration decay monitoring.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Shared domain schemas dependency
import cvguard_schemas

app = FastAPI(
    title="CVGuard Drift Plane Service",
    description="Offline statistical covariate shift, concept drift, and visual feature distribution assurance.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Pydantic v2 health status schema."""

    status: str = Field(default="ok", description="Operational status flag.")
    service: str = Field(default="drift-plane", description="Name of the reporting service.")
    version: str = Field(default="0.1.0", description="Service semantic version.")
    schemas_version: str = Field(
        default=cvguard_schemas.__version__,
        description="Version of cvguard_schemas linked to this service runtime.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness/liveness health check probe endpoint."""
    return HealthResponse(
        status="ok",
        service="drift-plane",
        version="0.1.0",
        schemas_version=cvguard_schemas.__version__,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=False)
