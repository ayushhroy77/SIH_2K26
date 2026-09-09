"""CVGuard Governance Report Generator and Renderer.

Produces deterministic, schema-validated Report objects consolidating
ledger findings, cross-plane coverage statements, and cryptographic reproducibility manifests.
Supports JSON, HTML (Jinja2), and PDF outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template
from pydantic import ValidationError

from cvguard_schemas import (
    AssetType,
    CoverageStatement,
    Disposition,
    Finding,
    Report,
    Severity,
    SignedFinding,
)
from ledger import AuditLedger

logger = logging.getLogger("cvguard.governance.reports")

SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 5,
    Severity.HIGH: 4,
    Severity.MEDIUM: 3,
    Severity.LOW: 2,
    Severity.INFO: 1,
}


def sort_findings_triage(
    items: list[SignedFinding | Finding],
) -> list[SignedFinding | Finding]:
    """Pure sorting function: orders findings by severity descending, then confidence descending."""
    def _sort_key(item: SignedFinding | Finding) -> tuple[int, float]:
        f = item.finding if isinstance(item, SignedFinding) else item
        sev_rank = SEVERITY_ORDER.get(f.severity, 0)
        return (sev_rank, f.confidence)

    return sorted(items, key=_sort_key, reverse=True)


# Manifest location resolution
MANIFEST_CANDIDATE_PATHS = [
    Path(os.getenv("COVERAGE_MANIFEST_PATH", "")),
    Path(__file__).resolve().parent.parent.parent / "libs" / "schemas" / "coverage_manifest.json",
    Path("/cvguard/libs/schemas/coverage_manifest.json"),
    Path("coverage_manifest.json"),
]


def find_coverage_manifest_path() -> Path:
    """Locate the canonical coverage_manifest.json file."""
    for p in MANIFEST_CANDIDATE_PATHS:
        if p and p.is_file():
            return p
    raise FileNotFoundError("Could not locate coverage_manifest.json in standard search paths.")


def load_coverage_manifest() -> tuple[dict[str, Any], str]:
    """Load coverage manifest JSON content and compute its deterministic SHA256 hex digest."""
    path = find_coverage_manifest_path()
    raw_bytes = path.read_bytes()
    manifest_hash = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    return data, manifest_hash


def extract_detector_version(detector_id: str) -> str:
    """Extract semantic version string from detector identifier (e.g. '...:v1.0' -> 'v1.0')."""
    if ":" in detector_id:
        return detector_id.split(":")[-1]
    return "v1.0"


def merge_coverage_statement(
    manifest_data: dict[str, Any],
    findings: list[SignedFinding | Finding],
) -> CoverageStatement:
    """Merge all plane sections of coverage_manifest.json into a unified CoverageStatement."""
    supported_classes: list[dict[str, Any]] = []
    uncovered_classes: list[dict[str, Any]] = []
    detector_versions: dict[str, str] = {}

    detectors = manifest_data.get("detectors", [])
    for det in detectors:
        det_id = det.get("id", "")
        det_name = det.get("name", det_id)
        det_ver = extract_detector_version(det_id)
        detector_versions[det_id] = det_ver

        for item in det.get("covered_attack_classes", []):
            supported_classes.append({
                "detector_id": det_id,
                "detector_name": det_name,
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "mitigation": item.get("mitigation", ""),
            })

        for item in det.get("uncovered_attack_classes", []):
            uncovered_classes.append({
                "detector_id": det_id,
                "detector_name": det_name,
                "name": item.get("name", ""),
                "reason": item.get("reason", ""),
            })

    # Sort deterministically
    supported_classes.sort(key=lambda x: (x["name"], x["detector_id"]))
    uncovered_classes.sort(key=lambda x: (x["name"], x["detector_id"]))
    detector_versions = dict(sorted(detector_versions.items()))

    # Calculate finding statistics
    total_assets = len(findings)
    passed_count = 0
    flagged_count = 0

    for item in findings:
        f = item.finding if isinstance(item, SignedFinding) else item
        if f.disposition == Disposition.ACCEPT:
            passed_count += 1
        else:
            flagged_count += 1

    return CoverageStatement(
        total_assets_scanned=total_assets,
        passed_count=passed_count,
        flagged_count=flagged_count,
        skipped_count=0,
        scope_description=(
            "Consolidated cross-plane integrity inspection across Data Plane, Model Plane, "
            "Inference Plane, and Drift Plane threat boundaries."
        ),
        supported_attack_classes=supported_classes,
        uncovered_attack_classes=uncovered_classes,
        detector_versions=detector_versions,
    )


def build_reproducibility_manifest(
    manifest_hash: str,
    findings: list[SignedFinding | Finding],
    manifest_detector_versions: dict[str, str],
    generation_timestamp: str,
) -> dict[str, Any]:
    """Construct a deterministic reproducibility manifest capturing detector versions and manifest hashes."""
    # Discover distinct detectors referenced in findings
    distinct_detectors = sorted({
        (item.finding.detector if isinstance(item, SignedFinding) else item.detector)
        for item in findings
    })

    detector_versions_map: dict[str, str] = {}
    for det in distinct_detectors:
        # Check manifest mapping first; fallback to parsed string
        ver = manifest_detector_versions.get(det) or extract_detector_version(det)
        detector_versions_map[det] = ver

    return {
        "coverage_manifest_hash": manifest_hash,
        "generation_timestamp": generation_timestamp,
        "detector_versions": dict(sorted(detector_versions_map.items())),
        "engine_version": "0.7.0",
    }


def generate_report(
    ledger: AuditLedger,
    since_id: int = 1,
    explicit_timestamp: str | None = None,
) -> Report:
    """Pull SignedFindings from the ledger starting from since_id and construct a validated Report.

    Guarantees strict determinism: generating a report twice over the same range
    with the same manifest and ledger state produces byte-identical reproducibility data.
    """
    # 1. Pull findings from audit ledger
    signed_findings = ledger.get_entries_since(since_id=since_id)

    # 2. Load coverage manifest
    manifest_data, manifest_hash = load_coverage_manifest()

    # 3. Merge coverage statement
    coverage = merge_coverage_statement(manifest_data, signed_findings)

    # 4. Determine deterministic generation timestamp
    if explicit_timestamp:
        generation_timestamp = explicit_timestamp
    elif signed_findings:
        # Derive deterministically from the latest finding's timestamp
        latest = signed_findings[-1].finding.created_at
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        generation_timestamp = latest.isoformat()
    else:
        generation_timestamp = "2026-09-09T00:00:00+00:00"

    # 5. Build reproducibility dictionary
    reproducibility = build_reproducibility_manifest(
        manifest_hash=manifest_hash,
        findings=signed_findings,
        manifest_detector_versions=coverage.detector_versions,
        generation_timestamp=generation_timestamp,
    )

    # Deterministic report ID based on inputs
    rep_seed = f"{since_id}_{len(signed_findings)}_{manifest_hash}_{generation_timestamp}"
    report_id = f"rep_{hashlib.sha256(rep_seed.encode('utf-8')).hexdigest()[:16]}"

    # Parse created_at datetime
    try:
        created_dt = datetime.fromisoformat(generation_timestamp)
    except Exception:
        created_dt = datetime.now(timezone.utc)

    # 6. Construct Report model
    try:
        report = Report(
            report_id=report_id,
            title=f"CVGuard Integrity Assurance Report (Range: #{since_id}+)",
            findings=signed_findings,
            coverage=coverage,
            reproducibility=reproducibility,
            created_at=created_dt,
        )
    except ValidationError as exc:
        logger.error("Failed to construct valid Report: %s", exc)
        raise RuntimeError(f"Report model validation failed: {exc}") from exc

    # Strict validation check before return
    report_dict = report.model_dump(mode="json")
    validated = Report.model_validate(report_dict)
    return validated


def classify_plane(detector_name: str, asset_type: AssetType | str) -> str:
    """Classify a finding into its architectural plane."""
    det = detector_name.lower()
    asset = str(asset_type).lower()
    if "drift" in det or "distribution" in det:
        return "Drift Plane"
    if "inference" in det or "replay" in det or "merkle" in det or "adversarial_perturbation" in det or "inference_record" in asset:
        return "Inference Plane"
    if "model" in det or "safetensors" in det or "weight" in det or "graph" in det or "model" in asset:
        return "Model Plane"
    return "Data Plane"


REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ report.title }}</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --heading: #f0f6fc;
      --critical: #f85149;
      --high: #ff7b72;
      --medium: #d29922;
      --low: #3fb950;
      --info: #58a6ff;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 40px;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
    }
    header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
      margin-bottom: 32px;
    }
    h1 {
      color: var(--heading);
      margin: 0 0 8px 0;
      font-size: 28px;
    }
    .meta-bar {
      font-size: 14px;
      color: #8b949e;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }
    .section-title {
      color: var(--heading);
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      margin-top: 40px;
      margin-bottom: 20px;
      font-size: 20px;
    }
    /* Executive Summary Grid */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }
    .summary-card {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
    }
    .summary-card .label {
      font-size: 12px;
      text-transform: uppercase;
      color: #8b949e;
      letter-spacing: 0.5px;
    }
    .summary-card .value {
      font-size: 28px;
      font-weight: 700;
      color: var(--heading);
      margin-top: 4px;
    }
    /* Severity Badges */
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-critical { background: #3d1214; color: var(--critical); border: 1px solid var(--critical); }
    .badge-high { background: #3e1b19; color: var(--high); border: 1px solid var(--high); }
    .badge-medium { background: #38280f; color: var(--medium); border: 1px solid var(--medium); }
    .badge-low { background: #122b18; color: var(--low); border: 1px solid var(--low); }
    .badge-info { background: #16243b; color: var(--info); border: 1px solid var(--info); }
    /* Findings Table */
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 24px;
      font-size: 14px;
    }
    th, td {
      border: 1px solid var(--border);
      padding: 10px 14px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background-color: var(--card-bg);
      color: var(--heading);
      font-weight: 600;
    }
    tr:nth-child(even) {
      background-color: rgba(22, 27, 34, 0.5);
    }
    /* Two-column Coverage Box */
    .coverage-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 32px;
    }
    .coverage-column {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 20px;
    }
    .coverage-column h3 {
      margin-top: 0;
      font-size: 16px;
      color: var(--heading);
    }
    .coverage-list {
      padding-left: 20px;
      margin: 0;
      font-size: 13px;
    }
    .coverage-list li {
      margin-bottom: 10px;
    }
    .coverage-list strong {
      color: #e6edf3;
    }
    /* Reproducibility Box */
    .code-box {
      background-color: #040d21;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>{{ report.title }}</h1>
    <div class="meta-bar">
      <span><strong>Report ID:</strong> {{ report.report_id }}</span>
      <span><strong>Sealed Timestamp:</strong> {{ report.created_at.isoformat() }}</span>
      <span><strong>Findings Count:</strong> {{ report.findings|length }}</span>
    </div>
  </header>

  <!-- Executive Risk Summary -->
  <h2 class="section-title">Executive Risk Summary</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="label">Total Scanned</div>
      <div class="value">{{ report.coverage.total_assets_scanned }}</div>
    </div>
    <div class="summary-card">
      <div class="label">Flagged Anomaly / Attack</div>
      <div class="value" style="color: var(--critical);">{{ report.coverage.flagged_count }}</div>
    </div>
    <div class="summary-card">
      <div class="label">Clean / Accepted</div>
      <div class="value" style="color: var(--low);">{{ report.coverage.passed_count }}</div>
    </div>
    <div class="summary-card">
      <div class="label">Critical Findings</div>
      <div class="value" style="color: var(--critical);">
        {{ stats.critical_count }}
      </div>
    </div>
  </div>

  <!-- Per-Plane Finding Sections -->
  <h2 class="section-title">Findings by Architectural Plane</h2>
  {% for plane_name, plane_findings in planes.items() %}
    <h3>{{ plane_name }} ({{ plane_findings|length }} finding{% if plane_findings|length != 1 %}s{% endif %})</h3>
    {% if plane_findings %}
      <table>
        <thead>
          <tr>
            <th>Severity</th>
            <th>Asset</th>
            <th>Detector</th>
            <th>Reason Summary</th>
            <th>Confidence</th>
            <th>Disposition</th>
          </tr>
        </thead>
        <tbody>
          {% for sf in plane_findings %}
            {% set f = sf.finding if sf.finding is defined else sf %}
            <tr>
              <td><span class="badge badge-{{ f.severity.value|lower }}">{{ f.severity.value }}</span></td>
              <td><code>{{ f.asset_ref[:24] }}...</code><br><small>{{ f.asset_type.value }}</small></td>
              <td><code>{{ f.detector }}</code></td>
              <td>
                <strong>{{ f.reason }}</strong>
                {% if f.evidence %}
                  <br><small style="color: #8b949e;">Evidence: {{ f.evidence|join(", ") }}</small>
                {% endif %}
              </td>
              <td>{{ (f.confidence * 100)|round(1) }}%</td>
              <td><strong>{{ f.disposition.value }}</strong></td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <p style="color: #8b949e; font-size: 14px;">No findings reported in this plane for the selected ledger range.</p>
    {% endif %}
  {% endfor %}

  <!-- Coverage & Limitations Statement -->
  <h2 class="section-title">Coverage &amp; Limitations Statement</h2>
  <p style="font-size: 14px; color: #8b949e; margin-bottom: 16px;">
    {{ report.coverage.scope_description }}
  </p>
  <div class="coverage-container">
    <div class="coverage-column">
      <h3 style="color: var(--low);">What This System Checks For ({{ report.coverage.supported_attack_classes|length }} Classes)</h3>
      <ul class="coverage-list">
        {% for item in report.coverage.supported_attack_classes %}
          <li>
            <strong>{{ item.name }}</strong> ({{ item.detector_name }})<br>
            {{ item.description }}<br>
            <em style="color: #8b949e;">Mitigation: {{ item.mitigation }}</em>
          </li>
        {% endfor %}
      </ul>
    </div>
    <div class="coverage-column">
      <h3 style="color: var(--high);">What It Explicitly Does NOT Check For ({{ report.coverage.uncovered_attack_classes|length }} Classes)</h3>
      <ul class="coverage-list">
        {% for item in report.coverage.uncovered_attack_classes %}
          <li>
            <strong>{{ item.name }}</strong> ({{ item.detector_name }})<br>
            <span style="color: #ff7b72;">Boundary Limitation:</span> {{ item.reason }}
          </li>
        {% endfor %}
      </ul>
    </div>
  </div>

  <!-- Reproducibility Manifest -->
  <h2 class="section-title">Cryptographic Reproducibility Manifest</h2>
  <div class="code-box">{{ reproducibility_json }}</div>
</div>
</body>
</html>
"""


def render_report_html(report: Report) -> str:
    """Render a validated Report instance to a comprehensive, styled HTML document."""
    # Organize findings by plane
    planes: dict[str, list[Any]] = {
        "Data Plane": [],
        "Model Plane": [],
        "Inference Plane": [],
        "Drift Plane": [],
    }

    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0
    info_count = 0

    for item in report.findings:
        f = item.finding if isinstance(item, SignedFinding) else item
        plane = classify_plane(f.detector, f.asset_type)
        planes[plane].append(item)

        if f.severity == Severity.CRITICAL:
            critical_count += 1
        elif f.severity == Severity.HIGH:
            high_count += 1
        elif f.severity == Severity.MEDIUM:
            medium_count += 1
        elif f.severity == Severity.LOW:
            low_count += 1
        else:
            info_count += 1

    stats = {
        "critical_count": critical_count,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "info_count": info_count,
    }

    reproducibility_json = json.dumps(report.reproducibility, indent=2, sort_keys=True)

    template = Template(REPORT_HTML_TEMPLATE)
    return template.render(
        report=report,
        planes=planes,
        stats=stats,
        reproducibility_json=reproducibility_json,
    )


def render_report_pdf(report: Report) -> bytes:
    """Render a validated Report instance to PDF bytes using WeasyPrint with robust fallback."""
    html_content = render_report_html(report)

    # Attempt WeasyPrint conversion if installed
    try:
        from weasyprint import HTML  # type: ignore

        return HTML(string=html_content).write_pdf()
    except Exception as exc:
        logger.warning(
            "WeasyPrint PDF rendering unavailable (%s). Using canonical PDF generation fallback.",
            exc,
        )
        return _render_minimal_pdf_fallback(report)


def _render_minimal_pdf_fallback(report: Report) -> bytes:
    """Generate a valid PDF-1.4 document containing the core report text."""
    # Build text stream
    lines = [
        f"CVGuard Integrity Assurance Report",
        f"Report ID: {report.report_id}",
        f"Timestamp: {report.created_at.isoformat()}",
        f"Findings Count: {len(report.findings)}",
        f"Total Scanned: {report.coverage.total_assets_scanned} | Flagged: {report.coverage.flagged_count} | Passed: {report.coverage.passed_count}",
        "",
        "FINDINGS BREAKDOWN:",
    ]
    for idx, item in enumerate(report.findings, start=1):
        f = item.finding if isinstance(item, SignedFinding) else item
        lines.append(f"#{idx} [{f.severity.value}] {f.detector}: {f.reason} (Conf: {f.confidence:.2f}, Disp: {f.disposition.value})")

    lines.extend([
        "",
        f"COVERAGE STATEMENT:",
        f"Supported Attack Classes: {len(report.coverage.supported_attack_classes)}",
        f"Uncovered Attack Classes: {len(report.coverage.uncovered_attack_classes)}",
        "",
        f"REPRODUCIBILITY MANIFEST:",
        f"Manifest Hash: {report.reproducibility.get('coverage_manifest_hash', 'N/A')}",
        f"Engine Version: {report.reproducibility.get('engine_version', '0.7.0')}",
    ])

    content_stream = "BT /F1 10 Tf 50 750 Td 14 TL "
    for line in lines:
        sanitized = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_stream += f"({sanitized}) '\n"
    content_stream += "ET"
    content_bytes = content_stream.encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    objects.append(f"<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1") + content_bytes + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(pdf)
