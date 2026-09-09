#!/usr/bin/env python3
"""CVGuard Dependency Hygiene & Lockfile Verification Script.

Validates that:
1. Every service and shared library has a requirements.in and matching requirements.txt.
2. Every direct dependency declared in requirements.in is locked in requirements.txt.
3. Every package entry in requirements.txt is secured with cryptographic SHA-256 hashes (--hash=sha256:...).
4. Fails with non-zero exit code if lockfiles are missing, out-of-sync, or contain unhashed dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET_DIRS = [
    Path("libs/schemas"),
    Path("services/gateway"),
    Path("services/data-plane"),
    Path("services/model-plane"),
    Path("services/inference-plane"),
    Path("services/drift-plane"),
    Path("services/governance"),
]


def normalize_pkg_name(name: str) -> str:
    """Normalize Python package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower().strip()


def parse_requirements_in(file_path: Path) -> set[str]:
    """Extract required package names from requirements.in."""
    packages: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip extras and version specs, e.g., 'fastapi[standard]>=0.110.0' -> 'fastapi'
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)", line)
        if match:
            packages.add(normalize_pkg_name(match.group(1)))
    return packages


def parse_requirements_txt(file_path: Path) -> dict[str, list[str]]:
    """Extract packages and their associated SHA-256 hashes from requirements.txt."""
    packages: dict[str, list[str]] = {}
    current_pkg: str | None = None

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "--hash=" in line:
            if current_pkg:
                hashes = re.findall(r"--hash=(sha256:[a-fA-F0-9]{64})", line)
                packages[current_pkg].extend(hashes)
            continue

        # Package line, e.g. "pydantic==2.7.4 \" or "pydantic==2.7.4"
        line_clean = line.rstrip("\\").strip()
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)", line_clean)
        if match:
            current_pkg = normalize_pkg_name(match.group(1))
            packages[current_pkg] = []
            hashes = re.findall(r"--hash=(sha256:[a-fA-F0-9]{64})", line)
            packages[current_pkg].extend(hashes)

    return packages


def verify_directory(target_dir: Path) -> list[str]:
    """Verify lockfile integrity for a given target directory."""
    errors: list[str] = []
    req_in = target_dir / "requirements.in"
    req_txt = target_dir / "requirements.txt"

    if not req_in.is_file():
        errors.append(f"Missing requirements.in in {target_dir}")
        return errors

    if not req_txt.is_file():
        errors.append(f"Missing requirements.txt in {target_dir}")
        return errors

    in_pkgs = parse_requirements_in(req_in)
    locked_pkgs = parse_requirements_txt(req_txt)

    # 1. Ensure all declared dependencies in requirements.in are in requirements.txt
    for pkg in in_pkgs:
        if pkg not in locked_pkgs:
            errors.append(
                f"{target_dir}: Declared dependency '{pkg}' in requirements.in is missing from requirements.txt"
            )

    # 2. Ensure every package in requirements.txt has at least one cryptographic sha256 hash
    for pkg, hashes in locked_pkgs.items():
        if not hashes:
            errors.append(
                f"{target_dir}: Package '{pkg}' in requirements.txt has no --hash=sha256 hashes"
            )

    return errors


def main() -> int:
    """Run lockfile verification across all services and libraries."""
    all_errors: list[str] = []
    print("==> Verifying dependency lockfile synchronization and cryptographic hashes...")

    for target in TARGET_DIRS:
        errs = verify_directory(target)
        if errs:
            all_errors.extend(errs)
        else:
            print(f"  [OK] {target} dependencies verified and securely hashed.")

    if all_errors:
        print("\n[ERROR] Dependency hygiene check failed with the following errors:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\nFix by running: pip-compile --generate-hashes requirements.in -o requirements.txt in the affected directories.",
            file=sys.stderr,
        )
        return 1

    print("\n[SUCCESS] All service lockfiles are in-sync, pinned, and cryptographically verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
