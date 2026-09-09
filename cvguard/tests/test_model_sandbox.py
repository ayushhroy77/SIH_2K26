"""Security Test Suite for CVGuard Model Plane Sandbox Boundary.

Tests:
1. Malicious pickle payload with __reduce__ arbitrary code execution exploit.
   Confirms:
   (a) No arbitrary code executes in the main application process.
   (b) Exploit attempt is caught and reported with security violation status.
   (c) The sandbox process is cleanly terminated without crashing the service.
2. Computational hang / infinite-loop model artifact.
   Confirms wall-clock timeout terminates child process group and returns timeout status.
3. Archive header Zip-Slip directory traversal attempt.
   Confirms pre-flight validation rejects path escapes before decompression.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
import zipfile
from pathlib import Path
import pytest

from sandbox import ModelSandbox, SandboxLimits, SubprocessModelSandbox


class MaliciousExploitPayload:
    """Deliberate hostile pickle payload attempting arbitrary command execution via __reduce__."""

    def __init__(self, target_canary_file: str) -> None:
        self.target_canary_file = target_canary_file

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        # In a vulnerable unpickler (e.g. raw torch.load without sandbox or weights_only),
        # this executes `touch <canary_file>` immediately upon deserialization.
        return (os.system, (f"touch {self.target_canary_file}",))


@pytest.mark.asyncio
async def test_malicious_pickle_exploit_intercepted_without_main_process_execution() -> None:
    """Verify that a hostile pickle with __reduce__ cannot execute shell commands or compromise main process."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        canary_file = tmp_path / "pwned_canary.txt"

        # 1. Construct serialized malicious exploit artifact
        exploit_obj = MaliciousExploitPayload(str(canary_file))
        raw_pickle_bytes = pickle.dumps(exploit_obj)

        # 2. Instantiate isolated sandbox
        limits = SandboxLimits(max_wall_clock_seconds=5.0)
        sandbox = SubprocessModelSandbox(limits=limits, scratch_base_dir=tmp_path)

        # 3. Submit hostile payload to sandbox
        result = await sandbox.execute_task(
            task_type="inspect_format",
            payload={},
            model_bytes=raw_pickle_bytes,
        )

        # 4. CRITICAL SECURITY ASSERTIONS:
        # (a) Exploit payload did NOT execute in the main process or host filesystem
        assert not canary_file.exists(), (
            "CRITICAL SECURITY FAILURE: Canary file was created! "
            "Arbitrary code execution succeeded despite sandbox boundary!"
        )

        # (b) Sandbox intercepted the raw pickle and reported security violation
        assert result.success is False, "Sandbox should have rejected raw pickle payload"
        assert result.status == "security_violation", (
            f"Expected status 'security_violation', got '{result.status}'"
        )
        assert "Raw unconstrained Python pickle stream detected" in (result.error_message or ""), (
            f"Expected raw pickle rejection reason, got: {result.error_message}"
        )


@pytest.mark.asyncio
async def test_computational_hang_model_terminated_by_sandbox_wall_clock_limit() -> None:
    """Verify that a model designed to hang or loop indefinitely is killed by the sandbox boundary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Custom worker simulating an infinite loop
        hang_script = tmp_path / "hang_worker.py"
        hang_script.write_text(
            "import sys, time\n"
            "time.sleep(100)\n"  # Hangs indefinitely
            "sys.stdout.write('{}')\n"
        )

        limits = SandboxLimits(max_wall_clock_seconds=1.5)
        sandbox = SubprocessModelSandbox(
            worker_script_path=hang_script,
            limits=limits,
            scratch_base_dir=tmp_path,
        )

        start = time.monotonic()
        result = await sandbox.execute_task(
            task_type="ping",
            payload={},
        )
        elapsed = time.monotonic() - start

        # Assertions:
        # (a) Did not block forever: terminated close to the 1.5s limit
        assert elapsed < 4.0, f"Sandbox hang took too long to terminate: {elapsed}s"

        # (b) Result status reflects timeout
        assert result.success is False
        assert result.status == "timeout"
        assert "wall-clock timeout limit" in (result.error_message or "")


@pytest.mark.asyncio
async def test_zip_slip_archive_traversal_rejected_prior_to_deserialization() -> None:
    """Verify that a TorchScript container with relative path escapes ('../../') is immediately rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        malicious_zip = tmp_path / "zip_slip.pt"

        # Create zip archive containing a directory traversal path
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            zf.writestr("../../etc/cron.d/malicious_job", b"* * * * * root reboot\n")
            zf.writestr("model.json", b'{"version": 1}')

        zip_bytes = malicious_zip.read_bytes()

        limits = SandboxLimits(max_wall_clock_seconds=5.0)
        sandbox = SubprocessModelSandbox(limits=limits, scratch_base_dir=tmp_path)

        result = await sandbox.execute_task(
            task_type="inspect_format",
            payload={},
            model_bytes=zip_bytes,
        )

        # Assertions:
        assert result.success is False
        assert result.status == "security_violation"
        assert "Zip-slip directory traversal path detected" in (result.error_message or "")
