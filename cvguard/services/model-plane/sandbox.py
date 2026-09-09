"""CVGuard Model Plane Sandbox Boundary.

Provides a fortified isolation interface and subprocess-based execution sandbox
for inspecting and running model artifacts supplied by untrusted parties.

Treats every model file as a hostile input capable of arbitrary code execution.
All PyTorch and ONNX runtime loading happens strictly inside this isolated boundary,
never in the main FastAPI application process.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cvguard.modelplane.sandbox")

# TODO: In production, upgrade this process-level sandbox to a full gVisor (runsc)
# or Firecracker microVM boundary before untrusted external model artifacts are ingested.
# While the SubprocessModelSandbox enforces hard memory, CPU, file-descriptor, and
# execution-time limits, true kernel attack surface isolation (seccomp-bpf filters,
# separate network/mount/PID namespaces, and syscall virtualisation) requires gVisor or
# a dedicated microVM hypervisor.


class SandboxError(RuntimeError):
    """Base exception for sandbox execution failures."""


class SandboxTimeoutError(SandboxError):
    """Raised when the sandbox worker exceeds maximum permissible wall-clock duration."""


class SandboxMemoryLimitError(SandboxError):
    """Raised when the sandbox worker exceeds maximum permissible address space/memory."""


class SandboxSecurityViolation(SandboxError):
    """Raised when the sandbox worker detects an exploit payload or unauthorized action."""


@dataclass(frozen=True)
class SandboxLimits:
    """Hard resource boundaries applied to sandboxed model loading processes."""

    max_memory_bytes: int = 1024 * 1024 * 1024  # 1 GB RAM
    max_cpu_seconds: int = 15  # 15 seconds CPU time
    max_wall_clock_seconds: float = 20.0  # 20 seconds real-time timeout
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB max output file
    max_open_files: int = 64  # Restrict open file descriptors


@dataclass
class SandboxResult:
    """Structured response returned by a sandbox worker invocation."""

    success: bool
    status: str  # "ok", "timeout", "oom", "security_violation", "load_failed", "error"
    data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    execution_time_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""


class ModelSandbox(abc.ABC):
    """Abstract interface for executing untrusted model operations within an isolation boundary."""

    @abc.abstractmethod
    async def execute_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        model_bytes: bytes | None = None,
        model_path: str | None = None,
    ) -> SandboxResult:
        """Dispatch a task to the sandboxed execution environment.

        Args:
            task_type: Identifier of the operation (e.g., "load_and_inspect", "weight_stats",
                       "predict_batch", "extract_activations").
            payload: Parameters for the task.
            model_bytes: Raw bytes of the untrusted model file.
            model_path: Optional existing path to the untrusted model file.

        Returns:
            SandboxResult with execution status and parsed output data.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Verify the sandbox environment is healthy and operational."""


def _set_resource_limits(limits: SandboxLimits) -> None:
    """Set process resource limits via the POSIX resource module.

    Executed in the child process immediately prior to exec (preexec_fn).
    """
    # Create a new process group so the parent can kill all descendants if needed
    os.setsid()

    # 1. Virtual memory / address space limit (OOM prevention)
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (limits.max_memory_bytes, limits.max_memory_bytes),
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Warning: Failed to set RLIMIT_AS: {exc}\n")

    # 2. CPU time limit
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (limits.max_cpu_seconds, limits.max_cpu_seconds + 2),
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Warning: Failed to set RLIMIT_CPU: {exc}\n")

    # 3. Maximum file size creation limit (prevents disk-filling DoS)
    try:
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (limits.max_file_size_bytes, limits.max_file_size_bytes),
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Warning: Failed to set RLIMIT_FSIZE: {exc}\n")

    # 4. Maximum open file descriptors
    try:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (limits.max_open_files, limits.max_open_files),
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Warning: Failed to set RLIMIT_NOFILE: {exc}\n")

    # 5. Core dump prevention (ensures untrusted memory isn't dumped to disk)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


class SubprocessModelSandbox(ModelSandbox):
    """Fortified subprocess sandbox for inspecting and executing untrusted models.

    Spawns an isolated Python worker process with strictly enforced POSIX limits,
    ephemeral scratch directories, and communication restricted to serialized JSON
    over standard input/output.
    """

    def __init__(
        self,
        worker_script_path: Path | None = None,
        limits: SandboxLimits | None = None,
        scratch_base_dir: Path | None = None,
    ) -> None:
        self.limits = limits or SandboxLimits()
        if worker_script_path is not None:
            self.worker_script = Path(worker_script_path).resolve()
        else:
            self.worker_script = (Path(__file__).parent / "sandbox_worker.py").resolve()

        self.scratch_base_dir = (
            Path(scratch_base_dir) if scratch_base_dir else Path(tempfile.gettempdir())
        )

    async def health_check(self) -> bool:
        """Run an empty ping task in the sandbox to verify subprocess lifecycle."""
        try:
            result = await self.execute_task("ping", {"ping": "pong"})
            return result.success and result.data.get("pong") == "ping"
        except Exception as exc:
            logger.error("Sandbox health check failed: %s", exc)
            return False

    async def execute_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        model_bytes: bytes | None = None,
        model_path: str | None = None,
    ) -> SandboxResult:
        """Execute a model operation inside the isolated subprocess boundary."""
        start_time = time.monotonic()
        temp_dir = tempfile.mkdtemp(prefix="cvguard_sandbox_", dir=self.scratch_base_dir)
        temp_path = Path(temp_dir)

        target_model_file: Path | None = None
        if model_bytes is not None:
            # Write model bytes into isolated scratch dir with read-only permissions
            target_model_file = temp_path / "model.artifact"
            target_model_file.write_bytes(model_bytes)
            try:
                os.chmod(target_model_file, 0o400)  # Read-only to child
            except OSError:
                pass
        elif model_path is not None:
            target_model_file = Path(model_path)

        request_payload = {
            "task_type": task_type,
            "model_path": str(target_model_file) if target_model_file else None,
            "params": payload,
            "scratch_dir": str(temp_path),
        }

        # Build clean environment with network isolation indicators
        sanitized_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(temp_path),
            "TEMP": str(temp_path),
            "TMP": str(temp_path),
            # Cut off outbound proxy routing as environment defense
            "HTTP_PROXY": "http://127.0.0.1:0",
            "HTTPS_PROXY": "http://127.0.0.1:0",
            "ALL_PROXY": "http://127.0.0.1:0",
            "no_proxy": "*",
            "NO_PROXY": "*",
        }

        cmd = [sys.executable, str(self.worker_script)]

        try:
            # Run child process via asyncio subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sanitized_env,
                cwd=str(temp_path),
                preexec_fn=lambda: _set_resource_limits(self.limits),
            )

            input_bytes = json.dumps(request_payload).encode("utf-8")

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=input_bytes),
                    timeout=self.limits.max_wall_clock_seconds,
                )
            except asyncio.TimeoutError:
                # Terminate entire child process group cleanly
                logger.warning(
                    "Sandbox process exceeded wall-clock limit (%ss); terminating process group.",
                    self.limits.max_wall_clock_seconds,
                )
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    await asyncio.sleep(0.5)
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

                execution_time = time.monotonic() - start_time
                return SandboxResult(
                    success=False,
                    status="timeout",
                    error_message=(
                        f"Sandbox process exceeded wall-clock timeout limit "
                        f"({self.limits.max_wall_clock_seconds}s)."
                    ),
                    execution_time_seconds=execution_time,
                )

            execution_time = time.monotonic() - start_time
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            if process.returncode != 0:
                # Check if killed by signal
                if process.returncode == -signal.SIGXCPU or "CPU time limit exceeded" in stderr_str:
                    return SandboxResult(
                        success=False,
                        status="timeout",
                        error_message="Sandbox process exceeded CPU resource limit (SIGXCPU).",
                        execution_time_seconds=execution_time,
                        stdout=stdout_str,
                        stderr=stderr_str,
                    )
                if (
                    process.returncode == -signal.SIGSEGV
                    or process.returncode == -signal.SIGKILL
                    or "MemoryError" in stderr_str
                    or "std::bad_alloc" in stderr_str
                ):
                    return SandboxResult(
                        success=False,
                        status="oom",
                        error_message="Sandbox process exceeded memory allocation limit or crashed (OOM/SIGSEGV).",
                        execution_time_seconds=execution_time,
                        stdout=stdout_str,
                        stderr=stderr_str,
                    )

                # Check for explicit security/exploit messages
                if "SECURITY_VIOLATION" in stderr_str or "SECURITY_VIOLATION" in stdout_str:
                    return SandboxResult(
                        success=False,
                        status="security_violation",
                        error_message="Malicious bytecode or arbitrary execution pattern intercepted during load.",
                        execution_time_seconds=execution_time,
                        stdout=stdout_str,
                        stderr=stderr_str,
                    )

                # General non-zero exit code
                return SandboxResult(
                    success=False,
                    status="load_failed",
                    error_message=f"Sandbox process exited with code {process.returncode}: {stderr_str.strip()}",
                    execution_time_seconds=execution_time,
                    stdout=stdout_str,
                    stderr=stderr_str,
                )

            # Parse JSON output from worker stdout
            try:
                result_data = json.loads(stdout_str)
                is_success = result_data.get("status") == "ok"
                status = result_data.get("status", "ok" if is_success else "error")
                return SandboxResult(
                    success=is_success,
                    status=status,
                    data=result_data.get("data", {}),
                    error_message=result_data.get("error"),
                    execution_time_seconds=execution_time,
                    stdout=stdout_str,
                    stderr=stderr_str,
                )
            except json.JSONDecodeError as exc:
                return SandboxResult(
                    success=False,
                    status="load_failed",
                    error_message=f"Failed to parse sandbox worker output JSON: {exc}. Stderr: {stderr_str}",
                    execution_time_seconds=execution_time,
                    stdout=stdout_str,
                    stderr=stderr_str,
                )

        finally:
            # Clean up ephemeral scratch directory
            try:
                shutil.rmtree(temp_path, ignore_errors=True)
            except OSError:
                pass
