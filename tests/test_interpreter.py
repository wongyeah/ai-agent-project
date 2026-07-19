"""
Unit tests for the Interpreter's sandbox hardening: memory/CPU limits and
network blocking.

Memory/CPU limit tests come in three flavors, mirroring the three
enforcement branches described in interpreter.py's module docstring:
POSIX (`resource`), Windows (`win_job_object.WindowsJobObjectLimiter`),
and the cross-platform psutil poller. Each is skipped on platforms where
it doesn't apply. Network blocking is implemented via a socket monkeypatch
inside the child process, so it works cross-platform.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interpreter.interpreter import _HAS_PSUTIL, Interpreter

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="memory/CPU resource limits use the POSIX-only `resource` module",
)
requires_psutil = pytest.mark.skipif(
    not _HAS_PSUTIL, reason="psutil is not installed"
)
windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows Job Object limits (src/interpreter/win_job_object.py) are Windows-only",
)


def test_basic_execution_returns_expected_output():
    interp = Interpreter(timeout=10, block_network=False, max_memory_mb=None)
    result = interp.run("print('hello from sandbox')")
    interp.cleanup_session()

    assert "hello from sandbox" in "".join(result.term_out)
    assert result.exc_type is None


def test_exceptions_in_code_are_captured_not_crashed():
    interp = Interpreter(timeout=10, block_network=False, max_memory_mb=None)
    result = interp.run("x = 1 / 0")
    interp.cleanup_session()

    assert result.exc_type == "ZeroDivisionError"


def test_network_blocked_by_default():
    interp = Interpreter(timeout=10)  # block_network=True by default
    result = interp.run(
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    )
    interp.cleanup_session()

    assert result.exc_type == "PermissionError"


def test_network_allowed_when_explicitly_disabled():
    interp = Interpreter(timeout=10, block_network=False, max_memory_mb=None)
    result = interp.run(
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.close()\n"
        "print('socket created fine')\n"
    )
    interp.cleanup_session()

    assert result.exc_type is None
    assert "socket created fine" in "".join(result.term_out)


@posix_only
def test_cpu_limit_terminates_busy_loop():
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=None,
        max_cpu_seconds=1,
    )
    result = interp.run("while True:\n    pass\n")
    interp.cleanup_session()

    assert result.exc_type == "ResourceLimitExceeded"


@posix_only
def test_memory_limit_blocks_huge_allocation():
    """
    NOTE on the chosen limit: RLIMIT_AS caps the *entire* virtual address
    space of the process. Since the child is created via fork() (copy-on-
    write), it inherits the parent's already-mapped memory the instant it's
    created — including any large libraries (torch, pandas, etc.) the
    parent process happened to import for other tests. So this limit has
    to comfortably exceed "whatever the parent process had mapped at fork
    time", not just "what this specific snippet needs". 1536MB gives that
    headroom while still being far below the 8GB allocation attempted
    below.
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=4096,
    )
    result = interp.run("x = bytearray(16 * 1024 * 1024 * 1024)\n")
    interp.cleanup_session()

    assert result.exc_type == "MemoryError"


@requires_psutil
def test_psutil_cpu_limit_works_without_resource_module():
    """
    Cross-platform check: with use_resource_limits=False AND
    use_job_object_limits=False, both kernel-level mechanisms are
    disabled entirely, so this exercises the pure psutil-based polling
    fallback path (this test runs and passes on Windows too, unlike the
    posix_only ones above — and, as of the Job Object addition, it now
    needs use_job_object_limits=False to make sure it's actually
    exercising the fallback and not silently getting kernel-level
    enforcement on Windows).
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=None,
        max_cpu_seconds=1,
        use_resource_limits=False,
        use_job_object_limits=False,
        poll_interval_seconds=0.2,
    )
    result = interp.run("while True:\n    pass\n")
    interp.cleanup_session()

    assert result.exc_type == "CPUTimeLimitExceeded"


@requires_psutil
def test_psutil_memory_limit_works_without_resource_module():
    """
    Same cross-platform path as above, but for memory. Uses code that
    grows memory gradually (rather than one huge allocation) so the
    ~0.2s polling interval has multiple chances to catch it before the
    snippet would naturally finish on its own.
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=200,
        max_cpu_seconds=None,
        use_resource_limits=False,
        use_job_object_limits=False,
        poll_interval_seconds=0.2,
    )
    code = (
        "import time\n"
        "data = []\n"
        "for _ in range(100):\n"
        "    data.append(bytearray(50 * 1024 * 1024))\n"
        "    time.sleep(0.1)\n"
    )
    result = interp.run(code)
    interp.cleanup_session()

    assert result.exc_type == "MemoryLimitExceeded"


def test_job_object_inactive_on_non_windows():
    """
    Sanity check for the parallel-branch wiring itself. This test runs
    everywhere (including Linux/Mac CI): on any non-Windows platform the
    Job Object machinery must stay completely inert, with the interpreter
    falling through to its pre-existing (resource/psutil) enforcement
    paths unchanged.
    """
    interp = Interpreter(timeout=10, block_network=False, max_memory_mb=64)
    if sys.platform != "win32":
        assert interp._job_object_enforcement_active is False
        assert interp._win_job_limiter is None
    interp.cleanup_session()


@windows_only
def test_job_object_memory_limit_blocks_huge_allocation():
    """
    Windows counterpart to test_memory_limit_blocks_huge_allocation above,
    enforced via win_job_object.WindowsJobObjectLimiter
    (JOB_OBJECT_LIMIT_PROCESS_MEMORY) instead of POSIX RLIMIT_AS.

    Unlike the fork()-based POSIX case, there's no copy-on-write headroom
    concern here: multiprocessing uses the 'spawn' method on Windows, so
    the child starts with a small, fresh memory footprint rather than
    inheriting whatever the parent process had mapped.
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=200,
        use_job_object_limits=True,
    )
    assert interp._job_object_enforcement_active
    result = interp.run("x = bytearray(1 * 1024 * 1024 * 1024)\n")
    interp.cleanup_session()

    assert result.exc_type == "MemoryError"


@windows_only
def test_job_object_cpu_limit_terminates_busy_loop():
    """
    Windows counterpart to test_cpu_limit_terminates_busy_loop above.
    Unlike POSIX's catchable SIGXCPU, the kernel terminates the process
    outright once JOB_OBJECT_LIMIT_PROCESS_TIME's PerProcessUserTimeLimit
    is exceeded; the parent detects this via the I/O completion port
    notification (see win_job_object.py) rather than by polling, and
    reports it as CPUTimeLimitExceeded -- same result shape as the
    psutil-detected case, but without the sampling lag.
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=None,
        max_cpu_seconds=1,
        use_job_object_limits=True,
    )
    assert interp._job_object_enforcement_active
    result = interp.run("while True:\n    pass\n")
    interp.cleanup_session()

    assert result.exc_type == "CPUTimeLimitExceeded"


@windows_only
def test_job_object_disabled_falls_back_to_psutil():
    """
    use_job_object_limits=False should behave exactly like the
    pre-existing Windows behavior (psutil-only enforcement), so that code
    path stays alive and testable even now that Job Objects are the
    Windows default.
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=None,
        max_cpu_seconds=1,
        use_job_object_limits=False,
        poll_interval_seconds=0.2,
    )
    assert not interp._job_object_enforcement_active
    result = interp.run("while True:\n    pass\n")
    interp.cleanup_session()

    assert result.exc_type == "CPUTimeLimitExceeded"
