"""
Unit tests for the Interpreter's sandbox hardening: memory/CPU limits and
network blocking.

Memory/CPU limit tests are POSIX-only (the `resource` module doesn't exist
on Windows) and are skipped there. Network blocking is implemented via a
socket monkeypatch inside the child process, so it works cross-platform.
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
    Cross-platform check: with use_resource_limits=False, the POSIX
    `resource` module's kernel-level limits are disabled entirely, so this
    exercises the same psutil-based polling path that Windows relies on
    (this test runs and passes on Windows too, unlike the posix_only ones
    above).
    """
    interp = Interpreter(
        timeout=30,
        block_network=False,
        max_memory_mb=None,
        max_cpu_seconds=1,
        use_resource_limits=False,
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
