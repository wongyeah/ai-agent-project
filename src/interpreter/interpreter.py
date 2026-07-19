"""
Python interpreter for executing LLM-generated code snippets in an isolated
child process and capturing their output.

This is largely the execution sandbox from the AIDE-style pipeline:
each call to `Interpreter.run()` writes the given code to a file, runs it
in a subprocess, and streams stdout/stderr back through queues while
enforcing a timeout.

Sandbox hardening (on top of plain process isolation):
    - Memory limit (RLIMIT_AS): the OS kills allocations past this, which
      surfaces as a normal Python MemoryError caught by our exec() handler.
    - CPU time limit (RLIMIT_CPU): the OS sends SIGXCPU past this, which we
      catch via a signal handler and turn into a clean ResourceLimitExceeded
      exception instead of an abrupt process kill.
    - Best-effort network blocking: monkeypatches `socket.socket` inside the
      child process so any attempt to open a network connection raises
      immediately, rather than silently phoning home.

Platform note: `resource.setrlimit` is POSIX-only (Linux/Mac) — it does not
exist on Windows. On Windows, memory/CPU limits are silently skipped (with
a one-time warning); this project's actual execution environment (e.g.
Google Colab) is Linux, so this only matters if you try to run the agent
loop itself on a Windows machine, not for developing/testing this code.

None of this is a substitute for real container-level isolation (e.g.
gVisor, Docker with --network=none and cgroup limits) in a production
system — it's a lightweight, dependency-free layer of defense appropriate
for a single-user local/Colab environment.
"""

import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue

import humanize
import shutup
from dataclasses_json import DataClassJsonMixin

try:
    import resource

    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - exercised only on Windows
    _HAS_RESOURCE = False


class ResourceLimitExceeded(Exception):
    """Raised inside the child process when a CPU time limit is hit."""


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """Result of executing a code snippet: output, timing, and exceptions."""

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e: BaseException, exec_file_name: str):
    """Build a string summary of an exception and its stack trace."""
    tb_lines = traceback.format_exception(e)
    tb_str = "".join(tb_lines)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    """File-like object that redirects writes into a multiprocessing queue."""

    def __init__(self, q: Queue, timeout: int = 5):
        self.queue = q
        self.timeout = timeout

    def write(self, msg: str) -> None:
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            pass  # drop output rather than block indefinitely

    def flush(self) -> None:
        pass


def _block_network() -> None:
    """
    Best-effort network blocking for the child process: monkeypatch
    `socket.socket` so any attempt to construct one raises immediately.
    This covers the vast majority of Python HTTP/networking libraries
    (requests, urllib, etc.), since they all eventually create a raw
    socket. It's not a kernel-level guarantee (a determined process could
    still route around it), but it stops accidental/naive network calls
    from LLM-generated code at zero infrastructure cost.
    """
    import socket

    def _blocked_init(self, *args, **kwargs):
        raise PermissionError(
            "Network access is disabled inside the code execution sandbox."
        )

    socket.socket.__init__ = _blocked_init


def _handle_cpu_limit(signum, frame) -> None:
    raise ResourceLimitExceeded("CPU time limit exceeded")


class Interpreter:
    """Simulates a standalone Python REPL with an execution time limit."""

    def __init__(
        self,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        max_memory_mb: int | None = 4096,
        max_cpu_seconds: int | None = None,
        block_network: bool = True,
    ):
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self.block_network = block_network
        self.process: Process | None = None

        if (max_memory_mb is not None or max_cpu_seconds is not None) and not _HAS_RESOURCE:
            print(
                "Warning: memory/CPU limits were requested but the `resource` "
                "module isn't available on this platform (Windows). Limits "
                "will not be enforced."
            )

    def child_proc_setup(self, result_outq: Queue) -> None:
        shutup.mute_warnings()
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

        if _HAS_RESOURCE:
            if self.max_memory_mb is not None:
                max_bytes = self.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            if self.max_cpu_seconds is not None:
                # Soft limit == hard limit would give the kernel no room to
                # deliver SIGXCPU before force-killing the process with
                # SIGKILL — leave a 1-second buffer so our signal handler
                # actually gets a chance to run and raise a catchable
                # exception instead of the process just vanishing.
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.max_cpu_seconds, self.max_cpu_seconds + 1),
                )
                signal.signal(signal.SIGXCPU, _handle_cpu_limit)

        if self.block_network:
            _block_network()

    def _run_session(self, code_inq: Queue, result_outq: Queue, event_outq: Queue) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e, self.agent_file_name
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            os.remove(self.agent_file_name)
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.join(timeout=0.5)

            if self.process.exitcode is None:
                self.process.kill()
                self.process.join(timeout=0.5)

                if self.process.exitcode is None:
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception as e:
            print(f"Error during process cleanup: {e}")
        finally:
            if self.process is not None:
                self.process.close()
                self.process = None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute the given Python code in a subprocess and return the result."""
        if reset_session:
            if self.process is not None:
                self.cleanup_session()
            self.create_process()
        else:
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            while not self.result_outq.empty():
                continue
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        child_in_overtime = False

        while True:
            try:
                state = self.event_outq.get(timeout=1)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                if not child_in_overtime and not self.process.is_alive():
                    raise RuntimeError("REPL child process died unexpectedly") from None

                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True

                    if running_time > self.timeout + 5:
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                if time.time() - start_collect > 5:
                    break
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                continue
        output.pop()

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of "
                f"{humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds "
                f"(time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
