"""
Python interpreter for executing LLM-generated code snippets in an isolated
child process and capturing their output.

This is largely the execution sandbox from the AIDE-style pipeline:
each call to `Interpreter.run()` writes the given code to a file, runs it
in a subprocess, and streams stdout/stderr back through queues while
enforcing a timeout.

Sandbox hardening (on top of plain process isolation):
    - Memory limit (RLIMIT_AS, POSIX only): the OS kills allocations past
      this, which surfaces as a normal Python MemoryError caught by our
      exec() handler. Fast and precise, but Linux/Mac only.
    - CPU time limit (RLIMIT_CPU, POSIX only): the OS sends SIGXCPU past
      this, which we catch via a signal handler and turn into a clean
      ResourceLimitExceeded exception instead of an abrupt process kill.
    - Windows Job Object limits (win32 only, see
      `src/interpreter/win_job_object.py`): a kernel-level, independent
      counterpart to the two POSIX mechanisms above — not a variant of
      the psutil poller. Memory is enforced via
      JOB_OBJECT_LIMIT_PROCESS_MEMORY (allocations past the limit fail
      immediately, the same way RLIMIT_AS does — no polling involved).
      CPU time is enforced via JOB_OBJECT_LIMIT_PROCESS_TIME, with a
      background thread blocking on an I/O completion port for the
      kernel's exceeded-limit notification (event-driven, not sampled).
      This is deliberately a second, platform-specific branch living
      alongside the POSIX one rather than a shared abstraction — see the
      module docstring in `win_job_object.py` for why that trade-off was
      made on purpose.
    - Cross-platform polling monitor (psutil, works everywhere): the
      parent process periodically checks the child's memory/CPU usage
      (piggybacking on the same ~1-second polling loop already used for
      the wall-clock timeout) and force-kills it if it exceeds the
      configured limits. This is slower to react (up to ~poll_interval
      lag) than either kernel-level mechanism above, since it's "check
      and kill" rather than "the kernel refuses/terminates outright" — it
      now serves purely as a fallback belt-and-suspenders layer, not as
      Windows' primary enforcement path.
    - Best-effort network blocking: monkeypatches `socket.socket` inside
      the child process so any attempt to open a network connection
      raises immediately, rather than silently phoning home.

Which layers are active, per platform:
    - POSIX (Linux/Mac), `use_resource_limits=True` (default): the fast
      RLIMIT_AS/RLIMIT_CPU kernel limits are enforced, IN ADDITION to the
      psutil poller (belt-and-suspenders — the kernel limit will almost
      always trigger first, since it's instant).
    - POSIX, `use_resource_limits=False`: only the psutil poller enforces
      `max_memory_mb`/`max_cpu_seconds`.
    - Windows, `use_job_object_limits=True` (default) and the Job Object
      was created/attached successfully: the kernel-level Job Object
      limits above are the primary enforcement mechanism, with the
      psutil poller still running underneath as a fallback (e.g. in case
      job-object creation is denied by policy on some locked-down
      machine, or attaching otherwise fails at runtime).
    - Windows, `use_job_object_limits=False`, or Job Object setup failed
      at runtime: falls back to psutil-only enforcement (the previous
      Windows behavior).

Platform note: `resource.setrlimit` is POSIX-only (Linux/Mac) — it does not
exist on Windows. Job Objects are the Windows-native equivalent used here;
no extra configuration is needed to benefit from them beyond running on a
Windows version/policy that allows job-object creation.

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

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - only if psutil isn't installed
    _HAS_PSUTIL = False

_HAS_WIN_JOB_OBJECT = False
if sys.platform == "win32":
    try:
        from src.interpreter.win_job_object import WindowsJobObjectLimiter

        _HAS_WIN_JOB_OBJECT = True
    except (ImportError, OSError, AttributeError):  # pragma: no cover
        # Job Objects should be available on any Windows this project
        # targets, but fail closed (fall back to psutil) rather than
        # crash the whole interpreter if the ctypes bindings ever don't
        # load on some exotic Windows configuration.
        _HAS_WIN_JOB_OBJECT = False


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
        use_resource_limits: bool = True,
        use_job_object_limits: bool = True,
        poll_interval_seconds: float = 1.0,
    ):
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self.block_network = block_network
        self.use_resource_limits = use_resource_limits
        self.use_job_object_limits = use_job_object_limits
        self.poll_interval_seconds = poll_interval_seconds
        self.process: Process | None = None
        self._win_job_limiter: "WindowsJobObjectLimiter | None" = None

        wants_limits = max_memory_mb is not None or max_cpu_seconds is not None
        posix_limits_active = use_resource_limits and _HAS_RESOURCE
        job_object_active = (
            sys.platform == "win32" and use_job_object_limits and _HAS_WIN_JOB_OBJECT
        )
        self._job_object_enforcement_active = job_object_active and wants_limits

        if wants_limits and not posix_limits_active and not job_object_active and not _HAS_PSUTIL:
            print(
                "Warning: memory/CPU limits were requested, but no "
                "enforcement mechanism (POSIX `resource`, Windows Job "
                "Objects, or `psutil`) is available, so no limits will be "
                "enforced. Install psutil for cross-platform enforcement "
                "(pip install psutil)."
            )
        elif wants_limits and self._job_object_enforcement_active:
            print(
                "Note: enforcing memory/CPU limits via Windows Job Objects "
                "(kernel-level; see src/interpreter/win_job_object.py), "
                "with the psutil poller kept running as a fallback."
            )
        elif wants_limits and not posix_limits_active:
            print(
                "Note: enforcing memory/CPU limits via the cross-platform "
                "psutil poller (checked roughly every "
                f"{poll_interval_seconds}s) rather than a kernel-level "
                "mechanism."
            )

    def child_proc_setup(self, result_outq: Queue) -> None:
        shutup.mute_warnings()
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

        if self.use_resource_limits and _HAS_RESOURCE:
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

        if self._job_object_enforcement_active:
            # Attach *before* any code is handed to the child (it blocks
            # on code_inq.get() first, see _run_session), so no untrusted
            # code ever runs outside the job's limits — see the race-
            # condition note in win_job_object.py's module docstring for
            # why this ordering matters.
            limiter = WindowsJobObjectLimiter(
                max_memory_mb=self.max_memory_mb, max_cpu_seconds=self.max_cpu_seconds
            )
            try:
                limiter.attach(self.process.pid)
                self._win_job_limiter = limiter
            except OSError as e:
                print(
                    f"Warning: failed to attach Windows Job Object limits "
                    f"({e}); falling back to psutil polling for this run."
                )
                limiter.close()
                self._win_job_limiter = None

    def cleanup_session(self) -> None:
        if self._win_job_limiter is not None:
            self._win_job_limiter.close()
            self._win_job_limiter = None

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

    def _psutil_limits_exceeded(self) -> str | None:
        """
        Cross-platform (Windows-compatible) check of the child process's
        current memory/CPU usage via psutil. Returns an exception-type-like
        string if a configured limit has been exceeded, else None.

        Unlike the POSIX `resource` limits, this doesn't stop the process
        from allocating memory/burning CPU the instant it happens — it's a
        "check every poll_interval_seconds and kill if over budget"
        approach, so there's up to ~poll_interval_seconds of lag. That
        trade-off is what makes it possible to implement without any
        platform-specific kernel API, so it works the same way on Windows,
        Linux, and Mac.
        """
        if not _HAS_PSUTIL or self.process is None:
            return None
        try:
            proc = psutil.Process(self.process.pid)
            if self.max_memory_mb is not None:
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                if mem_mb > self.max_memory_mb:
                    return "MemoryLimitExceeded"
            if self.max_cpu_seconds is not None:
                cpu_times = proc.cpu_times()
                if (cpu_times.user + cpu_times.system) > self.max_cpu_seconds:
                    return "CPUTimeLimitExceeded"
        except psutil.NoSuchProcess:
            pass
        return None

    def _job_object_limits_exceeded(self) -> str | None:
        """
        Non-blocking check of the Windows Job Object limiter, if one is
        attached to the current child process. Unlike
        `_psutil_limits_exceeded`, this doesn't sample anything itself —
        it just reads a flag that the limiter's background thread already
        set the instant the kernel posted a limit-exceeded notification
        (see win_job_object.py). Memory limit violations never reach here
        at all: they fail the child's own allocation call directly (kernel
        enforced, like RLIMIT_AS), surfacing as an ordinary MemoryError
        that the child reports through the normal state:finished path.
        """
        if self._win_job_limiter is None:
            return None
        return self._win_job_limiter.limit_exceeded()

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
        limit_triggered: str | None = None
        limit_trigger_time: float | None = None
        limit_source: str | None = None  # "job_object" | "psutil", for the report text below

        while True:
            try:
                state = self.event_outq.get(timeout=self.poll_interval_seconds)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                if not child_in_overtime and not self.process.is_alive():
                    raise RuntimeError("REPL child process died unexpectedly") from None

                if limit_triggered is None:
                    # Check the Windows Job Object limiter first: it's
                    # event-driven (the kernel already posted the
                    # notification the instant the limit was crossed), so
                    # it should always beat psutil's next scheduled
                    # sample. Falls through to psutil automatically when
                    # no job object is attached (returns None).
                    limit_triggered = self._job_object_limits_exceeded()
                    if limit_triggered is not None:
                        limit_source = "job_object"
                    else:
                        limit_triggered = self._psutil_limits_exceeded()
                        if limit_triggered is not None:
                            limit_source = "psutil"

                    if limit_triggered is not None:
                        # Same graceful approach as the timeout handling
                        # below: send SIGINT first so the child's own
                        # exec()/except block catches it and goes through
                        # its normal EOF-writing shutdown, rather than
                        # forcibly killing it from the parent (which would
                        # leave the output queue without an EOF marker).
                        # If the trigger came from a Job Object CPU-time
                        # limit, the kernel has typically already
                        # terminated the child by this point, in which
                        # case this just becomes a no-op on an
                        # already-gone pid.
                        try:
                            os.kill(self.process.pid, signal.SIGINT)
                        except OSError:
                            pass
                        child_in_overtime = True
                        limit_trigger_time = time.time()

                if limit_triggered is not None:
                    grace_expired = time.time() - limit_trigger_time > 5
                    child_already_dead = not self.process.is_alive()
                    if grace_expired or child_already_dead:
                        self.cleanup_session()
                        state = (None, limit_triggered, {}, [])
                        exec_time = time.time() - start_time
                        break
                    continue

                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    try:
                        os.kill(self.process.pid, signal.SIGINT)
                    except OSError:
                        pass
                    child_in_overtime = True

                    if running_time > self.timeout + 5:
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        # If the job-object/psutil monitor is what triggered the SIGINT
        # that stopped this run, relabel whatever the child reported (it
        # may say "KeyboardInterrupt", or "TimeoutError" — the child's own
        # except block unconditionally renames KeyboardInterrupt to
        # TimeoutError, which is only correct for the wall-clock-timeout
        # case) with the real cause.
        if limit_triggered is not None:
            state = (state[0], limit_triggered, state[2], state[3])

        output: list[str] = []
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                if time.time() - start_collect > 5:
                    break
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                continue
        # On some platforms (notably Windows, where os.kill(pid, SIGINT)
        # abruptly terminates the process instead of raising a catchable
        # KeyboardInterrupt inside it, unlike POSIX), the child may be
        # killed before it ever writes anything — including the "<|EOF|>"
        # marker — to result_outq. Guard against popping an empty list in
        # that case rather than assuming EOF was always written.
        if output and output[-1] == "<|EOF|>":
            output.pop()

        e_cls_name, exc_info, exc_stack = state[1:]

        detection_note = {
            "job_object": "detected via the Windows Job Object kernel notification",
            "psutil": "detected via psutil polling",
        }.get(limit_source, "detected via psutil polling")

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of "
                f"{humanize.naturaldelta(self.timeout)}"
            )
        elif e_cls_name == "MemoryLimitExceeded":
            output.append(
                f"MemoryLimitExceeded: process exceeded the {self.max_memory_mb}MB "
                f"memory limit ({detection_note})."
            )
        elif e_cls_name == "CPUTimeLimitExceeded":
            output.append(
                f"CPUTimeLimitExceeded: process exceeded the {self.max_cpu_seconds}s "
                f"CPU time limit ({detection_note})."
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds "
                f"(time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
