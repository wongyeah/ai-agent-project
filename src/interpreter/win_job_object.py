"""
Windows-native Job Object resource limiting.

This is a deliberately *independent* Windows-only code path, not a shim
layered on top of `interpreter.py`'s POSIX `resource`/`signal` branch or
the cross-platform psutil poller. It exists because none of the other two
mechanisms give Windows a real kernel-level guarantee:

    - The POSIX `resource` module (RLIMIT_AS / RLIMIT_CPU) does not exist
      on Windows at all.
    - The psutil poller works everywhere, but it is fundamentally
      "sample the child's stats every `poll_interval_seconds` and kill it
      if it's over budget" — there is an inherent detection lag of up to
      one polling interval, and it can miss short memory/CPU spikes
      entirely if they happen and subside between two polls.

Job Objects are a Win32 kernel primitive purpose-built for exactly this:
grouping one or more processes and imposing resource ceilings that the
kernel itself enforces, with no sampling involved. Accepting a second,
Windows-specific branch (instead of trying to unify everything behind the
psutil abstraction) is a conscious trade-off documented in
`interpreter.py`'s module docstring: more code to maintain, but real
kernel-level enforcement on the platform that previously had none.

Enforcement semantics (these differ from POSIX, and that's expected):

    - Memory (`JOB_OBJECT_LIMIT_PROCESS_MEMORY` / `ProcessMemoryLimit`):
      the kernel fails any commit that would push the process over the
      limit. This is a direct analogue of POSIX `RLIMIT_AS` — no parent
      involvement needed at all. The child's own allocation fails and
      surfaces as an ordinary Python `MemoryError`, caught by the same
      `exec()` try/except in `interpreter.py` that handles it on Linux.
      There is nothing to poll for; attaching the process to the job
      before any user code runs is sufficient.

    - CPU time (`JOB_OBJECT_LIMIT_PROCESS_TIME` / `PerProcessUserTimeLimit`):
      the kernel force-terminates every process in the job once its
      accumulated *user-mode* CPU time exceeds the limit. Windows has no
      equivalent of a catchable `SIGXCPU`, so — unlike the memory case —
      the child cannot intercept this and exit cleanly; it is killed
      outright by the OS, the same way `SIGKILL` would look on POSIX.
      To still let the parent report *why* the child died (instead of
      it surfacing as a generic "process died unexpectedly"), the job is
      associated with an I/O completion port, and a background thread
      blocks on `GetQueuedCompletionStatus` waiting for the
      `JOB_OBJECT_MSG_END_OF_PROCESS_TIME` notification. That wait
      returns the instant the kernel posts the message — there is no
      polling interval to tune, which is the practical win over the
      psutil-based poller for the CPU-limit case.

Known limitation worth calling out: the process is only assigned to the
job *after* `multiprocessing.Process.start()` returns and its pid is
known (Win32 offers no "start suspended, pre-attach the job, then
resume" hook through the stdlib `multiprocessing` API used here). In
practice this is safe for this codebase's usage: the child blocks on
`code_inq.get()` immediately after start and does no meaningful work
until `Interpreter.run()` hands it code, and `create_process()` attaches
the job before that happens — so no untrusted code ever runs outside the
job's limits. But it is a real (if narrow) race in the abstract, worth
knowing about if this module is reused elsewhere.
"""

import ctypes
import sys
import threading
from ctypes import wintypes

if sys.platform != "win32":  # pragma: no cover - exercised only on Windows
    raise ImportError("win_job_object is only usable on win32")

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# --- JOBOBJECTINFOCLASS values (winnt.h) ----------------------------------
JobObjectBasicLimitInformation = 2
JobObjectExtendedLimitInformation = 9
JobObjectAssociateCompletionPortInformation = 7

# --- JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags bits --------------------
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100

# --- Job object I/O completion port message identifiers -------------------
# (posted as the lpNumberOfBytesTransferred value of GetQueuedCompletionStatus)
JOB_OBJECT_MSG_END_OF_PROCESS_TIME = 8

# --- Process access rights needed for AssignProcessToJobObject ------------
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_INFORMATION = 0x0400

INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_void_p),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_ASSOCIATE_COMPLETION_PORT(ctypes.Structure):
    _fields_ = [
        ("CompletionKey", ctypes.c_void_p),
        ("CompletionPort", wintypes.HANDLE),
    ]


kernel32.CreateJobObjectW.restype = wintypes.HANDLE
kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]

kernel32.SetInformationJobObject.restype = wintypes.BOOL
kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]

kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
kernel32.CreateIoCompletionPort.argtypes = [
    wintypes.HANDLE,
    wintypes.HANDLE,
    ctypes.c_size_t,
    wintypes.DWORD,
]

kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
kernel32.GetQueuedCompletionStatus.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.DWORD,
]

kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _check(ok, fn_name: str) -> None:
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(
            f"{fn_name} failed (WinError {err}): {ctypes.FormatError(err)}"
        )


class WindowsJobObjectLimiter:
    """
    Attaches a single child process to a fresh Job Object with kernel-level
    memory and/or CPU-time limits, and (for the CPU-time case only) watches
    for the kernel's limit-exceeded notification on a background thread.

    Usage mirrors the POSIX `resource.setrlimit` call this replaces on
    Windows, except it must be driven from the *parent* process (Job
    Objects are assigned to a process handle from the outside) rather than
    from inside the child, since Windows has no `fork()`-time hook
    equivalent to calling `resource.setrlimit()` before `exec()`.
    """

    def __init__(self, max_memory_mb: int | None, max_cpu_seconds: int | None):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self._job = None
        self._port = None
        self._process_handle = None
        self._monitor_thread: threading.Thread | None = None
        self._result_lock = threading.Lock()
        self._exceeded: str | None = None
        self._stop = threading.Event()

    @property
    def active(self) -> bool:
        return self.max_memory_mb is not None or self.max_cpu_seconds is not None

    def attach(self, pid: int) -> None:
        """Create the job object, apply the configured limits, assign
        `pid` to it, and (if a CPU limit was requested) start the
        completion-port monitor thread. No-op if neither limit was
        configured."""
        if not self.active:
            return

        self._job = kernel32.CreateJobObjectW(None, None)
        _check(self._job, "CreateJobObjectW")

        wants_cpu_limit = self.max_cpu_seconds is not None
        if wants_cpu_limit:
            # Must associate the completion port with the job *before*
            # the process is assigned to it, so no early notification can
            # be missed.
            self._port = kernel32.CreateIoCompletionPort(
                INVALID_HANDLE_VALUE, None, 0, 1
            )
            _check(self._port, "CreateIoCompletionPort")

            assoc = JOBOBJECT_ASSOCIATE_COMPLETION_PORT()
            assoc.CompletionKey = None
            assoc.CompletionPort = self._port
            ok = kernel32.SetInformationJobObject(
                self._job,
                JobObjectAssociateCompletionPortInformation,
                ctypes.byref(assoc),
                ctypes.sizeof(assoc),
            )
            _check(ok, "SetInformationJobObject(AssociateCompletionPort)")

        ext = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = 0
        if wants_cpu_limit:
            # 100-nanosecond ticks, per-process user-mode CPU time.
            ext.BasicLimitInformation.PerProcessUserTimeLimit = int(
                self.max_cpu_seconds * 10_000_000
            )
            flags |= JOB_OBJECT_LIMIT_PROCESS_TIME
        if self.max_memory_mb is not None:
            ext.ProcessMemoryLimit = int(self.max_memory_mb) * 1024 * 1024
            flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
        ext.BasicLimitInformation.LimitFlags = flags

        ok = kernel32.SetInformationJobObject(
            self._job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(ext),
            ctypes.sizeof(ext),
        )
        _check(ok, "SetInformationJobObject(ExtendedLimitInformation)")

        self._process_handle = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION,
            False,
            pid,
        )
        _check(self._process_handle, "OpenProcess")

        ok = kernel32.AssignProcessToJobObject(self._job, self._process_handle)
        _check(ok, "AssignProcessToJobObject")

        if wants_cpu_limit:
            self._monitor_thread = threading.Thread(
                target=self._monitor_completion_port, daemon=True
            )
            self._monitor_thread.start()

    def _monitor_completion_port(self) -> None:
        bytes_transferred = wintypes.DWORD(0)
        completion_key = ctypes.c_size_t(0)
        overlapped = ctypes.c_void_p(0)
        while not self._stop.is_set():
            ok = kernel32.GetQueuedCompletionStatus(
                self._port,
                ctypes.byref(bytes_transferred),
                ctypes.byref(completion_key),
                ctypes.byref(overlapped),
                1000,  # wake up at least once a second to re-check _stop
            )
            if not ok:
                # Either a timeout (nothing to report yet) or the port
                # was closed out from under us during shutdown; either
                # way, loop back and let the `_stop` check above decide.
                continue
            if bytes_transferred.value == JOB_OBJECT_MSG_END_OF_PROCESS_TIME:
                with self._result_lock:
                    self._exceeded = "CPUTimeLimitExceeded"
                return

    def limit_exceeded(self) -> str | None:
        """Non-blocking check, safe to call from the parent's poll loop."""
        with self._result_lock:
            return self._exceeded

    def close(self) -> None:
        """Tear down the monitor thread and close all handles. Safe to
        call multiple times."""
        self._stop.set()
        for handle in (self._process_handle, self._port, self._job):
            if handle:
                kernel32.CloseHandle(handle)
        self._process_handle = self._port = self._job = None
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.5)
