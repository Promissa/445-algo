from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any


_DETECT_IO_LOCK = threading.Lock()
_NOISY_NATIVE_LINES = ("Error, more than one new minima found.",)


def _filtered_output(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="replace")
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not any(noisy in line for noisy in _NOISY_NATIVE_LINES)
    ]
    return "".join(kept).encode("utf-8", errors="replace")


@contextlib.contextmanager
def _suppress_native_output():
    """Filter known noisy stdout/stderr writes from native detector code."""
    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_write, stdout_fd)
        os.dup2(stderr_write, stderr_fd)
        os.close(stdout_write)
        os.close(stderr_write)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        stdout_data = os.read(stdout_read, 1_000_000)
        stderr_data = os.read(stderr_read, 1_000_000)
        filtered_stdout = _filtered_output(stdout_data)
        filtered_stderr = _filtered_output(stderr_data)
        if filtered_stdout:
            os.write(stdout_fd, filtered_stdout)
        if filtered_stderr:
            os.write(stderr_fd, filtered_stderr)
        os.close(stdout_read)
        os.close(stderr_read)
        os.close(saved_stdout)
        os.close(saved_stderr)


def detect_apriltags_silent(detector: Any, *args: Any, **kwargs: Any) -> Any:
    """Run pupil-apriltags detection without noisy native-library prints."""
    with _DETECT_IO_LOCK:
        with _suppress_native_output():
            return detector.detect(*args, **kwargs)
