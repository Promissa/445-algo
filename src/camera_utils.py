"""Camera helpers for Linux V4L2 (Raspberry Pi 5 / Debian).

On Raspberry Pi OS (Debian) USB cameras are exposed through the V4L2
subsystem (/dev/video*).  OpenCV's default backend works on Linux, but for
reliable high-resolution capture on the Pi 5 we explicitly request CAP_V4L2,
set the pixel format to MJPEG, and keep the internal buffer as small as
possible.  This avoids libcamera interference and makes the most of the Pi 5's
USB 3.0 bandwidth for multi-camera setups.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Linux V4L2 backend constant; not available on every OpenCV build, so fall
# back to the default (0) when it is missing.
_DEFAULT_BACKEND: int = getattr(cv2, "CAP_V4L2", 0)

# Reduce internal capture buffer to 1 frame – important for low-latency
# robot vision on embedded boards.
_DEFAULT_BUFFER_SIZE: int = 1

# MJPEG fourcc code used before setting resolution so that cheap USB
# cameras can reliably deliver 1920x1080 on ARM SBCs.
_MJPEG_FOURCC: int = cv2.VideoWriter_fourcc("M", "J", "P", "G")


def parse_cam_arg(value: str) -> int | str:
    """Return int if *value* is numeric, otherwise the raw string (path)."""
    try:
        return int(value)
    except ValueError:
        return value


def open_camera(
    device: int | str,
    width: int = 1920,
    height: int = 1080,
    backend: int = _DEFAULT_BACKEND,
    buffer_size: int = _DEFAULT_BUFFER_SIZE,
    mjpeg: bool = True,
) -> cv2.VideoCapture:
    """Open a V4L2 camera with settings tuned for Raspberry Pi 5.

    Parameters
    ----------
    device:
        V4L2 device index (e.g. 0) or path (e.g. ``/dev/video0``).
    width, height:
        Requested capture resolution.
    backend:
        OpenCV capture backend.  Defaults to ``cv2.CAP_V4L2`` when available.
    buffer_size:
        ``CAP_PROP_BUFFERSIZE`` value.  1 gives the lowest latency.
    mjpeg:
        If True (default), request MJPEG pixel format before setting
        resolution.  Most USB webcams on Linux require this for 1080p.

    Returns
    -------
    cap:
        An already-configured ``cv2.VideoCapture`` instance.

    Raises
    ------
    SystemExit:
        If the device cannot be opened.
    """
    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        # Try once more with the default backend in case V4L2 is unavailable
        # (e.g. running the code on a non-Linux workstation for a quick test).
        if backend != 0:
            cap = cv2.VideoCapture(device, 0)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: cannot open camera {device!r}")

    if mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, _MJPEG_FOURCC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if buffer_size >= 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    # Read back the actual resolution; some cameras ignore the request.
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (width, height):
        print(
            f"  [camera {device}] requested {width}x{height}, got {actual_w}x{actual_h}",
            file=sys.stderr,
        )

    return cap


def list_v4l2_devices() -> list[str]:
    """Return a sorted list of ``/dev/video*`` paths present on the system."""
    return sorted(
        str(p) for p in Path("/dev").glob("video*") if p.is_char_device()
    )


def list_cameras() -> list[tuple[str | int, int, int]]:
    """Probe V4L2 devices and return tuples of (device, width, height).

    Only devices that actually deliver frames are returned.
    """
    devices: list[str] = list_v4l2_devices()
    results: list[tuple[str | int, int, int]] = []
    for dev in devices:
        cap = cv2.VideoCapture(dev, _DEFAULT_BACKEND)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            results.append((dev, w, h))
        cap.release()
    return results
