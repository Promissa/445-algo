from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Optional

import cv2

CAMERA_PROTOCOL_CHOICES = ("auto", "debian")
_VIDEO_DEVICE_RE = re.compile(r"^video(\d+)$")


@dataclass(frozen=True)
class OpenedCamera:
    cap: cv2.VideoCapture
    requested: str
    source: int | str
    protocol: str
    backend_name: str
    fourcc: str


def default_camera_protocol() -> str:
    return "debian" if sys.platform.startswith("linux") else "auto"


def add_camera_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera-protocol",
        choices=CAMERA_PROTOCOL_CHOICES,
        default=default_camera_protocol(),
        help=(
            "Camera capture protocol. On Linux the default is 'debian', which "
            "uses V4L2 and maps numeric cameras to /dev/videoN. Use 'auto' to "
            "keep OpenCV's default backend."
        ),
    )
    parser.add_argument(
        "--camera-fourcc",
        default=None,
        metavar="FOURCC",
        help=(
            "Optional camera pixel format such as MJPG or YUYV. By default, "
            "Debian/V4L2 capture requests MJPG; pass 'none' to disable."
        ),
    )


def parse_camera_source(cam_arg: str, protocol: Optional[str] = None) -> tuple[int | str, str]:
    requested = str(cam_arg).strip()
    if not requested:
        raise ValueError("camera argument cannot be empty")

    explicit_v4l2 = False
    raw = requested
    if raw.startswith("v4l2://"):
        raw = raw[len("v4l2://") :]
        explicit_v4l2 = True
    elif raw.startswith("v4l2:"):
        raw = raw[len("v4l2:") :]
        explicit_v4l2 = True

    if not raw:
        raise ValueError(f"camera argument {requested!r} has no device")

    effective_protocol = "debian" if explicit_v4l2 else (protocol or default_camera_protocol())
    if effective_protocol not in CAMERA_PROTOCOL_CHOICES:
        raise ValueError(
            f"unsupported camera protocol {effective_protocol!r}; "
            f"expected one of {', '.join(CAMERA_PROTOCOL_CHOICES)}"
        )

    if effective_protocol == "debian":
        if raw.isdecimal():
            return f"/dev/video{raw}", effective_protocol
        video_match = _VIDEO_DEVICE_RE.fullmatch(raw)
        if video_match:
            return f"/dev/{raw}", effective_protocol
        return raw, effective_protocol

    try:
        return int(raw), effective_protocol
    except ValueError:
        return raw, effective_protocol


def normalize_camera_fourcc(fourcc: Optional[str], protocol: str) -> str:
    if fourcc is None:
        return "MJPG" if protocol == "debian" else ""

    normalized = fourcc.strip()
    if normalized.lower() in {"", "none", "off", "auto"}:
        return ""
    if len(normalized) != 4:
        raise ValueError(
            f"camera FOURCC must be exactly 4 characters, got {fourcc!r}"
        )
    return normalized


def open_camera(
    cam_arg: str,
    width: int,
    height: int,
    protocol: Optional[str] = None,
    fourcc: Optional[str] = None,
) -> OpenedCamera:
    source, effective_protocol = parse_camera_source(cam_arg, protocol=protocol)
    backend = cv2.CAP_V4L2 if effective_protocol == "debian" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)

    effective_fourcc = normalize_camera_fourcc(fourcc, effective_protocol)
    if effective_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*effective_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open camera {cam_arg!r} as {source!r} with "
            f"{effective_protocol} protocol"
        )

    try:
        backend_name = cap.getBackendName()
    except Exception:
        backend_name = "unknown"

    return OpenedCamera(
        cap=cap,
        requested=str(cam_arg),
        source=source,
        protocol=effective_protocol,
        backend_name=backend_name,
        fourcc=effective_fourcc,
    )
