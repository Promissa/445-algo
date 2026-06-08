"""Manual multi-camera calibration using the turntable AprilTags.

Opens up to 4 physical cameras, detects turntable tags in each live image,
solves the camera pose from the four tag centers, and lets you save all
valid calibrations at once.

Keys (in any OpenCV window):
  Space / s : save current valid calibration(s)
  q / ESC   : quit
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from calibration_core import (
    build_square_frame,
    get_tag_center,
    solve_camera_pose_from_square_centers,
)
from camera_utils import open_camera, parse_cam_arg
from pupil_apriltags import Detector
from view_camera_location import FIXED_CAMERA_K

DistanceCompare = Dict[int, Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Drawing helpers (same as manual_camera_calibration.py)
# ---------------------------------------------------------------------------
def draw_detections(
    frame: np.ndarray,
    detections: Iterable[object],
    target_ids: set[int],
) -> None:
    for det in detections:
        tid = int(getattr(det, "tag_id", -1))
        corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        color = (0, 220, 0) if tid in target_ids else (120, 120, 120)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        center = get_tag_center(det)
        cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"id={tid}",
            (int(corners[0, 0]), int(corners[0, 1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def put_lines(frame: np.ndarray, lines: list[Tuple[str, Tuple[int, int, int]]]) -> None:
    x = 12
    y = 28
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30


# ---------------------------------------------------------------------------
# Calibration helpers (same as manual_camera_calibration.py)
# ---------------------------------------------------------------------------
def load_intrinsics(path: str | None) -> Tuple[np.ndarray, np.ndarray]:
    if not path:
        return FIXED_CAMERA_K.copy(), np.zeros(5, dtype=np.float64)

    npz = np.load(path)
    if "K" not in npz:
        raise SystemExit(f"ERROR: calibration input has no K array: {path}")
    K = np.asarray(npz["K"], dtype=np.float64).reshape(3, 3)
    if "dist_coeffs" in npz:
        dist = np.asarray(npz["dist_coeffs"], dtype=np.float64).reshape(-1)
    else:
        dist = np.zeros(5, dtype=np.float64)
    return K, dist


def get_detection_distance(det: object) -> float | None:
    for attr in ("pose_t", "t"):
        if hasattr(det, attr):
            t = np.asarray(getattr(det, attr), dtype=np.float64).reshape(-1)
            if t.size >= 3:
                return float(np.linalg.norm(t[:3]))
    return None


def compare_tag_distances(
    p_w_c: np.ndarray,
    detections_by_id: Dict[int, object],
    tag_ids: tuple[int, int, int, int],
    world_points: np.ndarray,
) -> DistanceCompare:
    compare: DistanceCompare = {}
    for i, tid in enumerate(tag_ids):
        det = detections_by_id.get(tid)
        if det is None:
            continue
        image_dist = get_detection_distance(det)
        if image_dist is None:
            continue
        world_dist = float(np.linalg.norm(world_points[i] - p_w_c))
        compare[tid] = (world_dist, image_dist, image_dist - world_dist)
    return compare


def format_distance_summary(compare: DistanceCompare) -> str:
    if not compare:
        return "dist check: unavailable"
    abs_errs = [abs(row[2]) for row in compare.values()]
    return f"dist check: mean={np.mean(abs_errs):.3f} max={np.max(abs_errs):.3f}"


def print_distance_comparison(compare: DistanceCompare) -> None:
    if not compare:
        print("distance comparison unavailable")
        return
    print("distance comparison: tag world_dist image_dist diff")
    for tid, (world_dist, image_dist, diff) in sorted(compare.items()):
        print(f"  {tid}: {world_dist:.4f} {image_dist:.4f} {diff:+.4f}")


def save_calibration(
    out_path: Path,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    R_w_c: np.ndarray,
    p_w_c: np.ndarray,
    reproj: float,
    tag_ids: tuple[int, int, int, int],
    world_points: np.ndarray,
    image_size: tuple[int, int],
    distance_compare: DistanceCompare | None = None,
) -> None:
    if distance_compare:
        distance_rows = np.asarray(
            [
                [tid, world_dist, image_dist, diff]
                for tid, (world_dist, image_dist, diff) in sorted(
                    distance_compare.items()
                )
            ],
            dtype=np.float64,
        )
    else:
        distance_rows = np.empty((0, 4), dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        K=K,
        dist_coeffs=dist_coeffs,
        cam_pos_w=p_w_c,
        R_w_c=R_w_c,
        reproj=reproj,
        tag_ids=np.asarray(tag_ids, dtype=np.int32),
        world_points=world_points,
        image_width=image_size[0],
        image_height=image_size[1],
        distance_compare=distance_rows,
        saved_at_unix=time.time(),
    )
    p = p_w_c
    print(
        f"saved -> {out_path}  pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) reproj={reproj:.3f}px"
    )
    print_distance_comparison(distance_compare or {})


# ---------------------------------------------------------------------------
# Per-camera state & worker
# ---------------------------------------------------------------------------
@dataclass
class CamState:
    idx: int
    name: str
    cap: cv2.VideoCapture
    K: np.ndarray
    dist_coeffs: np.ndarray
    dist_map1: np.ndarray | None
    dist_map2: np.ndarray | None
    detector: Detector
    tag_size: float
    tag_ids: tuple[int, int, int, int]
    world_points: np.ndarray
    R_w_g: np.ndarray
    t_w_g: np.ndarray
    out_path: Path

    lock: threading.Lock = field(default_factory=threading.Lock)
    R_w_c: np.ndarray | None = None
    p_w_c: np.ndarray | None = None
    reproj: float = float("nan")
    distance_compare: DistanceCompare = field(default_factory=dict)
    n_visible: int = 0
    latest_frame_bgr: np.ndarray | None = None
    # last valid pose for save fallback
    last_pose: tuple[np.ndarray, np.ndarray, float, DistanceCompare] | None = None


def _camera_worker(state: CamState, stop_event: threading.Event) -> None:
    fx = float(state.K[0, 0])
    fy = float(state.K[1, 1])
    cx = float(state.K[0, 2])
    cy = float(state.K[1, 2])
    tag_id_set = set(state.tag_ids)

    while not stop_event.is_set():
        ok, frame = state.cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        if state.dist_map1 is not None:
            frame = cv2.remap(frame, state.dist_map1, state.dist_map2, cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = state.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=float(state.tag_size),
        )

        centers: Dict[int, np.ndarray] = {}
        detections_by_id: Dict[int, object] = {}
        for det in detections:
            tid = int(getattr(det, "tag_id", -1))
            if tid in tag_id_set:
                centers[tid] = get_tag_center(det)
                detections_by_id[tid] = det

        R_est = None
        p_est = None
        reproj_val = float("nan")
        distance_compare: DistanceCompare = {}
        missing = [tid for tid in state.tag_ids if tid not in centers]
        if not missing:
            img_centers = np.asarray(
                [centers[tid] for tid in state.tag_ids], dtype=np.float64
            )
            try:
                R_est, p_est, rp = solve_camera_pose_from_square_centers(
                    img_centers, state.K, state.R_w_g, state.t_w_g
                )
                reproj_val = float(np.asarray(rp).reshape(-1)[0])
                distance_compare = compare_tag_distances(
                    p_est, detections_by_id, state.tag_ids, state.world_points
                )
                with state.lock:
                    state.last_pose = (
                        R_est.copy(),
                        p_est.copy(),
                        reproj_val,
                        dict(distance_compare),
                    )
            except Exception:
                pass

        # Draw detections onto the frame inside the worker so the main thread
        # can just show it.
        draw_detections(frame, detections, tag_id_set)

        with state.lock:
            state.R_w_c = R_est
            state.p_w_c = p_est
            state.reproj = reproj_val
            state.distance_compare = dict(distance_compare)
            state.n_visible = len(centers)
            state.latest_frame_bgr = frame.copy()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_overlay(state: CamState) -> np.ndarray | None:
    with state.lock:
        frame = (
            None if state.latest_frame_bgr is None else state.latest_frame_bgr.copy()
        )
        R_w_c = state.R_w_c
        p_w_c = state.p_w_c
        reproj = state.reproj
        distance_compare = dict(state.distance_compare)
        n_visible = state.n_visible

    if frame is None:
        return None

    max_diff = 0.2
    if R_w_c is not None and p_w_c is not None:
        lines = [
            (f"READY visible=4/4 reproj={reproj:.3f}px", (0, 220, 0)),
            (
                f"pos=({p_w_c[0]:+.3f}, {p_w_c[1]:+.3f}, {p_w_c[2]:+.3f})",
                (0, 220, 0),
            ),
            (format_distance_summary(distance_compare), (0, 220, 0)),
            ("Space/S save all   Q/Esc quit", (255, 255, 255)),
        ]
        for tid, (world_dist, image_dist, diff) in sorted(distance_compare.items()):
            lines.append(
                (
                    f"id={tid} world={world_dist:.2f} image={image_dist:.2f} diff={diff:+.2f}",
                    (0, 220, 0) if abs(diff) < max_diff else (0, 165, 255),
                )
            )
    else:
        lines = [
            (f"WAIT visible={n_visible}/4", (0, 220, 255)),
            ("show all four turntable tags", (0, 220, 255)),
            ("Space/S save last valid   Q/Esc quit", (255, 255, 255)),
        ]
    put_lines(frame, lines)
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live multi-camera AprilTag turntable calibration (up to 4)."
    )
    ap.add_argument(
        "--cams",
        nargs="+",
        default=["0"],
        help="Camera device indices or paths (up to 4)",
    )
    ap.add_argument(
        "--cam-names",
        nargs="*",
        default=[],
        help="Names for each camera (default: cam0, cam1, ...)",
    )
    ap.add_argument(
        "--calib-in",
        nargs="*",
        default=[],
        help="Optional .npz files with K and dist_coeffs, one per camera",
    )
    ap.add_argument(
        "--calib-out",
        nargs="*",
        default=[],
        help="Output .npz paths (default: calib_cam{i}.npz)",
    )
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--tag-size", type=float, default=0.64)
    ap.add_argument(
        "--tag-ids",
        type=int,
        nargs=4,
        default=[300, 301, 302, 303],
        help="Four turntable tag ids in corner order: bottom-left bottom-right top-right top-left",
    )
    ap.add_argument(
        "--world-points",
        type=float,
        nargs=12,
        default=[
            -1.0, -1.0, 0.0,
            -1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, -1.0, 0.0,
        ],
        help="World points for the four tags, same order as --tag-ids",
    )
    ap.add_argument("--decimate", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=0.0)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    n_cams = min(len(args.cams), 4)
    if n_cams == 0:
        print("ERROR: provide at least one camera via --cams", file=sys.stderr)
        return 1

    tag_ids = tuple(int(tid) for tid in args.tag_ids)
    world_points = np.asarray(args.world_points, dtype=np.float64).reshape(4, 3)
    R_w_g, t_w_g = build_square_frame(world_points)

    # Open cameras and build states
    states: list[CamState] = []
    for i in range(n_cams):
        cam_arg = args.cams[i]
        name = args.cam_names[i] if i < len(args.cam_names) else f"cam{i}"
        cal_in = args.calib_in[i] if i < len(args.calib_in) else None
        cal_out = Path(
            args.calib_out[i]
            if i < len(args.calib_out)
            else f"calib_{name}.npz"
        )

        K, dist_coeffs = load_intrinsics(cal_in)
        cap = open_camera(parse_cam_arg(cam_arg), width=args.width, height=args.height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        map1 = map2 = None
        if np.any(dist_coeffs != 0):
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist_coeffs, None, K, (actual_w, actual_h), cv2.CV_16SC2
            )

        detector = Detector(
            families=args.family,
            nthreads=2,
            quad_decimate=args.decimate,
            quad_sigma=args.blur,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        st = CamState(
            idx=i,
            name=name,
            cap=cap,
            K=K,
            dist_coeffs=dist_coeffs,
            dist_map1=map1,
            dist_map2=map2,
            detector=detector,
            tag_size=args.tag_size,
            tag_ids=tag_ids,
            world_points=world_points,
            R_w_g=R_w_g,
            t_w_g=t_w_g,
            out_path=cal_out,
        )
        states.append(st)
        print(f"[{name}] device={cam_arg} actual={actual_w}x{actual_h}")

    print(f"target tags={tag_ids}")
    print("keys: Space/S save all valid calibrations, Q/ESC quit")

    # Start worker threads
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for st in states:
        t = threading.Thread(target=_camera_worker, args=(st, stop_event), daemon=True)
        t.start()
        threads.append(t)

    # Create windows
    for st in states:
        win = f"calib [{st.name}]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        with st.lock:
            ah = int(st.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aw = int(st.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.resizeWindow(win, min(aw, 960), min(ah, 540))

    try:
        while True:
            for st in states:
                win = f"calib [{st.name}]"
                frame = _render_overlay(st)
                if frame is not None:
                    cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (32, ord("s"), ord("S")):
                saved_any = False
                for st in states:
                    with st.lock:
                        pose = (
                            (st.R_w_c, st.p_w_c, st.reproj, dict(st.distance_compare))
                            if st.R_w_c is not None and st.p_w_c is not None
                            else st.last_pose
                        )
                    if pose is None:
                        print(f"[{st.name}] no valid pose yet; skipped")
                        continue
                    save_calibration(
                        st.out_path,
                        st.K,
                        st.dist_coeffs,
                        pose[0],
                        pose[1],
                        pose[2],
                        tag_ids,
                        world_points,
                        (
                            int(st.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(st.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ),
                        pose[3],
                    )
                    saved_any = True
                if not saved_any:
                    print("no valid camera poses to save")

    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        for st in states:
            st.cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
