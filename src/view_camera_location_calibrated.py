from __future__ import annotations

"""
Calibrate cameras first, then render every detected AprilTag's 3D world
position directly inside the MuJoCo 3D viewer.

Behavior:
  - Background threads continuously estimate camera poses from turntable tags.
  - Once all cameras have valid live poses, calibration is considered ready.
  - Press M to freeze the current calibrated poses.
  - Each detected AprilTag is shown as a labeled sphere in the MuJoCo viewer;
    a small cyan sphere is drawn per (camera, tag) detection, and a larger
    yellow labeled sphere is drawn at the mean position across cameras.
  - No brick reconstruction is performed.
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from pupil_apriltags import Detector

from calibration_core import (
    build_square_frame,
    get_tag_center,
    solve_camera_pose_from_square_centers,
)
from view_camera_location import (
    _CAM_PALETTES,
    FIXED_CAMERA_K,
    S_CV2MJ,
    TAG_LOCAL,
    CamState,
    _add_cylinder,
    draw_brick,
    draw_camera_geom,
)

MAX_BRICK_TAG_ID = 295

DEFAULT_PNP_TAG_IDS = (300, 301, 302, 303)
DistanceCompare = Dict[int, Tuple[float, float, float]]
_LIVE_WINDOWS_READY: set[str] = set()
_LIVE_WINDOW_ERROR_PRINTED = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Calibrate camera poses first, then press M to start live brick detection."
    )
    ap.add_argument("--xml", default="assets/scene_turntable_only_lowlookat.xml")
    ap.add_argument(
        "--cams",
        nargs="+",
        default=["0"],
        help="Camera device indices or paths (up to 4)",
    )
    ap.add_argument(
        "--calib-in",
        nargs="*",
        default=[],
        help=".npz files (K + dist_coeffs + reference pose), one per camera",
    )
    ap.add_argument(
        "--calib-out",
        nargs="*",
        default=[],
        help="Output .npz paths for saved calibration",
    )
    ap.add_argument("--cam-names", nargs="*", default=[])
    ap.add_argument("--fovy", type=float, default=100.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument(
        "--world-points",
        type=float,
        nargs=12,
        default=[
            -1,
            -1,
            0.0,
            1,
            -1,
            0.0,
            1,
            1,
            0.0,
            -1,
            1,
            0.0,
        ],
    )
    ap.add_argument("--tag-size", type=float, default=0.64)
    ap.add_argument(
        "--pnp-tag-ids",
        type=int,
        nargs=4,
        default=list(DEFAULT_PNP_TAG_IDS),
        help="Four AprilTag ids used for camera-pose PnP, in the same order as --world-points.",
    )
    ap.add_argument("--real-tag-size", type=float, default=0.64)
    ap.add_argument("--mujoco-tag-size", type=float, default=0.64)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--enforce-flat", action="store_true", default=True)
    ap.add_argument("--no-enforce-flat", dest="enforce_flat", action="store_false")
    ap.add_argument("--coord", default="cv2mj_yz", choices=["identity", "cv2mj_yz"])
    ap.add_argument("--out-poses", default="poses_live.txt")
    ap.add_argument("--frustum-depth", type=float, default=2.5)
    ap.add_argument("--axis-len", type=float, default=1.2)
    ap.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Target update rate for live brick detection after calibration is frozen.",
    )
    ap.add_argument(
        "--show-live-capture",
        action="store_true",
        help="Show a live OpenCV window for each camera feed.",
    )
    ap.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not launch the MuJoCo viewer; use OpenCV live windows only.",
    )
    ap.add_argument(
        "--bypass-calib",
        action="store_true",
        help="Skip live PnP calibration and use stored camera poses from --calib-in directly.",
    )
    return ap.parse_args()


def draw_tag_overlay(
    frame: np.ndarray,
    detections: List[object],
    pnp_tag_ids: Tuple[int, int, int, int],
) -> None:
    pnp_set = set(pnp_tag_ids)
    for det in detections:
        tid = int(getattr(det, "tag_id", -1))
        corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        color = (0, 220, 0) if tid in pnp_set else (120, 120, 120)
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
    tag_ids: Tuple[int, int, int, int],
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


def distance_rows_array(compare: DistanceCompare) -> np.ndarray:
    if not compare:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(
        [
            [tid, world_dist, image_dist, diff]
            for tid, (world_dist, image_dist, diff) in sorted(compare.items())
        ],
        dtype=np.float64,
    )


def print_distance_comparison(compare: DistanceCompare) -> None:
    if not compare:
        print("distance comparison unavailable")
        return
    print("distance comparison: tag world_dist image_dist diff")
    for tid, (world_dist, image_dist, diff) in sorted(compare.items()):
        print(f"  {tid}: {world_dist:.4f} {image_dist:.4f} {diff:+.4f}")


def gather_tag_world_positions(
    states: List[CamState], prefer_stored: bool
) -> Dict[str, Dict[int, np.ndarray]]:
    result: Dict[str, Dict[int, np.ndarray]] = {}
    for st in states:
        with st.lock:
            tag_positions_cam = dict(getattr(st, "tag_positions_cam", {}))
            stored_p = st.stored_p
            stored_R = st.stored_R
            live_p = st.p_w_c
            live_R = st.R_w_c
        if prefer_stored and stored_p is not None and stored_R is not None:
            p_w_c, R_w_c = stored_p, stored_R
        else:
            p_w_c, R_w_c = live_p, live_R
        positions: Dict[int, np.ndarray] = {}
        if p_w_c is not None and R_w_c is not None:
            for tid, t_c in tag_positions_cam.items():
                pos = p_w_c + R_w_c @ (S_CV2MJ @ t_c)
                positions[tid] = pos
        result[st.name] = positions
    return result


def print_tag_world_positions(per_cam: Dict[str, Dict[int, np.ndarray]]) -> None:
    print("\n--- AprilTag world positions ---")
    for cam_name, positions in per_cam.items():
        if not positions:
            print(f"  [{cam_name}] no tags")
            continue
        print(f"  [{cam_name}] {len(positions)} tag(s):")
        for tid, p in sorted(positions.items()):
            print(f"    id={tid:>4d}  ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")


_TAG_MARKER_RGBA = (1.00, 0.85, 0.10, 0.95)
_TAG_PERCAM_RGBA = (0.20, 0.85, 1.00, 0.55)


def _add_labeled_sphere(
    viewer,
    pos: np.ndarray,
    radius: float,
    rgba: Tuple[float, float, float, float],
    label: str,
) -> None:
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float32),
        np.asarray(pos, dtype=np.float32),
        np.eye(3, dtype=np.float32).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.matid = -1
    try:
        g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    except Exception:
        pass
    if label:
        try:
            g.label = label.encode("utf-8")[:99]
        except Exception:
            pass
    scn.ngeom += 1


def draw_tag_markers_in_viewer(
    viewer,
    per_cam: Dict[str, Dict[int, np.ndarray]],
) -> None:
    """Render every detected AprilTag's world position as a labeled sphere.

    Each (camera, tag) detection gets a small unlabeled sphere; the mean
    position across cameras gets a larger sphere with the tag id as a label.
    """
    by_tag: Dict[int, List[np.ndarray]] = {}
    for positions in per_cam.values():
        for tid, p in positions.items():
            _add_labeled_sphere(viewer, p, 0.05, _TAG_PERCAM_RGBA, "")
            by_tag.setdefault(tid, []).append(p)
    for tid, ps in by_tag.items():
        mean_p = np.mean(np.stack(ps, axis=0), axis=0)
        _add_labeled_sphere(viewer, mean_p, 0.09, _TAG_MARKER_RGBA, f"id={tid}")


_ALIGNED_LINE_RGBA = (0.20, 1.00, 0.30, 0.95)
_TOWER_BRICK_RGBA = (0.72, 0.45, 0.18, 0.55)


def _clip_z_to_half(z: float) -> float:
    """Round z to the nearest value of form n + 0.5 (3.432 -> 3.5, 4.2 -> 4.5)."""
    return math.floor(z) + 0.5


def _clip_xy_to_int(v: float) -> float:
    """Round to nearest integer (3.2 -> 3, 3.6 -> 4, -4.6 -> -5)."""
    return float(round(v))


def _brick_rotation_for_axis(axis: str, sign: float) -> np.ndarray:
    """Brick-to-world R with local +Z mapped to world ±X or ±Y."""
    if axis == "X":
        z = np.array([sign, 0.0, 0.0])
        x = np.array([0.0, 1.0, 0.0])
    else:
        z = np.array([0.0, sign, 0.0])
        x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _generate_brick_candidate_rotations() -> List[Tuple[str, float, np.ndarray]]:
    """Enumerate 16 brick orientations (long-axis along ±X or ±Y, 4 rotations each)."""
    candidates: List[Tuple[str, float, np.ndarray]] = []
    long_axes = [
        ("X", +1.0, np.array([1.0, 0.0, 0.0])),
        ("X", -1.0, np.array([-1.0, 0.0, 0.0])),
        ("Y", +1.0, np.array([0.0, 1.0, 0.0])),
        ("Y", -1.0, np.array([0.0, -1.0, 0.0])),
    ]
    for axis, sign, z in long_axes:
        if axis == "X":
            x_options = [
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
            ]
        else:
            x_options = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, -1.0]),
            ]
        for x in x_options:
            y = np.cross(z, x)
            R = np.stack([x, y, z], axis=1).astype(np.float64)
            candidates.append((axis, sign, R))
    return candidates


_BRICK_CANDIDATES = _generate_brick_candidate_rotations()


def _fit_end_tag_pair_pose(
    visible: Dict[int, np.ndarray],
) -> Optional[Tuple[str, float, np.ndarray, float]]:
    """Fit directly from the block's local 0/1 end-tag pair.

    This path is deliberately tolerant of noisy endpoint coordinates: any
    non-degenerate observed 0/1 pair yields a snapped X/Y brick orientation.
    """
    if 0 not in visible or 1 not in visible:
        return None

    p0 = np.asarray(visible[0], dtype=np.float64).reshape(3)
    p1 = np.asarray(visible[1], dtype=np.float64).reshape(3)
    d = p0 - p1
    d_xy = d[:2]
    if float(np.linalg.norm(d_xy)) < 1e-6:
        return None

    if abs(float(d_xy[0])) >= abs(float(d_xy[1])):
        axis = "X"
        sign = 1.0 if d_xy[0] >= 0.0 else -1.0
    else:
        axis = "Y"
        sign = 1.0 if d_xy[1] >= 0.0 else -1.0

    R = _brick_rotation_for_axis(axis, sign)
    C = 0.5 * (p0 + p1)
    p_l = np.stack([TAG_LOCAL[0], TAG_LOCAL[1]], axis=0)
    p_w = np.stack([p0, p1], axis=0)
    residual = float(np.sum((p_w - (C + p_l @ R.T)) ** 2))
    return axis, sign, C, residual


def _fit_brick_pose(
    visible: Dict[int, np.ndarray],
) -> Optional[Tuple[str, float, np.ndarray, float]]:
    """Find best (axis, sign, center, residual) for a brick given visible tag
    world positions keyed by local tag index (0..9). Needs >=2 tags."""
    if len(visible) < 2:
        return None
    end_pair_fit = _fit_end_tag_pair_pose(visible)
    if end_pair_fit is not None:
        return end_pair_fit
    p_l = np.stack([TAG_LOCAL[idx] for idx in visible.keys()], axis=0)
    p_w = np.stack(list(visible.values()), axis=0)
    best: Optional[Tuple[str, float, np.ndarray, float]] = None
    for axis, sign, R in _BRICK_CANDIDATES:
        rotated_l = p_l @ R.T
        C = (p_w - rotated_l).mean(axis=0)
        residual = float(np.sum((p_w - (C + rotated_l)) ** 2))
        if best is None or residual < best[3]:
            best = (axis, sign, C, residual)
    return best


def compute_aligned_lines(
    per_cam: Dict[str, Dict[int, np.ndarray]],
    start_id: int = 0,
) -> List[Dict[str, object]]:
    """Fit each visible brick's pose from any >=2 of its 10 tags, snap the
    long axis to ±X/±Y, clip the center (X,Y truncated to int; Z to n+0.5)."""
    start_id = int(start_id)
    by_tag: Dict[int, List[np.ndarray]] = {}
    for positions in per_cam.values():
        for tid, p in positions.items():
            if tid < start_id or tid > MAX_BRICK_TAG_ID:
                continue
            by_tag.setdefault(tid, []).append(p)
    mean_pos: Dict[int, np.ndarray] = {
        tid: np.mean(np.stack(ps, axis=0), axis=0) for tid, ps in by_tag.items()
    }

    bricks_visible: Dict[int, Dict[int, np.ndarray]] = {}
    for tid, p in mean_pos.items():
        rel_tid = tid - start_id
        bricks_visible.setdefault(rel_tid // 10, {})[rel_tid % 10] = p

    lines: List[Dict[str, object]] = []
    for base in sorted(bricks_visible.keys()):
        visible = bricks_visible[base]
        fit = _fit_brick_pose(visible)
        if fit is None:
            continue
        axis, sign, C_raw, residual = fit

        mid = C_raw.copy()
        mid[0] = _clip_xy_to_int(float(C_raw[0]))
        mid[1] = _clip_xy_to_int(float(C_raw[1]))
        mid[2] = _clip_z_to_half(float(C_raw[2]))

        R = _brick_rotation_for_axis(axis, sign)
        half_len = 1.5  # BAR_HALF[2]
        long_dir = R[:, 2]
        a_snapped = mid - long_dir * half_len
        b_snapped = mid + long_dir * half_len

        lines.append(
            {
                "base": base,
                "visible_ids": sorted(start_id + base * 10 + k for k in visible.keys()),
                "axis": axis,
                "sign": sign,
                "R": R,
                "mid_raw": C_raw,
                "mid": mid,
                "a_snapped": a_snapped,
                "b_snapped": b_snapped,
                "length": 2.0 * half_len,
                "residual": residual,
            }
        )
    return lines


def draw_aligned_lines_in_viewer(viewer, lines: List[Dict[str, object]]) -> None:
    for line in lines:
        a = np.asarray(line["a_snapped"], dtype=np.float64)
        b = np.asarray(line["b_snapped"], dtype=np.float64)
        _add_cylinder(viewer, a, b, radius=0.04, rgba=_ALIGNED_LINE_RGBA)


def draw_brick_models_in_viewer(viewer, lines: List[Dict[str, object]]) -> None:
    """Render each aligned line as a brick (1x1x3) centered on its clipped midpoint."""
    for line in lines:
        R = _brick_rotation_for_axis(str(line["axis"]), float(line["sign"]))
        pos = np.asarray(line["mid"], dtype=np.float64)
        draw_brick(viewer, pos, R, rgba=_TOWER_BRICK_RGBA)


def print_aligned_lines(lines: List[Dict[str, object]]) -> None:
    if not lines:
        print("\n--- Tower bricks: none ---")
        return
    print("\n--- Tower bricks (aligned; X,Y truncated to int, Z clipped to n+0.5) ---")
    for line in lines:
        a_s = line["a_snapped"]
        b_s = line["b_snapped"]
        mid_raw = line["mid_raw"]
        mid = line["mid"]
        axis = line["axis"]
        sign = "+" if float(line["sign"]) >= 0 else "-"
        visible = line["visible_ids"]
        print(
            f"  brick{line['base']:>3d}  tags={visible}  "
            f"axis={sign}{axis}  length={float(line['length']):.3f}  "
            f"residual={float(line['residual']):.4f}  "
            f"mid_raw=({float(mid_raw[0]):+.3f},{float(mid_raw[1]):+.3f},{float(mid_raw[2]):+.3f})"
            f" -> mid=({float(mid[0]):+.3f},{float(mid[1]):+.3f},{float(mid[2]):+.3f})"
        )
        print(
            f"    aligned: a=({a_s[0]:+.3f},{a_s[1]:+.3f},{a_s[2]:+.3f})  "
            f"b=({b_s[0]:+.3f},{b_s[1]:+.3f},{b_s[2]:+.3f})  "
            f"mid=({mid[0]:+.3f},{mid[1]:+.3f},{mid[2]:+.3f})"
        )


def _calibration_camera_worker(
    state: CamState,
    stop_event: threading.Event,
    pnp_tag_ids: Tuple[int, int, int, int],
    world_points: np.ndarray,
) -> None:
    fx = float(state.K[0, 0])
    fy = float(state.K[1, 1])
    cx = float(state.K[0, 2])
    cy = float(state.K[1, 2])

    while not stop_event.is_set():
        ok, frame = state.cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        if state.dist_map1 is not None:
            frame = cv2.remap(frame, state.dist_map1, state.dist_map2, cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = state.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=float(state.tag_size),
        )

        centers: Dict[int, np.ndarray] = {}
        detections_by_id: Dict[int, object] = {}
        tag_positions_cam: Dict[int, np.ndarray] = {}
        pnp_set = set(pnp_tag_ids)
        for det in dets:
            tid = int(getattr(det, "tag_id", -1))
            if tid in pnp_set:
                centers[tid] = get_tag_center(det)
                detections_by_id[tid] = det
            if hasattr(det, "pose_t"):
                tag_positions_cam[tid] = np.asarray(
                    det.pose_t, dtype=np.float64
                ).reshape(3)

        R_est = None
        p_est = None
        reproj_val = float("nan")
        distance_compare: DistanceCompare = {}
        if all(tid in centers for tid in pnp_tag_ids):
            img_centers = np.asarray(
                [centers[tid] for tid in pnp_tag_ids], dtype=np.float64
            )
            try:
                R_est, p_est, rp = solve_camera_pose_from_square_centers(
                    img_centers, state.K, state.R_w_g, state.t_w_g
                )
                reproj_val = float(np.asarray(rp).reshape(-1)[0])
                distance_compare = compare_tag_distances(
                    p_est, detections_by_id, pnp_tag_ids, world_points
                )
            except Exception:
                pass

        draw_tag_overlay(frame, dets, pnp_tag_ids)

        with state.lock:
            state.R_w_c = R_est
            state.p_w_c = p_est
            state.reproj = reproj_val
            state.n_visible = sum(1 for tid in pnp_tag_ids if tid in centers)
            state.latest_frame_bgr = frame.copy()
            state.distance_compare = dict(distance_compare)
            state.tag_positions_cam = tag_positions_cam


def build_states(
    args: argparse.Namespace, R_w_g: np.ndarray, t_w_g: np.ndarray
) -> List[CamState]:
    n_cams = min(len(args.cams), 4)
    states: List[CamState] = []
    for i in range(n_cams):
        cam_arg = args.cams[i]
        name = args.cam_names[i] if i < len(args.cam_names) else f"cam{i}"
        cal_in = args.calib_in[i] if i < len(args.calib_in) else None
        cal_out = args.calib_out[i] if i < len(args.calib_out) else f"calib_{name}.npz"

        try:
            cam_idx = int(cam_arg)
        except ValueError:
            cam_idx = cam_arg

        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera '{cam_arg}'")

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        K = FIXED_CAMERA_K.copy()
        dist_coeffs = np.zeros(5, dtype=np.float64)
        saved_R = None
        saved_p = None
        if cal_in:
            npz = np.load(cal_in)
            if "dist_coeffs" in npz:
                dist_coeffs = np.asarray(npz["dist_coeffs"], dtype=np.float64)
            if "cam_pos_w" in npz:
                saved_p = np.asarray(npz["cam_pos_w"], dtype=np.float64).reshape(3)
            if "R_w_c" in npz:
                saved_R = np.asarray(npz["R_w_c"], dtype=np.float64).reshape(3, 3)

        m1 = m2 = None
        if np.any(dist_coeffs != 0):
            m1, m2 = cv2.initUndistortRectifyMap(
                K, dist_coeffs, None, K, (aw, ah), cv2.CV_16SC2
            )

        st = CamState(
            idx=i,
            name=name,
            cap=cap,
            K=K,
            dist_map1=m1,
            dist_map2=m2,
            detector=Detector(
                families="tag36h11", nthreads=2, quad_decimate=1.0, refine_edges=1
            ),
            R_w_g=R_w_g,
            t_w_g=t_w_g,
            tag_size=args.tag_size,
            actual_w=aw,
            actual_h=ah,
            calib_out=cal_out,
            saved_K=K.copy(),
            saved_dist=dist_coeffs.copy(),
        )
        st.stored_R = saved_R
        st.stored_p = saved_p
        states.append(st)
    return states


def all_cameras_live(states: List[CamState]) -> bool:
    if not states:
        return False
    for st in states:
        with st.lock:
            if st.R_w_c is None or st.p_w_c is None:
                return False
    return True


def update_live_capture_windows(
    states: List[CamState], calibration_locked: bool
) -> bool:
    global _LIVE_WINDOW_ERROR_PRINTED

    if _LIVE_WINDOW_ERROR_PRINTED:
        return False

    for st in states:
        with st.lock:
            frame = None if st.latest_frame_bgr is None else st.latest_frame_bgr.copy()
            has_live_pose = st.p_w_c is not None and st.R_w_c is not None
            has_fixed_pose = st.stored_p is not None and st.stored_R is not None
            reproj = st.reproj
            visible = st.n_visible
            distance_compare = getattr(st, "distance_compare", {})

        if frame is None:
            continue
        frame = np.ascontiguousarray(frame)

        if calibration_locked and has_fixed_pose:
            status = "fixed pose | live brick detect"
            color = (0, 255, 0)
        elif has_live_pose:
            rp_text = f"{reproj:.2f}" if math.isfinite(reproj) else "nan"
            status = f"calibrating | rp={rp_text} | visible={visible}/4"
            color = (0, 255, 255)
        else:
            status = f"waiting for calibration | visible={visible}/4"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        if has_live_pose:
            cv2.putText(
                frame,
                format_distance_summary(distance_compare),
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            y = 102
            for tid, (world_dist, image_dist, diff) in sorted(distance_compare.items()):
                row_color = (0, 255, 0) if abs(diff) < 0.2 else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"id={tid} world={world_dist:.2f} image={image_dist:.2f} diff={diff:+.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    row_color,
                    2,
                    cv2.LINE_AA,
                )
                y += 26
        window_name = f"live_capture_{st.name}"
        try:
            if window_name not in _LIVE_WINDOWS_READY:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    window_name,
                    min(int(st.actual_w), 1280),
                    min(int(st.actual_h), 720),
                )
                _LIVE_WINDOWS_READY.add(window_name)
            cv2.imshow(window_name, frame)
        except cv2.error as exc:
            if not _LIVE_WINDOW_ERROR_PRINTED:
                print(
                    "\nOpenCV live capture window failed. On macOS this can happen when "
                    "cv2.imshow is used together with the MuJoCo viewer under mjpython. "
                    "Run with --no-viewer for OpenCV-only calibration windows.\n"
                    f"OpenCV error: {exc}\n"
                )
                _LIVE_WINDOW_ERROR_PRINTED = True
            return False

    try:
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        if not _LIVE_WINDOW_ERROR_PRINTED:
            print(f"\nOpenCV waitKey failed: {exc}\n")
            _LIVE_WINDOW_ERROR_PRINTED = True
        return False
    return key in (32, ord("s"), ord("S"), ord("m"), ord("M"))


def save_calib_with_distance(
    state: CamState,
    tag_ids: Tuple[int, int, int, int],
    world_points: np.ndarray,
) -> bool:
    with state.lock:
        if state.p_w_c is None or state.R_w_c is None:
            return False
        p = state.p_w_c.copy()
        R = state.R_w_c.copy()
        reproj = float(state.reproj)
        distance_compare = dict(getattr(state, "distance_compare", {}))
        state.stored_p = p.copy()
        state.stored_R = R.copy()

    np.savez(
        state.calib_out,
        K=state.saved_K,
        dist_coeffs=state.saved_dist,
        cam_pos_w=p,
        R_w_c=R,
        reproj=reproj,
        tag_ids=np.asarray(tag_ids, dtype=np.int32),
        world_points=world_points,
        image_width=state.actual_w,
        image_height=state.actual_h,
        distance_compare=distance_rows_array(distance_compare),
        saved_at_unix=time.time(),
    )
    print(
        f"\n  [{state.name}] calib saved -> {state.calib_out}  "
        f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) reproj={reproj:.3f}px"
    )
    print_distance_comparison(distance_compare)
    return True


def freeze_calibration(
    states: List[CamState],
    tag_ids: Tuple[int, int, int, int],
    world_points: np.ndarray,
) -> int:
    saved_count = 0
    for st in states:
        if save_calib_with_distance(st, tag_ids, world_points):
            saved_count += 1
    return saved_count


def main() -> None:
    args = parse_args()
    if args.target_fps <= 0:
        sys.exit("ERROR: --target-fps must be > 0")

    base_dir = Path(__file__).resolve().parent

    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = base_dir / xml_path
    if not xml_path.exists():
        sys.exit(f"ERROR: XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pts = np.asarray(args.world_points, dtype=np.float64).reshape(4, 3)
    R_w_g, t_w_g = build_square_frame(pts)
    pnp_tag_ids = tuple(int(tid) for tid in args.pnp_tag_ids)

    try:
        states = build_states(args, R_w_g, t_w_g)
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")

    stop_event = threading.Event()
    threads: List[threading.Thread] = []
    for st in states:
        t = threading.Thread(
            target=_calibration_camera_worker,
            args=(st, stop_event, pnp_tag_ids, pts),
            daemon=True,
        )
        t.start()
        threads.append(t)

    freeze_calibration_requested = threading.Event()
    calibration_ready_announced = False
    calibration_locked = False
    next_tag_print_time = 0.0
    tag_print_interval_s = 1.0 / args.target_fps

    if args.bypass_calib:
        missing = [
            st.name for st in states if st.stored_p is None or st.stored_R is None
        ]
        if missing:
            sys.exit(
                "ERROR: --bypass-calib requires a stored pose for every camera. "
                f"Missing for: {', '.join(missing)}. Pass valid --calib-in .npz files."
            )
        calibration_locked = True
        calibration_ready_announced = True
        print(
            "--bypass-calib: skipping live calibration; using stored camera poses from --calib-in.\n"
        )

    def key_callback(keycode: int) -> None:
        if keycode in (ord("m"), ord("M")):
            freeze_calibration_requested.set()

    print("\nViewer ready.")
    print(
        f"Phase 1: wait until all cameras are calibrated from PnP tags {list(pnp_tag_ids)}."
    )
    print(
        f"Press M in the MuJoCo viewer, or Space/S/M in a live capture window, to freeze calibration. "
        f"Detected AprilTags are rendered as labeled spheres directly in the MuJoCo 3D viewer."
    )
    if args.show_live_capture:
        print("Live camera preview windows are enabled.\n")
    else:
        print("Live camera preview windows are disabled.\n")

    if args.no_viewer:
        if not args.show_live_capture:
            print("--no-viewer enabled; forcing live capture windows on.\n")

        try:
            while True:
                live_ready = all_cameras_live(states)
                calibration_ready = calibration_locked or live_ready

                if live_ready and not calibration_ready_announced:
                    calibration_ready_announced = True
                    print(
                        "\nCalibration ready for all cameras. Press Space/S/M in a live capture window to freeze.\n"
                    )

                if update_live_capture_windows(states, calibration_locked):
                    freeze_calibration_requested.set()

                if freeze_calibration_requested.is_set():
                    freeze_calibration_requested.clear()
                    saved_count = freeze_calibration(states, pnp_tag_ids, pts)
                    if saved_count > 0:
                        calibration_locked = True
                        next_tag_print_time = 0.0
                        print(
                            f"\nCalibration frozen with {saved_count} camera pose(s). Logging AprilTag world positions; camera motion is now ignored.\n"
                        )
                    else:
                        print(
                            "\nSave ignored: no valid live camera poses are available yet.\n"
                        )

                if calibration_locked:
                    now = time.monotonic()
                    if now >= next_tag_print_time:
                        next_tag_print_time = now + tag_print_interval_s
                        per_cam = gather_tag_world_positions(states, prefer_stored=True)
                        print()
                        print_tag_world_positions(per_cam)
                        print_aligned_lines(
                            compute_aligned_lines(per_cam, start_id=args.start_id)
                        )

                parts: List[str] = []
                for st in states:
                    with st.lock:
                        lp = st.p_w_c
                        rp = st.reproj
                        nv = st.n_visible
                        distance_compare = dict(getattr(st, "distance_compare", {}))
                        fixed = (
                            st.stored_p is not None
                            and st.stored_R is not None
                            and calibration_locked
                        )
                    if fixed:
                        parts.append(f"[{st.name}]fixed")
                    elif lp is not None:
                        rp_text = f"{rp:.2f}" if math.isfinite(rp) else "nan"
                        dist_text = format_distance_summary(distance_compare).replace(
                            "dist check: ", ""
                        )
                        parts.append(
                            f"[{st.name}]cal ok rp={rp_text} {nv}/4 {dist_text}"
                        )
                    else:
                        parts.append(f"[{st.name}]cal wait {nv}/4")
                if calibration_locked:
                    phase = "tag-logging"
                elif calibration_ready:
                    phase = "ready-to-freeze"
                else:
                    phase = "calibrating"
                print(
                    f"\r  phase={phase}  {' | '.join(parts)}   ",
                    end="",
                    flush=True,
                )
                time.sleep(0.005)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=1.0)
            for st in states:
                st.cap.release()
            cv2.destroyAllWindows()
            print("\nDone.")
        return

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = t_w_g
        viewer.cam.distance = 12.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            live_ready = all_cameras_live(states)
            calibration_ready = calibration_locked or live_ready

            if live_ready and not calibration_ready_announced:
                calibration_ready_announced = True
                print(
                    "\nCalibration ready for all cameras. Press M to freeze and start logging AprilTag world positions.\n"
                )

            if freeze_calibration_requested.is_set():
                freeze_calibration_requested.clear()
                saved_count = freeze_calibration(states, pnp_tag_ids, pts)
                if saved_count > 0:
                    calibration_locked = True
                    next_tag_print_time = 0.0
                    print(
                        f"\nCalibration frozen with {saved_count} camera pose(s). Logging AprilTag world positions; camera motion is now ignored.\n"
                    )
                else:
                    print(
                        "\nM ignored: no valid live camera poses are available yet.\n"
                    )

            if args.show_live_capture and update_live_capture_windows(
                states, calibration_locked
            ):
                freeze_calibration_requested.set()

            viewer.user_scn.ngeom = 0

            for st in states:
                palette = _CAM_PALETTES[st.idx % len(_CAM_PALETTES)]
                with st.lock:
                    lp = st.p_w_c
                    lR = st.R_w_c
                    sp = st.stored_p
                    sR = st.stored_R
                if not calibration_locked and lp is not None and lR is not None:
                    draw_camera_geom(
                        viewer,
                        lp,
                        lR,
                        st.K,
                        st.actual_w,
                        st.actual_h,
                        args.axis_len,
                        args.frustum_depth,
                        palette,
                        alpha_scale=1.0,
                    )
                if sp is not None and sR is not None:
                    draw_camera_geom(
                        viewer,
                        sp,
                        sR,
                        st.K,
                        st.actual_w,
                        st.actual_h,
                        args.axis_len,
                        args.frustum_depth,
                        palette,
                        alpha_scale=(1.0 if calibration_locked else 0.28),
                    )

            tag_positions_per_cam = gather_tag_world_positions(
                states, prefer_stored=calibration_locked
            )
            aligned_lines = compute_aligned_lines(
                tag_positions_per_cam, start_id=args.start_id
            )
            draw_aligned_lines_in_viewer(viewer, aligned_lines)
            draw_brick_models_in_viewer(viewer, aligned_lines)
            draw_tag_markers_in_viewer(viewer, tag_positions_per_cam)

            now = time.monotonic()
            if now >= next_tag_print_time:
                next_tag_print_time = now + tag_print_interval_s
                print()
                print_aligned_lines(aligned_lines)

            viewer.sync()

            parts: List[str] = []
            for st in states:
                with st.lock:
                    lp = st.p_w_c
                    rp = st.reproj
                    nv = st.n_visible
                    distance_compare = dict(getattr(st, "distance_compare", {}))
                if (
                    st.stored_p is not None
                    and st.stored_R is not None
                    and calibration_locked
                ):
                    parts.append(f"[{st.name}]fixed")
                elif lp is not None:
                    rp_text = f"{rp:.2f}" if math.isfinite(rp) else "nan"
                    dist_text = format_distance_summary(distance_compare).replace(
                        "dist check: ", ""
                    )
                    parts.append(f"[{st.name}]cal ok rp={rp_text} {nv}/4 {dist_text}")
                else:
                    parts.append(f"[{st.name}]cal wait {nv}/4")
            if calibration_locked:
                phase = "tag-markers"
            elif calibration_ready:
                phase = "ready-to-freeze"
            else:
                phase = "calibrating"
            print(
                f"\r  phase={phase}  {' | '.join(parts)}   ",
                end="",
                flush=True,
            )

    stop_event.set()
    for t in threads:
        t.join(timeout=1.0)
    for st in states:
        st.cap.release()
    if args.show_live_capture:
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
