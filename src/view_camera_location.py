from __future__ import annotations

"""
Visualize predicted camera locations for up to 4 cameras in a MuJoCo viewer.
On SPACE: saves calibration files AND reconstructs brick positions from all
cameras, drawing them in the 3-D scene. Turntable tags (296-299) are
automatically excluded from brick reconstruction.

Each camera runs a background thread (capture + AprilTag pose estimation).
Reconstruction is single-shot from the main thread when SPACE is pressed.
"""

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from pupil_apriltags import Detector

try:
    from .calibration_core import (
        TURNTABLE_TAG_IDS,
        build_square_frame,
        get_tag_center,
        solve_camera_pose_from_square_centers,
    )
    from .apriltag_utils import detect_apriltags_silent
    from .camera_protocol import add_camera_protocol_args, open_camera
    from .reconstruction_core import (
        CameraObsInput,
        ReconConfig,
        reconstruct_blocks_multi_cam,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibration_core import (
        TURNTABLE_TAG_IDS,
        build_square_frame,
        get_tag_center,
        solve_camera_pose_from_square_centers,
    )
    from apriltag_utils import detect_apriltags_silent
    from camera_protocol import add_camera_protocol_args, open_camera
    from reconstruction_core import (
        CameraObsInput,
        ReconConfig,
        reconstruct_blocks_multi_cam,
    )

S_CV2MJ = np.diag([1.0, -1.0, -1.0])
FIXED_CAMERA_K = np.array(
    [
        [730.760254, 0.0, 661.803199],
        [0.0, 716.756829, 366.82691],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Brick local tag positions, matching the reconstruction model.
BAR_HALF = (0.5, 0.5, 1.5)
EPS = 0.004
END_Z = BAR_HALF[2] + EPS
SIDE_XY = BAR_HALF[0] + EPS
TAG_LOCAL = [
    np.array([0.0, 0.0, END_Z], dtype=np.float64),
    np.array([0.0, 0.0, -END_Z], dtype=np.float64),
    np.array([SIDE_XY, 0.0, 1.0], dtype=np.float64),
    np.array([SIDE_XY, 0.0, -1.0], dtype=np.float64),
    np.array([-SIDE_XY, 0.0, 1.0], dtype=np.float64),
    np.array([-SIDE_XY, 0.0, -1.0], dtype=np.float64),
    np.array([0.0, SIDE_XY, 1.0], dtype=np.float64),
    np.array([0.0, SIDE_XY, -1.0], dtype=np.float64),
    np.array([0.0, -SIDE_XY, 1.0], dtype=np.float64),
    np.array([0.0, -SIDE_XY, -1.0], dtype=np.float64),
]

_CAM_PALETTES = [
    (
        (1.0, 0.25, 0.20, 0.80),
        (1.0, 0.25, 0.20, 0.90),
        (0.20, 1.00, 0.20, 0.90),
        (0.20, 0.45, 1.00, 0.90),
        (1.00, 0.95, 0.20, 0.45),
    ),
    (
        (0.20, 0.45, 1.00, 0.80),
        (0.20, 0.45, 1.00, 0.90),
        (0.20, 1.00, 0.85, 0.90),
        (1.00, 0.55, 0.10, 0.90),
        (0.25, 0.80, 1.00, 0.45),
    ),
    (
        (0.20, 0.85, 0.20, 0.80),
        (0.20, 0.85, 0.20, 0.90),
        (0.90, 0.90, 0.20, 0.90),
        (0.85, 0.20, 0.85, 0.90),
        (0.20, 0.85, 0.50, 0.45),
    ),
    (
        (0.95, 0.50, 0.10, 0.80),
        (0.95, 0.50, 0.10, 0.90),
        (1.00, 0.90, 0.20, 0.90),
        (0.65, 0.20, 0.90, 0.90),
        (0.95, 0.65, 0.20, 0.45),
    ),
]

_BRICK_RGBA = (0.72, 0.45, 0.18, 0.70)
_DEBUG_BRICK_RGBA = (0.20, 0.80, 1.00, 0.30)


def _init_geom(viewer, gtype, size, pos, R, rgba) -> None:
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        gtype,
        np.asarray(size, dtype=np.float32),
        np.asarray(pos, dtype=np.float32),
        np.asarray(R, dtype=np.float32).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.matid = -1
    try:
        g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    except Exception:
        pass
    scn.ngeom += 1


def _add_box(viewer, pos, R, half=(0.12, 0.09, 0.07), rgba=(1, 0, 0, 0.7)):
    _init_geom(viewer, mujoco.mjtGeom.mjGEOM_BOX, half, pos, R, rgba)


def _add_sphere(viewer, pos, r=0.07, rgba=(1, 1, 0, 0.9)):
    _init_geom(viewer, mujoco.mjtGeom.mjGEOM_SPHERE, [r, r, r], pos, np.eye(3), rgba)


def _add_cylinder(
    viewer, a: np.ndarray, b: np.ndarray, radius: float = 0.03, rgba=(1, 0, 0, 0.9)
) -> None:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    diff = b - a
    L = float(np.linalg.norm(diff))
    if L < 1e-6:
        return
    mid = (a + b) * 0.5
    z = diff / L
    x = np.array([1.0, 0.0, 0.0])
    if abs(z @ x) > 0.9:
        x = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    _init_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [radius, radius, L * 0.5],
        mid,
        np.stack([x, y, z], axis=1),
        rgba,
    )


def draw_camera_geom(
    viewer,
    p: np.ndarray,
    R: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    axis_len: float,
    frustum_depth: float,
    palette: tuple,
    alpha_scale: float = 1.0,
) -> None:
    body_c, fwd_c, up_c, right_c, frust_c = palette

    def _a(c):
        return (*c[:3], c[3] * alpha_scale)

    _add_box(viewer, p, R, rgba=_a(body_c))

    fwd = R @ np.array([0.0, 0.0, -1.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    right = R @ np.array([1.0, 0.0, 0.0])

    _add_cylinder(viewer, p, p + axis_len * fwd, 0.030, _a(fwd_c))
    _add_cylinder(viewer, p, p + axis_len * 0.6 * up, 0.022, _a(up_c))
    _add_cylinder(viewer, p, p + axis_len * 0.6 * right, 0.022, _a(right_c))
    _add_sphere(viewer, p + axis_len * fwd, r=0.07, rgba=_a(fwd_c))

    Ki = np.linalg.inv(K)
    corners_px = [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]
    cw = []
    for u, v in corners_px:
        d = Ki @ np.array([u, v, 1.0])
        d /= np.linalg.norm(d)
        dw = R @ (S_CV2MJ @ d)
        dw /= np.linalg.norm(dw)
        far = p + frustum_depth * dw
        cw.append(far)
        _add_cylinder(viewer, p, far, 0.015, _a(frust_c))
    for i in range(4):
        _add_cylinder(viewer, cw[i], cw[(i + 1) % 4], 0.010, _a(frust_c))


def draw_brick(viewer, pos: np.ndarray, R: np.ndarray, rgba=_BRICK_RGBA) -> None:
    _add_box(viewer, pos, R, half=BAR_HALF, rgba=rgba)


@dataclass
class CamState:
    idx: int
    name: str
    cap: cv2.VideoCapture
    K: np.ndarray
    dist_map1: Optional[np.ndarray]
    dist_map2: Optional[np.ndarray]
    detector: Detector
    R_w_g: np.ndarray
    t_w_g: np.ndarray
    tag_size: float
    actual_w: int
    actual_h: int
    calib_out: str
    saved_K: np.ndarray
    saved_dist: np.ndarray

    lock: threading.Lock = field(default_factory=threading.Lock)
    R_w_c: Optional[np.ndarray] = None
    p_w_c: Optional[np.ndarray] = None
    reproj: float = float("nan")
    n_visible: int = 0
    stored_R: Optional[np.ndarray] = None
    stored_p: Optional[np.ndarray] = None
    latest_frame_bgr: Optional[np.ndarray] = None


def _camera_worker(state: CamState, stop_event: threading.Event) -> None:
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
        dets = detect_apriltags_silent(
            state.detector,
            gray,
            estimate_tag_pose=False,
            camera_params=(fx, fy, cx, cy),
            tag_size=float(state.tag_size),
        )

        centers: dict[int, np.ndarray] = {}
        for det in dets:
            tid = int(getattr(det, "tag_id", -1))
            if tid in TURNTABLE_TAG_IDS:
                centers[tid] = get_tag_center(det)

        R_est = None
        p_est = None
        reproj_val = float("nan")
        if len(centers) == 4:
            img_centers = np.array(
                [centers[tid] for tid in TURNTABLE_TAG_IDS], dtype=np.float64
            )
            try:
                R_est, p_est, rp = solve_camera_pose_from_square_centers(
                    img_centers, state.K, state.R_w_g, state.t_w_g
                )
                reproj_val = float(rp[0])
            except Exception:
                pass

        with state.lock:
            state.R_w_c = R_est
            state.p_w_c = p_est
            state.reproj = reproj_val
            state.n_visible = len(centers)
            state.latest_frame_bgr = frame.copy()


def save_calib(state: CamState) -> bool:
    with state.lock:
        if state.p_w_c is None or state.R_w_c is None:
            return False
        p = state.p_w_c.copy()
        R = state.R_w_c.copy()
        state.stored_p = p.copy()
        state.stored_R = R.copy()

    np.savez(
        state.calib_out,
        K=state.saved_K,
        dist_coeffs=state.saved_dist,
        cam_pos_w=p,
        R_w_c=R,
    )
    print(
        f"\n  [{state.name}] calib saved -> {state.calib_out}  pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
    )
    return True


def run_reconstruction(
    states: List[CamState],
    recon_detector: Detector,
    recon_cfg: ReconConfig,
    scale: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    cam_obs: List[CameraObsInput] = []
    for st in states:
        with st.lock:
            frame_bgr = st.latest_frame_bgr
            p = st.stored_p if st.stored_p is not None else st.p_w_c
            R = st.stored_R if st.stored_R is not None else st.R_w_c

        if frame_bgr is None:
            print(f"  [{st.name}] no frame yet, skipping")
            continue
        if p is None or R is None:
            print(f"  [{st.name}] no pose estimate yet, skipping")
            continue

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cam_obs.append(
            CameraObsInput(
                cam_id=st.idx,
                cam_name=st.name,
                rgb=rgb,
                K=st.K,
                cam_pos_w=p,
                R_w_c=R,
            )
        )

    if not cam_obs:
        print("  No cameras available for reconstruction.")
        return {}

    result = reconstruct_blocks_multi_cam(
        cams=cam_obs,
        detector=recon_detector,
        tag_local=[tl.copy() for tl in TAG_LOCAL],
        gt_tag_world=None,
        gt_block_pose=None,
        config=recon_cfg,
    )

    fitted: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for bi, br in result.block_results.items():
        if br.R_w_b is not None and br.t_w_b is not None:
            t_mj = br.t_w_b * scale
            fitted[bi] = (t_mj, br.R_w_b)

    print(
        f"  Reconstruction: {len(result.det_rows)} tags detected, {len(fitted)} bricks OK"
    )
    for bi, (t, _) in sorted(fitted.items()):
        print(f"    brick{bi}: pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
    return fitted


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize camera locations + reconstruct bricks in MuJoCo viewer."
    )
    ap.add_argument("--xml", default="assets/scene_turntable_only_lowlookat.xml")
    ap.add_argument(
        "--cams",
        nargs="+",
        default=["0"],
        help="Camera device indices or paths (up to 4). On Debian, 0 maps to /dev/video0.",
    )
    add_camera_protocol_args(ap)
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
        help="Output .npz paths for SPACE-save (default: calib_cam{i}.npz)",
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
            -1,
            1,
            0.0,
            1,
            1,
            0.0,
            1,
            -1,
            0.0,
        ],
    )
    ap.add_argument("--tag-size", type=float, default=0.64)
    ap.add_argument("--real-tag-size", type=float, default=0.64)
    ap.add_argument("--mujoco-tag-size", type=float, default=0.64)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--enforce-flat", action="store_true", default=True)
    ap.add_argument("--no-enforce-flat", dest="enforce_flat", action="store_false")
    ap.add_argument("--coord", default="cv2mj_yz", choices=["identity", "cv2mj_yz"])
    ap.add_argument("--out-poses", default="poses_live.txt")
    ap.add_argument("--frustum-depth", type=float, default=2.5)
    ap.add_argument("--axis-len", type=float, default=1.2)
    return ap.parse_args()


def K_from_fovy(fovy_deg: float, w: int, h: int) -> np.ndarray:
    f = 0.5 * h / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array(
        [[f, 0, (w - 1) * 0.5], [0, f, (h - 1) * 0.5], [0, 0, 1.0]], dtype=np.float64
    )


def write_poses_txt(
    out_path: Path, fitted: Dict[int, Tuple[np.ndarray, np.ndarray]]
) -> None:
    def _quat_xyzw(R):
        tr = float(np.trace(R))
        if tr > 0:
            S = math.sqrt(tr + 1) * 2
            return np.array(
                [
                    (R[2, 1] - R[1, 2]) / S,
                    (R[0, 2] - R[2, 0]) / S,
                    (R[1, 0] - R[0, 1]) / S,
                    0.25 * S,
                ]
            )
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            return np.array(
                [
                    0.25 * S,
                    (R[0, 1] + R[1, 0]) / S,
                    (R[0, 2] + R[2, 0]) / S,
                    (R[2, 1] - R[1, 2]) / S,
                ]
            )
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            return np.array(
                [
                    (R[0, 1] + R[1, 0]) / S,
                    0.25 * S,
                    (R[1, 2] + R[2, 1]) / S,
                    (R[0, 2] - R[2, 0]) / S,
                ]
            )
        else:
            S = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            return np.array(
                [
                    (R[0, 2] + R[2, 0]) / S,
                    (R[1, 2] + R[2, 1]) / S,
                    0.25 * S,
                    (R[1, 0] - R[0, 1]) / S,
                ]
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = [(bi, fitted[bi][0], fitted[bi][1]) for bi in sorted(fitted)]
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{len(items)}\n")
        for bi, t, R in items:
            q = _quat_xyzw(R)
            q /= np.linalg.norm(q)
            if q[3] < 0:
                q = -q
            f.write(
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )
    print(f"  poses.txt -> {out_path.resolve()} ({len(items)} bricks)")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    n_cams = min(len(args.cams), 4)
    if n_cams == 0:
        sys.exit("ERROR: provide at least one camera via --cams")

    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = base_dir / xml_path
    if not xml_path.exists():
        sys.exit(f"ERROR: XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pts = np.array(args.world_points, dtype=np.float64).reshape(4, 3)
    R_w_g, t_w_g = build_square_frame(pts)
    print(f"G-frame origin: {t_w_g.tolist()}")

    scale = args.mujoco_tag_size / args.real_tag_size
    S_cv2mj = np.diag([1.0, -1.0, -1.0]) if args.coord == "cv2mj_yz" else np.eye(3)
    recon_cfg = ReconConfig(
        family="tag36h11",
        tag_size=args.real_tag_size,
        start_id=args.start_id,
        max_block_tag_id=295,
        S_cv2mj=S_cv2mj,
        enforce_flat=args.enforce_flat,
        save_overlay=False,
    )
    recon_detector = Detector(
        families="tag36h11", nthreads=2, quad_decimate=1.5, refine_edges=1
    )
    print(
        f"Scale: {args.real_tag_size * 1000:.0f}mm -> {args.mujoco_tag_size} units (x{scale:.1f}) coord={args.coord}"
    )

    states: List[CamState] = []
    for i in range(n_cams):
        cam_arg = args.cams[i]
        name = args.cam_names[i] if i < len(args.cam_names) else f"cam{i}"
        cal_in = args.calib_in[i] if i < len(args.calib_in) else None
        cal_out = args.calib_out[i] if i < len(args.calib_out) else f"calib_{name}.npz"

        try:
            opened = open_camera(
                cam_arg,
                width=args.width,
                height=args.height,
                protocol=args.camera_protocol,
                fourcc=args.camera_fourcc,
            )
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"ERROR: {exc}")

        cap = opened.cap
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        format_msg = f" fourcc={opened.fourcc}" if opened.fourcc else ""
        print(
            f"  [{name}] device={opened.source} backend={opened.backend_name}"
            f"{format_msg} {aw}x{ah}"
        )

        K = FIXED_CAMERA_K.copy()
        dist_coeffs = np.zeros(5, dtype=np.float64)
        saved_R = saved_p = None
        if cal_in:
            npz = np.load(cal_in)
            if "dist_coeffs" in npz:
                dist_coeffs = np.asarray(npz["dist_coeffs"], dtype=np.float64)
            if "cam_pos_w" in npz:
                saved_p = np.asarray(npz["cam_pos_w"], dtype=np.float64).reshape(3)
            if "R_w_c" in npz:
                saved_R = np.asarray(npz["R_w_c"], dtype=np.float64).reshape(3, 3)
            print(f"  [{name}] loaded pose/distortion from {cal_in}; using fixed K")
        else:
            print(f"  [{name}] using fixed K")

        m1 = m2 = None
        if np.any(dist_coeffs != 0):
            m1, m2 = cv2.initUndistortRectifyMap(
                K, dist_coeffs, None, K, (aw, ah), cv2.CV_16SC2
            )

        worker_detector = Detector(
            families="tag36h11", nthreads=2, quad_decimate=1.0, refine_edges=1
        )
        st = CamState(
            idx=i,
            name=name,
            cap=cap,
            K=K,
            dist_map1=m1,
            dist_map2=m2,
            detector=worker_detector,
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

    stop_event = threading.Event()
    threads = []
    for st in states:
        t = threading.Thread(target=_camera_worker, args=(st, stop_event), daemon=True)
        t.start()
        threads.append(t)

    fitted_bricks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    fitted_lock = threading.Lock()
    space_pressed = threading.Event()

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            space_pressed.set()

    print(f"\nViewer ready - {n_cams} camera(s). SPACE=save+reconstruct ESC=quit\n")
    out_poses_path = Path(args.out_poses)
    if not out_poses_path.is_absolute():
        out_poses_path = base_dir / out_poses_path

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = t_w_g
        viewer.cam.distance = 12.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            if space_pressed.is_set():
                space_pressed.clear()
                print("\n--- SPACE pressed ---")

                saved_any = False
                for st in states:
                    if save_calib(st):
                        saved_any = True
                if not saved_any:
                    print("  No valid camera poses to save yet.")

                new_fitted = run_reconstruction(
                    states, recon_detector, recon_cfg, scale
                )
                with fitted_lock:
                    fitted_bricks.clear()
                    fitted_bricks.update(new_fitted)

                if new_fitted:
                    write_poses_txt(out_poses_path, new_fitted)
                else:
                    print("  No bricks reconstructed - poses.txt not updated.")

            viewer.user_scn.ngeom = 0

            for st in states:
                palette = _CAM_PALETTES[st.idx % len(_CAM_PALETTES)]

                with st.lock:
                    sp = st.stored_p
                    sR = st.stored_R
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
                        alpha_scale=0.28,
                    )

                with st.lock:
                    lp = st.p_w_c
                    lR = st.R_w_c
                if lp is not None and lR is not None:
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

            with fitted_lock:
                bricks_snapshot = dict(fitted_bricks)
            for bi, (t_mj, R_b) in bricks_snapshot.items():
                draw_brick(viewer, t_mj, R_b, rgba=_DEBUG_BRICK_RGBA)

            viewer.sync()

            parts = []
            for st in states:
                with st.lock:
                    lp = st.p_w_c
                    rp = st.reproj
                    nv = st.n_visible
                if lp is not None:
                    parts.append(
                        f"[{st.name}]({lp[0]:.2f},{lp[1]:.2f},{lp[2]:.2f}) rp={rp:.2f} {nv}/4"
                    )
                else:
                    parts.append(f"[{st.name}]wait {nv}/4")
            nb = len(bricks_snapshot)
            print(f"\r  {'  |  '.join(parts)}  | bricks={nb}   ", end="", flush=True)

    stop_event.set()
    for t in threads:
        t.join(timeout=1.0)
    for st in states:
        st.cap.release()
    print("\nDone.")


if __name__ == "__main__":
    main()
