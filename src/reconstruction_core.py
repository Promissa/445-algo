# reconstruction_core.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import cv2

try:
    from .apriltag_utils import detect_apriltags_silent
except ImportError:
    from apriltag_utils import detect_apriltags_silent

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CameraObsInput:
    """
    输入：每个相机的一帧观测
    - cam_id: 用于标识 (也会写入 det_dump.csv)
    - cam_name: 可读名
    - rgb: (H,W,3) RGB uint8
    - K: (3,3) intrinsics
    - cam_pos_w: (3,) world
    - R_w_c: (3,3) world-from-camera rotation
    """
    cam_id: int
    cam_name: str
    rgb: np.ndarray
    K: np.ndarray
    cam_pos_w: np.ndarray
    R_w_c: np.ndarray


@dataclass
class ReconConfig:
    family: str = "tag36h11"
    tag_size: float = 0.8
    start_id: int = 0
    max_block_tag_id: int = 290

    # cv->mj camera coord transform (like your args.coord)
    # use diag matrix: cv point -> mj point
    S_cv2mj: np.ndarray = np.eye(3)

    # weights
    w_eps: float = 1e-4

    # constraints
    enforce_flat: bool = False
    enforce_z0: bool = False  # debug-only, needs gt block z

    # debug
    save_overlay: bool = False
    overlay_thickness: int = 2


@dataclass
class DetectionRow:
    cam: int
    id: int
    l_mean: float
    area_px: float
    area_norm: float
    tz: float
    cos_n: float
    reproj: float
    conf: float
    err: float
    ex: float
    ey: float
    ez: float


@dataclass
class BlockFitResult:
    block_index: int
    R_w_b: Optional[np.ndarray]   # (3,3)
    t_w_b: Optional[np.ndarray]   # (3,)
    rms: float
    obs_seen: int
    uniq_seen: int
    flat_axis: str

    # compare to GT if provided
    pos_err: Optional[float] = None
    rot_err_deg: Optional[float] = None


@dataclass
class ReconOutput:
    block_results: Dict[int, BlockFitResult]

    # debug points for viewer
    est_tag_centers_w: List[Tuple[np.ndarray, int, float]]  # (pos, cam_id, conf)
    fitted_origins_w: List[Tuple[np.ndarray, int]]          # (origin, block)
    ghost_boxes: List[Tuple[np.ndarray, np.ndarray, str]]   # (pos, R, style_key)
    ghost_tag_centers_w: List[np.ndarray]                   # tag points (ghost)

    det_rows: List[DetectionRow]
    per_cam_overlay_bgr: Dict[int, np.ndarray]  # only if save_overlay enabled


# ----------------------------
# Math helpers
# ----------------------------

def rot_err_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_est.T @ R_gt
    tr = float(np.trace(R))
    c = (tr - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def cv_point_to_world(p_cv: np.ndarray, S: np.ndarray, R_w_c: np.ndarray, cam_pos_w: np.ndarray) -> np.ndarray:
    p_mj = S @ p_cv
    return R_w_c @ p_mj + cam_pos_w


def is_block_tag_id(tid: int, config: ReconConfig) -> bool:
    return int(config.start_id) <= tid <= int(config.max_block_tag_id)


def tag_area_px(corners_uv: np.ndarray) -> float:
    c = np.asarray(corners_uv, dtype=np.float64).reshape(4, 2)
    return float(abs(cv2.contourArea(c.astype(np.float32))))


def weighted_kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray, w_eps: float = 1e-4):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    if P.shape[0] < 3:
        return None

    w = np.clip(w, 0.0, None)
    w = np.maximum(w, float(w_eps))
    sw = float(np.sum(w))
    if sw <= 1e-12:
        return None

    wn = w / sw
    p_bar = np.sum(P * wn[:, None], axis=0)
    q_bar = np.sum(Q * wn[:, None], axis=0)

    X = P - p_bar
    Y = Q - q_bar
    H = (X * wn[:, None]).T @ Y

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = q_bar - R @ p_bar

    res = (R @ P.T).T + t - Q
    rms = float(np.sqrt(np.sum(wn * np.sum(res * res, axis=1))))
    return R, t, rms, wn, p_bar, q_bar


def weighted_rms(P: np.ndarray, Q: np.ndarray, wn: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    res = (R @ P.T).T + t - Q
    return float(np.sqrt(np.sum(wn * np.sum(res * res, axis=1))))


def enforce_one_face_parallel_ground(R_w_b: np.ndarray, z_world=np.array([0., 0., 1.], dtype=np.float64)):
    axes = [
        np.array([ 1.,0.,0.], dtype=np.float64), np.array([-1.,0.,0.], dtype=np.float64),
        np.array([ 0.,1.,0.], dtype=np.float64), np.array([ 0.,-1.,0.], dtype=np.float64),
        np.array([ 0.,0.,1.], dtype=np.float64), np.array([ 0.,0.,-1.], dtype=np.float64),
    ]
    best = None
    for a in axes:
        v = R_w_b @ a
        s = float(np.dot(v, z_world))
        score = abs(s)
        if (best is None) or (score > best[0]):
            best = (score, a, v, s)

    _, a_local, v_world, s = best
    b_world = z_world if s >= 0 else -z_world

    v = v_world / (np.linalg.norm(v_world) + 1e-12)
    b = b_world / (np.linalg.norm(b_world) + 1e-12)

    axis = np.cross(v, b)
    sa = np.linalg.norm(axis)
    ca = float(np.dot(v, b))

    if sa < 1e-9:
        return R_w_b, a_local, (1 if s >= 0 else -1)

    axis = axis / sa
    angle = math.atan2(sa, ca)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    Rc = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    R_proj = Rc @ R_w_b
    return R_proj, a_local, (1 if s >= 0 else -1)


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def solve_two_points_with_flat(P2, Q2, z_world=np.array([0.,0.,1.], dtype=np.float64)):
    P2 = np.asarray(P2, dtype=np.float64).reshape(2,3)
    Q2 = np.asarray(Q2, dtype=np.float64).reshape(2,3)

    vp = _normalize(P2[1] - P2[0])
    vq = _normalize(Q2[1] - Q2[0])
    if vp is None or vq is None:
        return None

    axes = [
        np.array([ 1.,0.,0.]), np.array([-1.,0.,0.]),
        np.array([ 0.,1.,0.]), np.array([ 0.,-1.,0.]),
        np.array([ 0.,0.,1.]), np.array([ 0.,0.,-1.]),
    ]
    best = None

    for a_local in axes:
        for sign in (+1, -1):
            b_world = sign * z_world

            z0 = _normalize(a_local)
            if z0 is None:
                continue

            vp_perp = vp - np.dot(vp, z0) * z0
            x0 = _normalize(vp_perp)
            if x0 is None:
                continue
            y0 = np.cross(z0, x0)

            zw = _normalize(b_world)
            vq_perp = vq - np.dot(vq, zw) * zw
            xw = _normalize(vq_perp)
            if xw is None:
                continue
            yw = np.cross(zw, xw)

            R = np.stack([xw, yw, zw], axis=1) @ np.stack([x0, y0, z0], axis=1).T
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            t = Q2[0] - R @ P2[0]

            Qhat = (R @ P2.T).T + t
            err = Qhat - Q2
            score = float(np.sum(err * err))

            if (best is None) or (score < best[-1]):
                best = (R, t, a_local, sign, score)

    return best


# ----------------------------
# Confidence (keep same)
# ----------------------------

def tag_confidence(corners_uv: np.ndarray, R_c_t: np.ndarray, t_c_t: np.ndarray, K: np.ndarray,
                   img_w: int, img_h: int,
                   l0: float = 80.0, cmin: float = 0.35, p: float = 3.0, alpha: float = 1.5) -> float:
    corners = np.asarray(corners_uv, dtype=np.float64).reshape(4, 2)

    edges = np.array([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)], dtype=np.float64)
    l_mean = float(edges.mean())
    w_size = float(np.clip((l_mean / l0) ** 2, 0.0, 1.0))

    n_c = R_c_t @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(abs(n_c[2]))
    w_tilt = float(np.clip((c - cmin) / (1.0 - cmin), 0.0, 1.0) ** p)

    cx, cy = float(K[0, 2]), float(K[1, 2])
    uv = corners.mean(axis=0)
    rx = (float(uv[0]) - cx) / (0.5 * img_w)
    ry = (float(uv[1]) - cy) / (0.5 * img_h)
    r2 = rx * rx + ry * ry
    w_center = float(np.exp(-alpha * r2))

    tz = float(t_c_t[2])
    w_dist = 1.0 if tz < 8.0 else float(np.exp(-0.15 * (tz - 8.0)))

    return float(np.clip(w_size * w_tilt * w_center * w_dist, 0.0, 1.0))


# ----------------------------
# pupil detection pose getter
# ----------------------------

def get_pupil_pose(det) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    for (rname, tname) in [("pose_R", "pose_t"), ("R", "t")]:
        if hasattr(det, rname) and hasattr(det, tname):
            R = np.asarray(getattr(det, rname), dtype=np.float64)
            t = np.asarray(getattr(det, tname), dtype=np.float64).reshape(3)
            if R.shape == (3, 3) and t.shape == (3,):
                return R, t
    return None


# ----------------------------
# Main reconstruction function
# ----------------------------

def reconstruct_blocks_multi_cam(
    cams: List[CameraObsInput],
    detector,                         # pupil_apriltags.Detector instance
    tag_local: List[np.ndarray],      # length 10
    gt_tag_world: Optional[Dict[int, np.ndarray]] = None,
    gt_block_pose: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,  # bi -> (p_gt, R_gt)
    config: ReconConfig = ReconConfig(),
    logger: Optional[logging.Logger] = None,
) -> ReconOutput:
    """
    输入：
      - cams: 每个相机一帧的 RGB + (K, R_w_c, cam_pos_w)
      - detector: pupil_apriltags Detector
      - tag_local: block frame 下 10 个 tag center
      - gt_tag_world: (optional) tid -> p_gt for per-det error debug
      - gt_block_pose: (optional) bi -> (p_gt, R_gt) for final compare
      - config: see ReconConfig
    输出：
      - ReconOutput: 每个 block 的拟合 R,t + debug rows + overlay
    """
    if logger is None:
        logger = logging.getLogger("recon")

    per_block_P: Dict[int, List[np.ndarray]] = {}
    per_block_Q: Dict[int, List[np.ndarray]] = {}
    per_block_w: Dict[int, List[float]] = {}
    stats_seen_obs: Dict[int, int] = {}
    stats_seen_tags: Dict[int, Set[int]] = {}

    det_rows: List[DetectionRow] = []
    per_cam_overlay: Dict[int, np.ndarray] = {}

    est_tag_centers_w: List[Tuple[np.ndarray, int, float]] = []
    fitted_origins_w: List[Tuple[np.ndarray, int]] = []
    ghost_boxes: List[Tuple[np.ndarray, np.ndarray, str]] = []
    ghost_tag_centers_w: List[np.ndarray] = []

    # -------------- per camera detect --------------
    for cam in cams:
        rgb = cam.rgb
        H, W = rgb.shape[:2]
        K = cam.K
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        dets = detect_apriltags_silent(
            detector,
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=float(config.tag_size),
        )
        logger.info(f"[cam {cam.cam_id} {cam.cam_name}] dets={len(dets)} W,H={W},{H} fx={fx:.1f}")

        overlay = None
        if config.save_overlay:
            overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

        for det in dets:
            tid = int(getattr(det, "tag_id", -1))
            if not is_block_tag_id(tid, config):
                continue

            pose = get_pupil_pose(det)
            if pose is None:
                continue
            R_c_t, t_c_t = pose

            bi = (tid - int(config.start_id)) // 10
            k = (tid - int(config.start_id)) % 10
            if not (0 <= k < len(tag_local)):
                continue

            stats_seen_obs[bi] = stats_seen_obs.get(bi, 0) + 1
            stats_seen_tags.setdefault(bi, set()).add(k)

            p_est_w = cv_point_to_world(t_c_t, config.S_cv2mj, cam.R_w_c, cam.cam_pos_w)

            corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
            edges = np.array([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)], dtype=np.float64)
            l_mean = float(edges.mean())
            area_px = tag_area_px(corners)
            area_norm = area_px / float(W * H)

            n_c = R_c_t @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            cos_n = float(abs(n_c[2]))
            tz = float(t_c_t[2])

            # reprojection error
            try:
                s = float(config.tag_size)
                half = 0.5 * s
                obj = np.array([[-half, -half, 0.0],
                                [ half, -half, 0.0],
                                [ half,  half, 0.0],
                                [-half,  half, 0.0]], dtype=np.float64)
                rvec, _ = cv2.Rodrigues(R_c_t)
                tvec = np.asarray(t_c_t, dtype=np.float64).reshape(3, 1)
                dist = np.zeros((4, 1), dtype=np.float64)
                proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                proj = proj.reshape(-1, 2)
                reproj = float(np.mean(np.linalg.norm(proj - corners, axis=1)))
            except Exception:
                reproj = float("nan")

            conf = tag_confidence(corners, R_c_t, t_c_t, K, W, H)

            # GT tag error (optional)
            if gt_tag_world is not None and tid in gt_tag_world:
                p_gt = gt_tag_world[tid]
                e = (p_est_w - p_gt).astype(np.float64)
                e[2] += 1
                ex, ey, ez = float(e[0]), float(e[1]), float(e[2])
                err = float(np.linalg.norm(e))
            else:
                err = float("nan")
                ex = ey = ez = float("nan")

            det_rows.append(DetectionRow(
                cam=cam.cam_id, id=tid,
                l_mean=l_mean, area_px=area_px, area_norm=area_norm,
                tz=tz, cos_n=cos_n, reproj=reproj, conf=conf,
                err=err, ex=ex, ey=ey, ez=ez
            ))

            # keep all for fitting
            per_block_P.setdefault(bi, []).append(tag_local[k])
            per_block_Q.setdefault(bi, []).append(p_est_w)
            per_block_w.setdefault(bi, []).append(conf)

            est_tag_centers_w.append((p_est_w, cam.cam_id, conf))

            if overlay is not None:
                uv = corners.mean(axis=0)
                cv2.polylines(overlay, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), config.overlay_thickness)
                cv2.putText(overlay, f"id={tid} k={k} c={conf:.2f}",
                            (int(uv[0]) + 4, int(uv[1]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if overlay is not None:
            per_cam_overlay[cam.cam_id] = overlay

    # -------------- per block fit --------------
    block_results: Dict[int, BlockFitResult] = {}
    all_blocks = sorted(set(list(per_block_P.keys()) + list(stats_seen_obs.keys())))

    for bi in all_blocks:
        P_list = per_block_P.get(bi, [])
        Q_list = per_block_Q.get(bi, [])
        w_list = per_block_w.get(bi, [])
        n = len(P_list)

        obs_seen = stats_seen_obs.get(bi, 0)
        uniq_seen = len(stats_seen_tags.get(bi, set()))

        if n == 0:
            block_results[bi] = BlockFitResult(
                block_index=bi, R_w_b=None, t_w_b=None,
                rms=float("nan"), obs_seen=obs_seen, uniq_seen=uniq_seen, flat_axis="-"
            )
            continue

        P = np.stack(P_list, axis=0)
        Q = np.stack(Q_list, axis=0)
        w = np.array(w_list, dtype=np.float64)

        R_fit = None
        t_fit = None
        rms = float("nan")
        flat_axis = "-"

        if n >= 3:
            out = weighted_kabsch(P, Q, w, w_eps=float(config.w_eps))
            if out is None:
                block_results[bi] = BlockFitResult(
                    block_index=bi, R_w_b=None, t_w_b=None,
                    rms=float("nan"), obs_seen=obs_seen, uniq_seen=uniq_seen, flat_axis="-"
                )
                continue

            R_fit, t_fit, rms, wn, p_bar, q_bar = out

            if config.enforce_flat:
                R_proj, a_local, sign = enforce_one_face_parallel_ground(R_fit)
                t_proj = q_bar - R_proj @ p_bar
                rms_proj = weighted_rms(P, Q, wn, R_proj, t_proj)
                R_fit, t_fit, rms = R_proj, t_proj, rms_proj
                flat_axis = f"kabsch {a_local.tolist()}*{sign}"
        else:
            if not config.enforce_flat:
                block_results[bi] = BlockFitResult(
                    block_index=bi, R_w_b=None, t_w_b=None,
                    rms=float("nan"), obs_seen=obs_seen, uniq_seen=uniq_seen,
                    flat_axis="(need enforce_flat for n<3)"
                )
                continue

            if n == 1:
                R0 = np.eye(3, dtype=np.float64)
                R_fit, a_local, sign = enforce_one_face_parallel_ground(R0)
                t_fit = Q[0] - R_fit @ P[0]
                rms = 0.0
                flat_axis = f"n1 {a_local.tolist()}*{sign}"
            else:
                best = solve_two_points_with_flat(P[:2], Q[:2])
                if best is None:
                    block_results[bi] = BlockFitResult(
                        block_index=bi, R_w_b=None, t_w_b=None,
                        rms=float("nan"), obs_seen=obs_seen, uniq_seen=uniq_seen,
                        flat_axis="(fallback failed)"
                    )
                    continue
                R_fit, t_fit, a_local, sign, _score = best
                Qhat = (R_fit @ P[:2].T).T + t_fit
                rms = float(np.sqrt(np.mean(np.sum((Qhat - Q[:2]) ** 2, axis=1))))
                flat_axis = f"n2 {a_local.tolist()}*{sign}"

        if R_fit is None or t_fit is None:
            block_results[bi] = BlockFitResult(
                block_index=bi, R_w_b=None, t_w_b=None,
                rms=float("nan"), obs_seen=obs_seen, uniq_seen=uniq_seen, flat_axis="-"
            )
            continue

        # optional clamp z to GT z
        if config.enforce_z0 and gt_block_pose is not None and bi in gt_block_pose:
            z_gt = float(gt_block_pose[bi][0][2])
            t_fit = t_fit.copy()
            t_fit[2] = z_gt

        # compare with GT block pose if exists
        pos_err = None
        ang_err = None
        if gt_block_pose is not None and bi in gt_block_pose:
            p_gt, R_gt = gt_block_pose[bi]
            pos_err = float(np.linalg.norm(t_fit - p_gt))
            ang_err = rot_err_deg(R_fit, R_gt)

        block_results[bi] = BlockFitResult(
            block_index=bi, R_w_b=R_fit, t_w_b=t_fit, rms=rms,
            obs_seen=obs_seen, uniq_seen=uniq_seen, flat_axis=flat_axis,
            pos_err=pos_err, rot_err_deg=ang_err
        )

        fitted_origins_w.append((t_fit, bi))

    return ReconOutput(
        block_results=block_results,
        est_tag_centers_w=est_tag_centers_w,
        fitted_origins_w=fitted_origins_w,
        ghost_boxes=ghost_boxes,
        ghost_tag_centers_w=ghost_tag_centers_w,
        det_rows=det_rows,
        per_cam_overlay_bgr=per_cam_overlay
    )
