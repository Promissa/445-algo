from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import mujoco
import numpy as np

TURNTABLE_TAG_IDS = (300, 301, 302, 303)
S_CV2MJ = np.diag([1.0, -1.0, -1.0])


def K_from_fovy_mujoco(fovy_deg: float, w: int, h: int) -> np.ndarray:
    fovy = np.deg2rad(float(fovy_deg))
    f = 0.5 * h / np.tan(0.5 * fovy)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def cam_id_from_arg(model: mujoco.MjModel, cam_arg: str) -> int:
    cam_arg = cam_arg.strip()
    try:
        cid = int(cam_arg)
    except ValueError:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_arg)
    if cid < 0 or cid >= model.ncam:
        raise ValueError(f"Invalid camera: {cam_arg}")
    return cid


def get_tag_center(det) -> np.ndarray:
    if hasattr(det, "center"):
        return np.asarray(det.center, dtype=np.float64).reshape(2)
    corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
    return corners.mean(axis=0)


def render_camera(
    model: mujoco.MjModel, data: mujoco.MjData, cam_id: int, w: int, h: int
) -> np.ndarray:
    renderer = mujoco.Renderer(model, width=w, height=h)
    try:
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render()
    finally:
        renderer.close()
    return rgb


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def rot_err_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_est.T @ R_gt
    tr = float(np.trace(R))
    c = (tr - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def build_square_frame(world_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the G frame from the four tag centers.
    The coordinates in G are [-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0].
    """
    p0, p1, p2, p3 = [np.asarray(p, dtype=np.float64).reshape(3) for p in world_points]
    origin = np.mean(np.stack([p0, p1, p2, p3], axis=0), axis=0)
    x_axis = 0.5 * (p1 - p0)
    y_axis = 0.5 * (p3 - p0)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R_w_g = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R_w_g, origin


def solve_camera_pose_from_square_centers(
    centers_uv: np.ndarray,
    K: np.ndarray,
    R_w_g: np.ndarray,
    t_w_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve pose using only tag centers.
    Returns:
      R_w_c_mj, p_w_c, reprojection_errors
    """
    obj_g = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    img = np.asarray(centers_uv, dtype=np.float64).reshape(4, 2)

    try:
        ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
            obj_g,
            img,
            K,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except Exception:
        ok = False
        rvecs = []
        tvecs = []
        reproj = []

    candidates: List[Tuple[np.ndarray, np.ndarray, float]] = []
    if ok and rvecs:
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            rv = np.asarray(rvec, dtype=np.float64).reshape(3)
            tv = np.asarray(tvec, dtype=np.float64).reshape(3)
            err = (
                float(np.asarray(reproj[i]).reshape(-1)[0]) if i < len(reproj) else 0.0
            )
            candidates.append((rv, tv, err))
    else:
        ok2, rvec, tvec = cv2.solvePnP(
            obj_g,
            img,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok2:
            candidates.append(
                (
                    np.asarray(rvec, dtype=np.float64).reshape(3),
                    np.asarray(tvec, dtype=np.float64).reshape(3),
                    0.0,
                )
            )

    if not candidates:
        raise RuntimeError("solvePnP failed for the four turntable centers")

    best = None
    for rvec, tvec, reproj_err in candidates:
        R_c_g, _ = cv2.Rodrigues(rvec)
        R_c_w = R_c_g @ R_w_g.T
        t_c_w = tvec - R_c_g @ R_w_g.T @ t_w_g
        R_w_c_cv = R_c_w.T
        R_w_c_mj = R_w_c_cv @ S_CV2MJ
        p_w_c = -R_w_c_cv @ t_c_w
        center_w = t_w_g
        forward_w = R_w_c_mj @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        facing = float(np.dot(forward_w, center_w - p_w_c))
        above = float(p_w_c[2] - t_w_g[2])
        score = float(reproj_err) - 0.05 * facing - 0.01 * above
        if (best is None) or (score < best[0]):
            best = (
                score,
                R_w_c_mj,
                p_w_c,
                np.array([float(reproj_err)], dtype=np.float64),
            )

    assert best is not None
    _, R_w_c_mj, p_w_c, reproj_err = best
    return R_w_c_mj, p_w_c, reproj_err
