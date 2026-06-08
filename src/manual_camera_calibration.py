from __future__ import annotations

"""
Manual single-camera calibration using the turntable AprilTags.

This is the calibration-only path from view_camera_location.py:
  - open one physical camera
  - detect turntable tags in the live image
  - solve the camera pose from the four tag centers
  - press Space or S to save a calibration .npz

Keys:
  Space / s : save current valid calibration
  q / ESC   : quit
"""

import argparse
import time
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live single-camera AprilTag turntable calibration."
    )
    ap.add_argument("--cam", required=True, help="Camera device index or path, e.g. 0")
    ap.add_argument(
        "--name", default="", help="Camera name used for the default output filename"
    )
    ap.add_argument(
        "--calib-in", default="", help="Optional .npz with K and dist_coeffs"
    )
    ap.add_argument("--calib-out", default="", help="Output .npz path")
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
            -1.0,
            -1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            -1.0,
            0.0,
        ],
        help="World points for the four tags, same order as --tag-ids",
    )
    ap.add_argument("--decimate", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=0.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cam_name = args.name or f"cam{args.cam}"
    out_path = Path(args.calib_out or f"calib_{cam_name}.npz")
    tag_ids = tuple(int(tid) for tid in args.tag_ids)
    tag_id_set = set(tag_ids)

    K, dist_coeffs = load_intrinsics(args.calib_in or None)
    world_points = np.asarray(args.world_points, dtype=np.float64).reshape(4, 3)
    R_w_g, t_w_g = build_square_frame(world_points)

    cap = open_camera(parse_cam_arg(args.cam), width=args.width, height=args.height)

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

    print(f"camera={args.cam} actual={actual_w}x{actual_h}")
    print(f"K=\n{K}")
    print(f"target tags={tag_ids}")
    print("keys: Space/S save current valid calibration, Q/ESC quit")
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    win = f"manual calibration [{cam_name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(actual_w, 1280), min(actual_h, 720))

    last_pose: tuple[np.ndarray, np.ndarray, float, DistanceCompare] | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=float(args.tag_size),
        )
        centers: Dict[int, np.ndarray] = {}
        detections_by_id: Dict[int, object] = {}
        for det in detections:
            tid = int(getattr(det, "tag_id", -1))
            if tid in tag_id_set:
                centers[tid] = get_tag_center(det)
                detections_by_id[tid] = det

        R_w_c = None
        p_w_c = None
        reproj = float("nan")
        distance_compare: DistanceCompare = {}
        missing = [tid for tid in tag_ids if tid not in centers]
        if not missing:
            img_centers = np.asarray(
                [centers[tid] for tid in tag_ids], dtype=np.float64
            )
            try:
                R_w_c, p_w_c, rp = solve_camera_pose_from_square_centers(
                    img_centers, K, R_w_g, t_w_g
                )
                reproj = float(np.asarray(rp).reshape(-1)[0])
                distance_compare = compare_tag_distances(
                    p_w_c, detections_by_id, tag_ids, world_points
                )
                last_pose = (
                    R_w_c.copy(),
                    p_w_c.copy(),
                    reproj,
                    dict(distance_compare),
                )
            except Exception as exc:
                missing = []
                put_lines(frame, [(f"solvePnP failed: {exc}", (0, 0, 255))])

        draw_detections(frame, detections, tag_id_set)

        max_diff = 0.2
        if R_w_c is not None and p_w_c is not None:
            lines = [
                (f"READY visible=4/4 reproj={reproj:.3f}px", (0, 220, 0)),
                (
                    f"pos=({p_w_c[0]:+.3f}, {p_w_c[1]:+.3f}, {p_w_c[2]:+.3f})",
                    (0, 220, 0),
                ),
                (format_distance_summary(distance_compare), (0, 220, 0)),
                ("Space/S save   Q/Esc quit", (255, 255, 255)),
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
                (f"WAIT visible={len(centers)}/4 missing={missing}", (0, 220, 255)),
                ("show all four turntable tags", (0, 220, 255)),
                ("Space/S save last valid   Q/Esc quit", (255, 255, 255)),
            ]
        put_lines(frame, lines)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (32, ord("s"), ord("S")):
            pose = (
                (R_w_c, p_w_c, reproj, distance_compare)
                if R_w_c is not None and p_w_c is not None
                else last_pose
            )
            if pose is None:
                print("no valid pose yet; not saved")
                continue
            save_calibration(
                out_path,
                K,
                dist_coeffs,
                pose[0],
                pose[1],
                pose[2],
                tag_ids,
                world_points,
                (actual_w, actual_h),
                pose[3],
            )

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
