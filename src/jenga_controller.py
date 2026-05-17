"""Top-level Jenga robot controller.

Orchestrates Arduino motion, 4-camera AprilTag vision, and human interaction
through a state machine:
    INIT -> TOWER_INIT -> WAIT -> FULL_LAYER_1 -> WAIT -> LASTLY -> WAIT ...
    COLLAPSE (emergency halt)

Usage:
    uv run python src/jenga_controller.py --cams 0 1 2 3
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

try:
    from .arduino_serial import (
        ArduinoMotionClient,
        find_arduino_port,
        is_terminal_response,
    )
    from .calibration_core import (
        TURNTABLE_TAG_IDS,
        build_square_frame,
        get_tag_center,
        solve_camera_pose_from_square_centers,
    )
    from .reconstruction_core import (
        CameraObsInput,
        ReconConfig,
        ReconOutput,
        reconstruct_blocks_multi_cam,
    )
    from .view_camera_location import (
        BAR_HALF,
        FIXED_CAMERA_K,
        TAG_LOCAL,
        CamState,
    )
    from .view_camera_location_calibrated import (
        _calibration_camera_worker,
        all_cameras_live,
        build_states,
        compute_aligned_lines,
        draw_aligned_lines_in_viewer,
        draw_brick_models_in_viewer,
        freeze_calibration,
        gather_tag_world_positions,
        save_calib_with_distance,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from arduino_serial import (
        ArduinoMotionClient,
        find_arduino_port,
        is_terminal_response,
    )
    from calibration_core import (
        TURNTABLE_TAG_IDS,
        build_square_frame,
        get_tag_center,
        solve_camera_pose_from_square_centers,
    )
    from reconstruction_core import (
        CameraObsInput,
        ReconConfig,
        ReconOutput,
        reconstruct_blocks_multi_cam,
    )
    from view_camera_location import (
        BAR_HALF,
        FIXED_CAMERA_K,
        TAG_LOCAL,
        CamState,
    )
    from view_camera_location_calibrated import (
        _calibration_camera_worker,
        all_cameras_live,
        build_states,
        compute_aligned_lines,
        draw_aligned_lines_in_viewer,
        draw_brick_models_in_viewer,
        freeze_calibration,
        gather_tag_world_positions,
        save_calib_with_distance,
    )

DEFAULT_PNP_TAG_IDS = (300, 301, 302, 303)
S_CV2MJ = np.diag([1.0, -1.0, -1.0])

# Threshold for considering two bricks in the same layer (MuJoCo units)
LAYER_Z_THRESHOLD = 0.6
# Threshold for brick facing pushrod (alignment with world X)
FACE_PUSHROD_THRESHOLD = 0.85
# Empirical push duration (seconds)
DEFAULT_PUSH_DURATION = 1.5
# DC motor duty for push (0-255)
DEFAULT_PUSH_DC_DUTY = 180
# A brick is considered gone once its midpoint moves this far from its
# TOWER_INIT midpoint.
BRICK_GONE_DISTANCE = 2.0
# First layer considered by the full-layer AUTO_PUSH search.
FULL_LAYER_SEARCH_FIRST_LAYER = 1
# Target refresh rate for the optional MuJoCo live tower viewer.
LIVE_VIEW_UPDATE_SECONDS = 0.25

GAME_WIN = "win"
GAME_LOSE = "lose"


def _bypass_calib_camera_worker(
    state: CamState,
    stop_event: threading.Event,
) -> None:
    """Capture frames while keeping the stored calibration pose fixed."""
    with state.lock:
        if state.stored_p is not None and state.stored_R is not None:
            state.p_w_c = state.stored_p.copy()
            state.R_w_c = state.stored_R.copy()
            state.reproj = 0.0

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
            camera_params=(
                float(state.K[0, 0]),
                float(state.K[1, 1]),
                float(state.K[0, 2]),
                float(state.K[1, 2]),
            ),
            tag_size=float(state.tag_size),
        )
        tag_positions_cam: Dict[int, np.ndarray] = {}
        for det in dets:
            tid = int(getattr(det, "tag_id", -1))
            if hasattr(det, "pose_t"):
                tag_positions_cam[tid] = np.asarray(
                    det.pose_t, dtype=np.float64
                ).reshape(3)

        with state.lock:
            if state.stored_p is not None and state.stored_R is not None:
                state.p_w_c = state.stored_p.copy()
                state.R_w_c = state.stored_R.copy()
                state.reproj = 0.0
            state.n_visible = 4
            state.latest_frame_bgr = frame.copy()
            state.tag_positions_cam = tag_positions_cam


class Phase(Enum):
    INIT = auto()
    TOWER_INIT = auto()
    WAIT = auto()
    FULL_LAYER_1 = auto()
    LASTLY = auto()
    COLLAPSE = auto()


@dataclass
class GridResult:
    """Grid mapping result for one reconstruction frame."""

    # (x, y, z) -> brick_index
    grid: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    # y -> occupancy string, e.g. "101"
    occupancy: Dict[int, str] = field(default_factory=dict)
    # y -> list of (brick_index, t_w_b, R_w_b) sorted by x
    layers: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    # Total brick count
    total: int = 0


def bricks_to_grid(
    fitted: Dict[int, Tuple[np.ndarray, np.ndarray]],
    layer_count: int = 8,
    bricks_per_layer: int = 3,
) -> GridResult:
    """Map reconstructed brick positions to (x, y, z) grid.

    Args:
        fitted: brick_index -> (t_w_b, R_w_b)
        layer_count: expected number of layers
        bricks_per_layer: expected bricks per layer

    Returns:
        GridResult with grid, occupancy strings, and layer groupings.
    """
    result = GridResult()
    if not fitted:
        return result

    # Extract (brick_index, position, rotation)
    bricks = [(bi, t.copy(), R.copy()) for bi, (t, R) in fitted.items()]
    result.total = len(bricks)

    # Sort by world Z (vertical height in MuJoCo)
    bricks.sort(key=lambda item: item[1][2])

    # Greedy clustering into layers by height
    layers_list: List[List[Tuple[int, np.ndarray, np.ndarray]]] = []
    for bi, t, R in bricks:
        assigned = False
        for layer in layers_list:
            ref_z = layer[0][1][2]
            if abs(t[2] - ref_z) < LAYER_Z_THRESHOLD:
                layer.append((bi, t, R))
                assigned = True
                break
        if not assigned:
            layers_list.append([(bi, t, R)])

    # Assign y indices bottom-to-top
    layers_list.sort(key=lambda layer: layer[0][1][2])

    for y_idx, layer in enumerate(layers_list):
        if y_idx >= layer_count:
            break

        # Determine layer orientation from average long axis
        avg_long = np.zeros(3, dtype=np.float64)
        for _bi, _t, R in layer:
            # Local Z is the long axis (length 3.0)
            long_axis = R[:, 2]
            avg_long += long_axis
        avg_long /= max(len(layer), 1)

        # Normalize to XY plane for orientation angle
        avg_xy = avg_long[:2].copy()
        norm_xy = np.linalg.norm(avg_xy)
        if norm_xy > 1e-6:
            avg_xy /= norm_xy
        else:
            avg_xy = np.array([1.0, 0.0], dtype=np.float64)

        # z orientation in 90-degree steps
        angle = math.atan2(float(avg_xy[1]), float(avg_xy[0]))
        z_orient = int(round(math.degrees(angle) / 90)) % 4

        # Determine sorting axis: X-aligned vs Y-aligned
        is_x_aligned = abs(avg_long[0]) > abs(avg_long[1])
        if is_x_aligned:
            layer.sort(key=lambda item: item[1][0])
        else:
            layer.sort(key=lambda item: item[1][1])

        result.layers[y_idx] = layer

        # Build occupancy string and grid mapping
        occ = ["0"] * bricks_per_layer
        for slot_idx, (bi, t, R) in enumerate(layer):
            if slot_idx >= bricks_per_layer:
                continue
            occ[slot_idx] = "1"
            result.grid[(slot_idx, y_idx, z_orient)] = bi

        result.occupancy[y_idx] = "".join(occ)

    return result


def brick_faces_pushrod(R_w_b: np.ndarray, threshold: float = FACE_PUSHROD_THRESHOLD) -> bool:
    """Return True if brick's short face (local X) is aligned with world X."""
    local_x = R_w_b[:, 0]
    return abs(float(local_x[0])) > threshold


def layer_faces_pushrod(layer: List[Tuple[int, np.ndarray, np.ndarray]]) -> bool:
    """Return True if any brick in the layer faces the pushrod."""
    if not layer:
        return False
    for _bi, _t, R in layer:
        if brick_faces_pushrod(R):
            return True
    return False


def layer_axis_is_y_aligned(layer: List[Tuple[int, np.ndarray, np.ndarray]]) -> bool:
    """Return True when tag ends 0/1 face cam0: brick long axis is ±Y."""
    if not layer:
        return False
    avg_long = np.zeros(3, dtype=np.float64)
    for _bi, _t, R in layer:
        avg_long += R[:, 2]
    avg_long /= max(len(layer), 1)
    return abs(float(avg_long[1])) >= abs(float(avg_long[0]))


@dataclass
class JengaController:
    # Hardware
    arduino: ArduinoMotionClient
    cam_states: List[CamState]
    stop_event: threading.Event
    threads: List[threading.Thread]

    # Reconstruction
    recon_detector: Detector
    recon_cfg: ReconConfig
    scale: float

    # State machine
    phase: Phase = Phase.INIT
    next_phase_after_wait: Phase = Phase.FULL_LAYER_1

    # Tower state
    layer_count: int = 8
    bricks_per_layer: int = 3
    last_reconstruction: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    last_grid: GridResult = field(default_factory=GridResult)
    last_brick_count: int = 0
    initial_brick_midpoints: Dict[int, np.ndarray] = field(default_factory=dict)
    gone_brick_ids: set[int] = field(default_factory=set)
    stable_brick_count: int = 0
    stable_since: float = 0.0
    player_wait_started_at: float = 0.0
    player_wait_announced: bool = False
    current_layer_id: int = 0

    # Push parameters
    push_dc_duty: int = DEFAULT_PUSH_DC_DUTY
    push_duration: float = DEFAULT_PUSH_DURATION

    # Live visualization
    live: bool = False
    live_xml_path: Optional[Path] = None
    live_lookat: Optional[np.ndarray] = None

    # Flags
    calibration_locked: bool = False
    bypass_calib: bool = False
    game_outcome: Optional[str] = None

    # Internal
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _full_layer_search_start: int = FULL_LAYER_SEARCH_FIRST_LAYER
    _last_live_view_update: float = 0.0

    # ------------------------------------------------------------------
    # Serial helpers
    # ------------------------------------------------------------------
    def _send_and_wait(
        self, cmd: str, timeout: float = 30.0, ignore_errors: bool = False
    ) -> list[str]:
        if ignore_errors:
            self.arduino.send(cmd)
            lines: list[str] = []
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = self.arduino._readline()
                if not line:
                    continue
                if line.startswith("ERR "):
                    print(f"  Ignored Arduino error on '{cmd}': {line}")
                    continue
                lines.append(line)
                if is_terminal_response(line):
                    break
            return lines

        lines = self.arduino.command(cmd, timeout=timeout)
        for line in lines:
            if line.startswith("ERR "):
                raise RuntimeError(f"Arduino error on '{cmd}': {line}")
        return lines

    def _wait_for_done(self, axis: str, timeout: float = 30.0) -> None:
        """Poll STATUS? until BUSY=0."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            lines = self.arduino.command("STATUS?", timeout=2.0)
            for line in lines:
                if line.startswith("POS "):
                    # Parse BUSY=... at end
                    busy_part = line.split("BUSY=")[-1]
                    try:
                        busy = int(busy_part.strip())
                    except ValueError:
                        continue
                    if busy == 0:
                        return
            time.sleep(0.05)
        raise TimeoutError(f"Timeout waiting for axis {axis}")

    def _wait_for_done_signal(self, axis: str, timeout: float = 30.0) -> None:
        """Read serial until the firmware reports DONE for a specific axis."""
        expected = f"DONE {axis.upper()}"
        deadline = time.monotonic() + timeout
        last_line = ""
        while time.monotonic() < deadline:
            line = self.arduino._readline()
            if not line:
                continue
            last_line = line
            if line.startswith("ERR "):
                print(f"  Ignored Arduino error while waiting for {expected}: {line}")
                continue
            if line == expected or line.startswith(expected + " "):
                return
        raise TimeoutError(f"Timeout waiting for {expected}; last response: {last_line}")

    def _send_rotation_command_ignore_errors(self, cmd: str, drain_seconds: float = 0.5) -> None:
        """Send a rotation-related command and ignore all immediate ERR responses."""
        self.arduino.send(cmd)
        deadline = time.monotonic() + drain_seconds
        while time.monotonic() < deadline:
            line = self.arduino._readline()
            if not line:
                continue
            if line.startswith("ERR "):
                print(f"  Ignored Arduino error during rotation command '{cmd}': {line}")
                continue
            if line in {"OK", "PONG"} or line.startswith("DONE "):
                return

    def _rotate_z_with_x_retract(self, z_steps: int) -> None:
        """Retract X before any Z rotation."""
        self._send_rotation_command_ignore_errors("MOVE X -10000")
        self._wait_for_done("X")
        self.arduino.send(f"BM Z {z_steps}")
        self._wait_for_done_signal("Z", timeout=max(30.0, abs(float(z_steps)) * 3.0))

    def _finish_auto_push_x_move(self) -> None:
        self._send_and_wait("MOVE X 10000", timeout=2.0, ignore_errors=True)
        self._wait_for_done("X")

    def _run_auto_push(self) -> bool:
        self.arduino.send("AUTO_PUSH")
        deadline = time.monotonic() + 180.0
        last_line = ""
        while time.monotonic() < deadline:
            line = self.arduino._readline()
            if not line:
                continue
            last_line = line
            if line == "AUTO_PUSH SUCCESS":
                self._finish_auto_push_x_move()
                return True
            if line.startswith("ERR "):
                print(f"  Ignored Arduino error during AUTO_PUSH: {line}")
                continue
            if line == "WARN: AUTO_PUSH FAILED":
                self._finish_auto_push_x_move()
                raise RuntimeError(line)
        raise TimeoutError(
            f"AUTO_PUSH did not return a final success/failure signal; last response: {last_line}"
        )

    def _read_a1(self, timeout: float = 2.0) -> Optional[int]:
        self.arduino.send("A1?")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.arduino._readline()
            if not line:
                continue
            if line.startswith("ERR "):
                print(f"  Ignored Arduino error while reading A1: {line}")
                continue
            if line.startswith("A1="):
                try:
                    return int(line.split("=", 1)[1].strip().split()[0])
                except ValueError:
                    return None
        return None

    def _move_to_layer(self, new_layer_id: int) -> None:
        delta = int(new_layer_id) - int(self.current_layer_id)
        if delta == 0:
            return
        self._send_and_wait(f"BM Y {delta}")
        self._wait_for_done("Y")
        self.current_layer_id = int(new_layer_id)

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------
    def _reconstruct(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Run one-shot multi-camera reconstruction using frozen poses."""
        cam_obs: List[CameraObsInput] = []
        for st in self.cam_states:
            with st.lock:
                frame_bgr = (
                    None if st.latest_frame_bgr is None else st.latest_frame_bgr.copy()
                )
                p = None if st.stored_p is None else st.stored_p.copy()
                R = None if st.stored_R is None else st.stored_R.copy()

            if frame_bgr is None:
                continue
            if p is None or R is None:
                continue

            cam_obs.append(
                CameraObsInput(
                    cam_id=st.idx,
                    cam_name=st.name,
                    rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    K=st.K,
                    cam_pos_w=p,
                    R_w_c=R,
                )
            )

        if not cam_obs:
            return {}

        result = reconstruct_blocks_multi_cam(
            cams=cam_obs,
            detector=self.recon_detector,
            tag_local=[tl.copy() for tl in TAG_LOCAL],
            gt_tag_world=None,
            gt_block_pose=None,
            config=self.recon_cfg,
        )

        fitted: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for bi, br in result.block_results.items():
            if br.R_w_b is not None and br.t_w_b is not None:
                fitted[bi] = (br.t_w_b * self.scale, br.R_w_b)

        return fitted

    def _check_collapse(self, current_count: int) -> bool:
        """Return True if tower has collapsed."""
        if current_count < 4:
            return True
        return False

    def _record_initial_brick_midpoints(
        self, fitted: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self.initial_brick_midpoints = {
            bi: t.copy() for bi, (t, _R) in fitted.items()
        }
        self.gone_brick_ids.clear()
        self.last_brick_count = len(self.initial_brick_midpoints)

    def _gone_bricks_from_initial(
        self, fitted: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> set[int]:
        gone = set(self.gone_brick_ids)
        for bi, initial_t in self.initial_brick_midpoints.items():
            current = fitted.get(bi)
            if current is None:
                gone.add(bi)
                continue
            current_t, _R = current
            if np.linalg.norm(current_t - initial_t) >= BRICK_GONE_DISTANCE:
                gone.add(bi)
        return gone

    def _active_fitted(
        self,
        fitted: Dict[int, Tuple[np.ndarray, np.ndarray]],
        gone_ids: Optional[set[int]] = None,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        if not self.initial_brick_midpoints:
            return fitted
        if gone_ids is None:
            gone_ids = self._gone_bricks_from_initial(fitted)
        return {
            bi: pose
            for bi, pose in fitted.items()
            if bi in self.initial_brick_midpoints and bi not in gone_ids
        }

    def _set_gone_bricks(
        self,
        fitted: Dict[int, Tuple[np.ndarray, np.ndarray]],
        gone_ids: set[int],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        self.gone_brick_ids = set(gone_ids)
        active = self._active_fitted(fitted, self.gone_brick_ids)
        self.last_brick_count = len(self.initial_brick_midpoints) - len(
            self.gone_brick_ids
        )
        self.last_reconstruction = active
        self.last_grid = bricks_to_grid(active, self.layer_count, self.bricks_per_layer)
        return active

    def _enter_wait(self) -> None:
        self.player_wait_started_at = time.monotonic()
        self.player_wait_announced = False
        self.phase = Phase.WAIT

    def _end_game(self, outcome: str) -> None:
        self.game_outcome = outcome
        self.phase = Phase.COLLAPSE

    def _handle_successful_robot_push(self, phase_label: str) -> bool:
        """Return False if the robot push removed too many bricks and loses."""
        time.sleep(0.5)
        fitted = self._reconstruct()
        gone_ids = self._gone_bricks_from_initial(fitted)
        newly_gone = gone_ids - self.gone_brick_ids

        if len(newly_gone) >= 2:
            active = self._set_gone_bricks(fitted, gone_ids)
            print(
                f"{phase_label} Robot push moved {len(newly_gone)} bricks at least "
                f"{BRICK_GONE_DISTANCE:g} units from their initial midpoints. "
                f"Machine failed. Active bricks: {len(active)}."
            )
            self._end_game(GAME_LOSE)
            return False

        if len(newly_gone) == 1:
            self._set_gone_bricks(fitted, gone_ids)

        return True

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------
    def _do_init(self) -> None:
        print("[Phase INIT] Sending INIT to Arduino...")
        lines = self.arduino.initialize(timeout=120.0)
        for line in lines:
            print(f"  Arduino: {line}")
            if line == "INIT DONE":
                break
            if line.startswith("ERR "):
                raise RuntimeError(f"Arduino INIT failed: {line}")

        if self.calibration_locked:
            print(
                "[Phase INIT] Camera calibration already complete; skipping live PnP calibration."
            )
            self.phase = Phase.TOWER_INIT
            return

        print("[Phase INIT] Waiting for all 4 cameras to calibrate...")
        while not self.stop_event.is_set():
            if all_cameras_live(self.cam_states):
                break
            time.sleep(0.1)

        print("[Phase INIT] Freezing calibration poses...")
        saved = freeze_calibration(
            self.cam_states, DEFAULT_PNP_TAG_IDS, self._get_world_points()
        )
        print(f"  Saved {saved} camera pose(s)")
        self.calibration_locked = True

        self.phase = Phase.TOWER_INIT

    def _get_world_points(self) -> np.ndarray:
        # Default world points matching view_camera_location_calibrated.py
        return np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

    def _do_tower_init(self) -> None:
        print("[Phase TOWER_INIT] Reconstructing tower...")
        fitted = self._reconstruct()
        count = len(fitted)
        print(f"  Detected {count} bricks")

        if count == self.layer_count * self.bricks_per_layer:
            if self.stable_brick_count != count:
                self.stable_brick_count = count
                self.stable_since = time.monotonic()
            elif time.monotonic() - self.stable_since > 3.0:
                print("[Phase TOWER_INIT] 24 bricks stable for 3s. Tower ready.")
                self.last_reconstruction = fitted
                self.last_grid = bricks_to_grid(
                    fitted, self.layer_count, self.bricks_per_layer
                )
                self._record_initial_brick_midpoints(fitted)
                print(
                    f"[Phase TOWER_INIT] Recorded {len(self.initial_brick_midpoints)} "
                    "initial brick midpoint(s)."
                )
                self._enter_wait()
                return
        else:
            self.stable_brick_count = 0
            self.stable_since = 0.0

        time.sleep(1)

    def _do_wait(self) -> None:
        a1_value = self._read_a1()
        if a1_value != 0:
            if not self.player_wait_announced:
                print("[Phase WAIT] Waiting until Arduino A1 reads 0...")
                self.player_wait_announced = True
            time.sleep(0.2)
            return
        if self.player_wait_announced:
            print("[Phase WAIT] Arduino A1 reads 0. Checking whether the player removed a brick...")
            self.player_wait_announced = False

        fitted = self._reconstruct()
        gone_ids = self._gone_bricks_from_initial(fitted)
        newly_gone = gone_ids - self.gone_brick_ids

        if len(newly_gone) >= 2:
            active = self._set_gone_bricks(fitted, gone_ids)
            print(
                f"[Phase WAIT] Player moved {len(newly_gone)} bricks at least "
                f"{BRICK_GONE_DISTANCE:g} units from their initial midpoints. "
                f"System wins. Active bricks: {len(active)}."
            )
            self._end_game(GAME_WIN)
            return

        if len(newly_gone) == 1:
            active = self._set_gone_bricks(fitted, gone_ids)
            print(
                f"[Phase WAIT] Brick removed by midpoint displacement. "
                f"Active bricks: {len(active)}."
            )
            self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
            self.phase = self.next_phase_after_wait
            return

        active = self._active_fitted(fitted, gone_ids)
        if self._check_collapse(len(active)):
            print(f"[Phase WAIT] COLLAPSE detected. Active bricks: {len(active)}")
            self._end_game(GAME_LOSE)
            return

        print("[Phase WAIT] No brick removed. Waiting for Arduino A1 to read 0 again.")
        self.player_wait_announced = False
        time.sleep(0.5)

    def _do_full_layer_1(self) -> None:
        print("[Phase FULL_LAYER_1] Mapping grid...")
        fitted = self._reconstruct()
        gone_ids = self._gone_bricks_from_initial(fitted)
        active = self._active_fitted(fitted, gone_ids)
        grid_result = bricks_to_grid(active, self.layer_count, self.bricks_per_layer)
        self.last_grid = grid_result

        if self._check_collapse(grid_result.total):
            print(f"[Phase FULL_LAYER_1] COLLAPSE. Active bricks: {grid_result.total}")
            self._end_game(GAME_LOSE)
            return

        # Find a full layer (occupancy "111") in sequential order. A new robot
        # turn starts from layer 1; retries continue upward from the last target.
        target_layer: Optional[int] = None
        search_start = max(
            FULL_LAYER_SEARCH_FIRST_LAYER,
            min(self._full_layer_search_start, self.layer_count),
        )
        for y in range(search_start, self.layer_count):
            if grid_result.occupancy.get(y) == "111":
                target_layer = y
                break

        if target_layer is None:
            print("[Phase FULL_LAYER_1] No full layer found. Switching to LASTLY.")
            self.next_phase_after_wait = Phase.LASTLY
            self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
            self._enter_wait()
            return

        layer = grid_result.layers.get(target_layer, [])
        if layer_axis_is_y_aligned(layer):
            print(f"[Phase FULL_LAYER_1] Layer {target_layer} is accessible. Running AUTO_PUSH.")
        else:
            print(
                f"[Phase FULL_LAYER_1] Layer {target_layer} is full but not accessible "
                "(axis is not +Y/-Y). Rotating tower before AUTO_PUSH."
            )
            try:
                self._rotate_z_with_x_retract(1)
            except (RuntimeError, TimeoutError) as exc:
                print(f"[Phase FULL_LAYER_1] Rotation failed: {exc}")
                self._full_layer_search_start = target_layer + 1
                time.sleep(0.5)
                return
            print(
                f"[Phase FULL_LAYER_1] DONE Z received. Running AUTO_PUSH on layer {target_layer}."
            )

        try:
            self._move_to_layer(target_layer)
            self._run_auto_push()
        except (RuntimeError, TimeoutError) as exc:
            print(
                f"[Phase FULL_LAYER_1] AUTO_PUSH failed on layer {target_layer}: {exc}. "
                "Retrying another layer."
            )
            self._full_layer_search_start = target_layer + 1
            time.sleep(0.5)
            return

        print("[Phase FULL_LAYER_1] AUTO_PUSH succeeded. Checking tower state.")
        if not self._handle_successful_robot_push("[Phase FULL_LAYER_1]"):
            return
        print("[Phase FULL_LAYER_1] Jumping to next phase.")
        self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
        self._enter_wait()

    def _try_push_layer(
        self, layer: List[Tuple[int, np.ndarray, np.ndarray]], target_y: int
    ) -> bool:
        """Attempt to push one brick from the given layer. Returns True on success."""
        if not layer:
            return False

        # Check orientation
        if not layer_faces_pushrod(layer):
            print(f"  Layer {target_y} not facing pushrod. Rotating Z...")
            self._rotate_z_with_x_retract(1)
            # Reconstruct after rotation
            fitted = self._reconstruct()
            gone_ids = self._gone_bricks_from_initial(fitted)
            active = self._active_fitted(fitted, gone_ids)
            grid_result = bricks_to_grid(active, self.layer_count, self.bricks_per_layer)
            layer = grid_result.layers.get(target_y, [])
            if not layer_faces_pushrod(layer):
                print(f"  Layer {target_y} still not facing after rotation. Skip.")
                return False

        # Select brick: prefer middle (x=1) to avoid side instability,
        # but any present brick is valid for a full layer.
        for bi, t, R in layer:
            # Find slot index for this brick
            slot = None
            for (sx, sy, sz), idx in self.last_grid.grid.items():
                if idx == bi and sy == target_y:
                    slot = sx
                    break
            if slot is None:
                # Fallback: use position order within layer
                slot = layer.index((bi, t, R))

            print(f"  Pushing brick {bi} at layer={target_y} slot={slot}")
            try:
                self._execute_push(slot, target_y)
            except (RuntimeError, TimeoutError) as exc:
                print(f"  Push error: {exc}")
                continue

            # Verify by reconstruction
            time.sleep(0.5)
            post_fitted = self._reconstruct()
            post_gone_ids = self._gone_bricks_from_initial(post_fitted)
            if bi in post_gone_ids:
                print(
                    f"  Brick {bi} confirmed gone by midpoint displacement "
                    f">= {BRICK_GONE_DISTANCE:g}."
                )
                return True
            else:
                print(f"  Brick {bi} has not moved far enough. Trying next.")

        return False

    def _execute_push(self, target_x: int, target_y: int) -> None:
        """Orchestrate pushrod movement and DC motor push."""
        # 1. Move to layer height
        self._move_to_layer(target_y)

        # 2. Move to brick x-position
        self._send_and_wait(f"BM X {target_x}")
        self._wait_for_done("X")

        # 3. Push with DC motor
        self._send_and_wait(f"DC F {self.push_dc_duty}")
        time.sleep(self.push_duration)
        self._send_and_wait("DC 0")

        # 4. Retract X
        self._send_and_wait("MOVE X -200 F 800 A 600")
        self._wait_for_done("X")

        # 5. Home Y
        self._send_and_wait("MOVE Y 0 F 800 A 600")
        self._wait_for_done("Y")
        self.current_layer_id = 0

    def _do_lastly(self) -> None:
        print("[Phase LASTLY] Mapping grid...")
        fitted = self._reconstruct()
        gone_ids = self._gone_bricks_from_initial(fitted)
        active = self._active_fitted(fitted, gone_ids)
        grid_result = bricks_to_grid(active, self.layer_count, self.bricks_per_layer)
        self.last_grid = grid_result

        if self._check_collapse(grid_result.total):
            print(f"[Phase LASTLY] COLLAPSE. Active bricks: {grid_result.total}")
            self._end_game(GAME_LOSE)
            return

        # Search for "110" or "011" patterns from the bottom layer upward.
        candidates: List[Tuple[int, int]] = []  # (y, x_to_push)
        for y in range(0, self.layer_count):
            occ = grid_result.occupancy.get(y, "")
            if occ == "110":
                candidates.append((y, 1))  # push middle brick (x=1) which is at side
            elif occ == "011":
                candidates.append((y, 1))  # push middle brick (x=1) which is at side

        if not candidates:
            # Random selection from any existing brick.
            all_bricks: List[Tuple[int, int]] = []
            for y in range(0, self.layer_count):
                layer = grid_result.layers.get(y, [])
                for bi, _t, _R in layer:
                    for (sx, sy, sz), idx in grid_result.grid.items():
                        if idx == bi and sy == y:
                            all_bricks.append((y, sx))
                            break
            if all_bricks:
                candidates = [random.choice(all_bricks)]

        if not candidates:
            print("[Phase LASTLY] No bricks found. Halting.")
            self._end_game(GAME_LOSE)
            return

        target_y, target_x = candidates[0]
        layer = grid_result.layers.get(target_y, [])
        pushed = self._try_push_layer(layer, target_y)

        if pushed:
            print("[Phase LASTLY] Push succeeded. Checking tower state.")
            if not self._handle_successful_robot_push("[Phase LASTLY]"):
                return
            print("[Phase LASTLY] Returning to WAIT.")
            self._enter_wait()
        else:
            print("[Phase LASTLY] Push failed. Retrying.")
            time.sleep(0.5)

    def _do_collapse(self) -> None:
        print("[Phase COLLAPSE] HALTING ALL MOTION")
        if self.game_outcome in (GAME_WIN, GAME_LOSE):
            z_steps = 20 if self.game_outcome == GAME_WIN else -20
            label = "WIN" if self.game_outcome == GAME_WIN else "LOSE"
            try:
                print(f"  Game result: {label}. Running BM Z {z_steps} before halt.")
                self._rotate_z_with_x_retract(z_steps)
            except Exception as exc:
                print(f"  Result motion error (ignored): {exc}")

        try:
            self.arduino.command("STOP!", timeout=2.0)
            self.arduino.command("DC 0", timeout=2.0)
        except Exception as exc:
            print(f"  Arduino stop error (ignored): {exc}")

        # Log final poses
        try:
            try:
                from .view_camera_location import write_poses_txt
            except ImportError:
                from view_camera_location import write_poses_txt

            out = Path("collapse_poses.txt")
            write_poses_txt(out, self.last_reconstruction)
            print(f"  Final poses logged to {out}")
        except Exception as exc:
            print(f"  Pose log error: {exc}")

        # Cleanup cameras
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=1.0)
        for st in self.cam_states:
            st.cap.release()

        print("[Phase COLLAPSE] Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run_phase_once(self) -> None:
        if self.phase == Phase.INIT:
            self._do_init()
        elif self.phase == Phase.TOWER_INIT:
            self._do_tower_init()
        elif self.phase == Phase.WAIT:
            self._do_wait()
        elif self.phase == Phase.FULL_LAYER_1:
            self._do_full_layer_1()
        elif self.phase == Phase.LASTLY:
            self._do_lastly()

    def _update_live_viewer(self, viewer, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_live_view_update < LIVE_VIEW_UPDATE_SECONDS:
            return
        self._last_live_view_update = now

        tag_positions_per_cam = gather_tag_world_positions(
            self.cam_states, prefer_stored=self.calibration_locked or self.bypass_calib
        )
        aligned_lines = compute_aligned_lines(
            tag_positions_per_cam, start_id=self.recon_cfg.start_id
        )
        if self.gone_brick_ids:
            aligned_lines = [
                line
                for line in aligned_lines
                if int(line.get("base", -1)) not in self.gone_brick_ids
            ]

        viewer.user_scn.ngeom = 0
        draw_aligned_lines_in_viewer(viewer, aligned_lines)
        draw_brick_models_in_viewer(viewer, aligned_lines)
        viewer.sync()

    def _run_with_live_viewer(self) -> None:
        if self.live_xml_path is None:
            raise RuntimeError("--live requires a MuJoCo XML path")

        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_path(str(self.live_xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        print(f"[Live] Opening MuJoCo viewer: {self.live_xml_path}")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            if self.live_lookat is not None:
                viewer.cam.lookat[:] = self.live_lookat
            viewer.cam.distance = 12.0
            viewer.cam.elevation = -25.0

            while viewer.is_running() and self.phase != Phase.COLLAPSE:
                self._run_phase_once()
                self._update_live_viewer(viewer)
                time.sleep(0.01)

        if self.phase != Phase.COLLAPSE:
            print("[Live] MuJoCo viewer closed; continuing controller without live view.")
            self.live = False
            while self.phase != Phase.COLLAPSE:
                self._run_phase_once()
                time.sleep(0.01)

    def run(self) -> None:
        print("=" * 60)
        print("Jenga Controller started")
        print(f"  Phase: {self.phase.name}")
        if self.live:
            print("  Live MuJoCo tower viewer: enabled")
        print("=" * 60)

        try:
            if self.live:
                self._run_with_live_viewer()
            else:
                while self.phase != Phase.COLLAPSE:
                    self._run_phase_once()
                    time.sleep(0.01)
            self._do_collapse()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            self.phase = Phase.COLLAPSE
            self._do_collapse()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Jenga robot top-level controller.")
    # Serial
    ap.add_argument(
        "--port",
        help="Arduino serial port. Auto-detected when omitted.",
    )
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=2.0)

    # Cameras
    ap.add_argument(
        "--cams",
        nargs="+",
        default=["0"],
        help="Camera device indices or paths (up to 4)",
    )
    ap.add_argument("--cam-names", nargs="*", default=[])
    ap.add_argument("--calib-in", nargs="*", default=[])
    ap.add_argument("--calib-out", nargs="*", default=[])
    ap.add_argument(
        "--bypass-calib",
        action="store_true",
        help="Skip live PnP calibration and use stored camera poses from --calib-in directly.",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Show a live MuJoCo viewer with the real-time reconstructed tower model.",
    )
    ap.add_argument("--xml", default="assets/scene_turntable_only_lowlookat.xml")
    ap.add_argument("--fovy", type=float, default=100.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument(
        "--world-points",
        type=float,
        nargs=12,
        default=[
            -1, -1, 0.0,
            1, -1, 0.0,
            1, 1, 0.0,
            -1, 1, 0.0,
        ],
    )
    ap.add_argument("--tag-size", type=float, default=0.64)
    ap.add_argument("--real-tag-size", type=float, default=0.64)
    ap.add_argument("--mujoco-tag-size", type=float, default=0.64)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--enforce-flat", action="store_true", default=True)
    ap.add_argument("--no-enforce-flat", dest="enforce_flat", action="store_false")
    ap.add_argument("--coord", default="cv2mj_yz", choices=["identity", "cv2mj_yz"])
    ap.add_argument("--pnp-tag-ids", type=int, nargs=4, default=list(DEFAULT_PNP_TAG_IDS))

    # Tower / game
    ap.add_argument("--layer-count", type=int, default=8)
    ap.add_argument("--bricks-per-layer", type=int, default=3)

    # Push params
    ap.add_argument("--push-dc-duty", type=int, default=DEFAULT_PUSH_DC_DUTY)
    ap.add_argument("--push-duration", type=float, default=DEFAULT_PUSH_DURATION)

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    # --------------------------------------------------------------
    # Arduino
    # --------------------------------------------------------------
    port = args.port or find_arduino_port()
    if not port:
        print("ERROR: Could not auto-detect Arduino port. Pass --port.", file=sys.stderr)
        return 1

    arduino = ArduinoMotionClient(
        port=port,
        baud=args.baud,
        timeout=args.timeout,
        reset_delay=2.0,
    )

    live_xml_path: Optional[Path] = None
    if args.live:
        live_xml_path = Path(args.xml)
        if not live_xml_path.is_absolute():
            live_xml_path = base_dir / live_xml_path
        if not live_xml_path.exists():
            print(f"ERROR: XML not found for --live: {live_xml_path}", file=sys.stderr)
            arduino.close()
            return 1

    # --------------------------------------------------------------
    # Cameras
    # --------------------------------------------------------------
    pts = np.asarray(args.world_points, dtype=np.float64).reshape(4, 3)
    R_w_g, t_w_g = build_square_frame(pts)
    pnp_tag_ids = tuple(int(tid) for tid in args.pnp_tag_ids)

    scale = args.mujoco_tag_size / args.real_tag_size
    S = np.diag([1.0, -1.0, -1.0]) if args.coord == "cv2mj_yz" else np.eye(3)

    recon_cfg = ReconConfig(
        family="tag36h11",
        tag_size=args.real_tag_size,
        start_id=args.start_id,
        max_block_tag_id=239,  # 24 bricks * 10 tags = 240 tags -> max id 239
        S_cv2mj=S,
        enforce_flat=args.enforce_flat,
        save_overlay=False,
    )
    recon_detector = Detector(
        families="tag36h11", nthreads=2, quad_decimate=1.5, refine_edges=1
    )

    # Use build_states from view_camera_location_calibrated.py
    # but we need to patch in our own args Namespace with the right fields
    try:
        states = build_states(args, R_w_g, t_w_g)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.bypass_calib:
        missing = [
            st.name for st in states if st.stored_p is None or st.stored_R is None
        ]
        if missing:
            print(
                "ERROR: --bypass-calib requires a stored pose for every camera. "
                f"Missing for: {', '.join(missing)}. Pass valid --calib-in .npz files.",
                file=sys.stderr,
            )
            for st in states:
                st.cap.release()
            arduino.close()
            return 1
        for st in states:
            with st.lock:
                st.p_w_c = st.stored_p.copy()
                st.R_w_c = st.stored_R.copy()
                st.reproj = 0.0
            setattr(st, "bypass_calib", True)
        print(
            "--bypass-calib: skipping live calibration; using stored camera poses from --calib-in.\n"
        )

    stop_event = threading.Event()
    threads: List[threading.Thread] = []
    for st in states:
        if args.bypass_calib:
            target = _bypass_calib_camera_worker
            thread_args = (st, stop_event)
        else:
            target = _calibration_camera_worker
            thread_args = (st, stop_event, pnp_tag_ids, pts)
        t = threading.Thread(
            target=target,
            args=thread_args,
            daemon=True,
        )
        t.start()
        threads.append(t)

    # --------------------------------------------------------------
    # Controller
    # --------------------------------------------------------------
    controller = JengaController(
        arduino=arduino,
        cam_states=states,
        stop_event=stop_event,
        threads=threads,
        recon_detector=recon_detector,
        recon_cfg=recon_cfg,
        scale=scale,
        layer_count=args.layer_count,
        bricks_per_layer=args.bricks_per_layer,
        push_dc_duty=args.push_dc_duty,
        push_duration=args.push_duration,
        live=args.live,
        live_xml_path=live_xml_path,
        live_lookat=t_w_g.copy(),
        calibration_locked=args.bypass_calib,
        bypass_calib=args.bypass_calib,
    )

    try:
        controller.run()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        for st in states:
            st.cap.release()
        arduino.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
