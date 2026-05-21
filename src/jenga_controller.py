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
import subprocess
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
        cv_point_to_world,
        get_pupil_pose,
        is_block_tag_id,
        reconstruct_blocks_multi_cam,
    )
    from .apriltag_utils import detect_apriltags_silent
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
        cv_point_to_world,
        get_pupil_pose,
        is_block_tag_id,
        reconstruct_blocks_multi_cam,
    )
    from apriltag_utils import detect_apriltags_silent
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
# First layer considered by the full-layer AUTO_PUSH search.
FULL_LAYER_SEARCH_FIRST_LAYER = 1
# Slot index of the middle brick within a 3-brick layer.
MIDDLE_BRICK_SLOT = 1
# Empty slot marker in the cached layer state tuple.
EMPTY_BRICK_ID = -1
# Target refresh rate for the optional MuJoCo live tower viewer.
LIVE_VIEW_UPDATE_SECONDS = 0.25

# Cached layer state: (axis_flag, slot0_brick_id, slot1_brick_id, slot2_brick_id).
# axis_flag == 0 means the layer is currently accessible; 1 means rotate first.
LayerState = Tuple[int, int, int, int]

GAME_WIN = "win"
GAME_LOSE = "lose"


class _SilentDetectorProxy:
    """Detector wrapper for imported workers that call .detect() directly."""

    def __init__(self, detector):
        self._detector = detector

    def detect(self, *args, **kwargs):
        return detect_apriltags_silent(self._detector, *args, **kwargs)


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
        dets = detect_apriltags_silent(
            state.detector,
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

        # Sort across the layer width, perpendicular to the brick long axis.
        is_x_aligned = abs(avg_long[0]) > abs(avg_long[1])
        if is_x_aligned:
            layer.sort(key=lambda item: item[1][1])
        else:
            layer.sort(key=lambda item: item[1][0])

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
    initial_reconstruction: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    initial_brick_midpoints: Dict[int, np.ndarray] = field(default_factory=dict)
    initial_tag_ids_by_brick: Dict[int, set[int]] = field(default_factory=dict)
    initial_layer_middle_brick_ids: Dict[int, int] = field(default_factory=dict)
    initial_layer_slot_brick_ids: Dict[int, Dict[int, int]] = field(
        default_factory=dict
    )
    layer_push_states: List[LayerState] = field(default_factory=list)
    gone_brick_ids: set[int] = field(default_factory=set)
    stable_brick_count: int = 0
    stable_since: float = 0.0
    player_wait_started_at: float = 0.0
    player_wait_announced: bool = False
    current_layer_id: int = 0
    tower_init_y_raised: bool = False
    tower_init_x_retracted: bool = False
    na_layer: List[int] = field(default_factory=list)

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
    xy_log: bool = False
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
        if abs(int(z_steps)) % 2 == 1:
            self._toggle_layer_axes_after_z_rotation()

    def _finish_auto_push_x_move(self) -> None:
        self._send_and_wait("MOVE X 10000", timeout=2.0, ignore_errors=True)
        self._wait_for_done("X")

    def _finish_auto_push_motion(self) -> None:
        self._finish_auto_push_x_move()

    def _play_done_sound(self, label: str = "sound") -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Glass.aiff"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.0,
                    check=False,
                )
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(1.0)
        except Exception as exc:
            print(f"  {label} error (ignored): {exc}")

    @staticmethod
    def _is_auto_push_success(line: str) -> bool:
        parts = line.strip().split()
        if len(parts) == 4 and parts[:3] == ["AUTO", "PUSH", "SUCCESS"]:
            return True
        if len(parts) == 3 and parts[:2] == ["AUTO_PUSH", "SUCCESS"]:
            return True
        if line == "AUTO_PUSH SUCCESS":
            return True
        return False

    def _run_auto_push(self, target_layer: int, push_mask: int) -> None:
        self._move_to_layer(target_layer)
        print(self.na_layer)
        if (push_mask == 6):
            self.arduino.send("AUTO_PUSH 4")
        elif (push_mask == 3):
            self.arduino.send("AUTO_PUSH 1")
        else:
            self.arduino.send(f"AUTO_PUSH {push_mask}")
        deadline = time.monotonic() + 180.0
        last_line = ""
        while time.monotonic() < deadline:
            line = self.arduino._readline()
            if not line:
                continue
            last_line = line
            if self._is_auto_push_success(line):
                self._finish_auto_push_motion()
                print(
                    "[Phase FULL_LAYER_1] AUTO_PUSH reported success; "
                    "using tag visibility to identify removed brick."
                )
                return
            if line.startswith("AUTO_PUSH A0_GROUNDED"):
                print(
                    "[Phase FULL_LAYER_1] AUTO_PUSH reported A0 grounded. "
                    "Updating layer state from one-tag visibility."
                )
                self._update_gone_from_visibility("[Phase FULL_LAYER_1 AUTO_PUSH]")
                continue
            if line.startswith("ERR "):
                print(f"  Ignored Arduino error during AUTO_PUSH: {line}")
                continue
            if line == "WARN: AUTO_PUSH FAILED":
                self._finish_auto_push_motion()
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
        self,
        fitted: Dict[int, Tuple[np.ndarray, np.ndarray]],
        grid_result: GridResult,
    ) -> None:
        self.initial_reconstruction = {
            bi: (t.copy(), R.copy()) for bi, (t, R) in fitted.items()
        }
        self.initial_brick_midpoints = {
            bi: t.copy() for bi, (t, _R) in fitted.items()
        }
        self.initial_tag_ids_by_brick = {}
        self.initial_layer_middle_brick_ids = {}
        self.initial_layer_slot_brick_ids = {}
        self.layer_push_states = []
        for y in range(self.layer_count):
            slot_ids: Dict[int, int] = {}
            for slot_id in range(self.bricks_per_layer):
                brick_id = self._grid_slot_brick_id(grid_result, y, slot_id)
                if brick_id is None:
                    continue
                slot_ids[slot_id] = brick_id

            if slot_ids:
                self.initial_layer_slot_brick_ids[y] = slot_ids
            mid_id = slot_ids.get(MIDDLE_BRICK_SLOT)
            if mid_id is not None:
                self.initial_layer_middle_brick_ids[y] = mid_id

            axis_flag = 0 if layer_axis_is_y_aligned(grid_result.layers.get(y, [])) else 1
            slots = (
                slot_ids.get(0, EMPTY_BRICK_ID),
                slot_ids.get(1, EMPTY_BRICK_ID),
                slot_ids.get(2, EMPTY_BRICK_ID),
            )
            if axis_flag != 0:
                slots = self._reverse_layer_slots(slots)
            self.layer_push_states.append(
                (
                    axis_flag,
                    slots[0],
                    slots[1],
                    slots[2],
                )
            )
        self.gone_brick_ids.clear()
        self.last_brick_count = len(self.initial_brick_midpoints)

    @staticmethod
    def _grid_slot_brick_id(
        grid_result: GridResult, layer_id: int, slot_id: int
    ) -> Optional[int]:
        for (sx, sy, _sz), brick_id in grid_result.grid.items():
            if sx == slot_id and sy == layer_id:
                return brick_id
        return None

    def _layer_state(self, layer_id: int) -> LayerState:
        if 0 <= layer_id < len(self.layer_push_states):
            return self.layer_push_states[layer_id]
        return 1, EMPTY_BRICK_ID, EMPTY_BRICK_ID, EMPTY_BRICK_ID

    @staticmethod
    def _layer_axis_flag(state: LayerState) -> int:
        return int(state[0])

    @staticmethod
    def _layer_slots(state: LayerState) -> Tuple[int, int, int]:
        return state[1], state[2], state[3]

    @staticmethod
    def _reverse_layer_slots(slots: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return slots[2], slots[1], slots[0]

    @staticmethod
    def _layer_push_mask(state: LayerState) -> int:
        mask = 0
        for slot_id, brick_id in enumerate(state[1:]):
            if brick_id != EMPTY_BRICK_ID:
                mask |= 1 << slot_id
        return mask

    @staticmethod
    def _layer_active_count(state: LayerState) -> int:
        return sum(1 for brick_id in state[1:] if brick_id != EMPTY_BRICK_ID)

    def _layer_auto_push_candidate(
        self, layer_id: int, ignore_na_layer: bool = False
    ) -> Tuple[bool, str]:
        state = self._layer_state(layer_id)
        slots = self._layer_slots(state)

        if not ignore_na_layer and layer_id in self.na_layer:
            return False, "in na_layer"
        if self._layer_active_count(state) == 0:
            return False, "no bricks left in cached layer state"
        if slots[MIDDLE_BRICK_SLOT] == EMPTY_BRICK_ID:
            return False, f"middle brick is gone in cached layer state {state}"
        if slots[0] == EMPTY_BRICK_ID and slots[2] == EMPTY_BRICK_ID:
            return False, f"only middle brick remains in cached layer state {state}"

        return True, ""

    def _has_available_auto_push_layer(self) -> bool:
        for layer_id in range(FULL_LAYER_SEARCH_FIRST_LAYER, self.layer_count):
            can_push, _reason = self._layer_auto_push_candidate(
                layer_id, ignore_na_layer=True
            )
            if can_push:
                return True
        return False

    def _mark_na_layer(self, layer_id: int, reason: str) -> None:
        if layer_id not in self.na_layer:
            self.na_layer.append(layer_id)
        print(
            f"[Phase FULL_LAYER_1] Marked layer {layer_id} in na_layer: "
            f"{reason}. na_layer={self.na_layer}"
        )

    def _advance_after_auto_push_failure(self, layer_id: int) -> None:
        self._mark_na_layer(layer_id, "AUTO_PUSH returned FAILED")
        if layer_id >= self.layer_count - 1:
            if not self._has_available_auto_push_layer():
                print(
                    "[Phase FULL_LAYER_1] No available AUTO_PUSH layer remains "
                    "in tower state. Conceding."
                )
                self._end_game(GAME_LOSE)
                return
            print(
                "[Phase FULL_LAYER_1] AUTO_PUSH failed at top layer. "
                f"Clearing na_layer {self.na_layer} and retrying from bottom."
            )
            self.na_layer.clear()
            self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
        else:
            self._full_layer_search_start = layer_id + 1

    def _update_layer_push_states_from_gone(self) -> None:
        states: List[LayerState] = []
        for layer_id in range(self.layer_count):
            state = self._layer_state(layer_id)
            axis_flag = self._layer_axis_flag(state)
            slots = [
                brick_id if brick_id not in self.gone_brick_ids else EMPTY_BRICK_ID
                for brick_id in self._layer_slots(state)
            ]
            states.append((axis_flag, slots[0], slots[1], slots[2]))
        self.layer_push_states = states

    def _toggle_layer_axes_after_z_rotation(self) -> None:
        if not self.layer_push_states:
            return
        states: List[LayerState] = []
        for state in self.layer_push_states:
            axis_flag = self._layer_axis_flag(state)
            slots = self._layer_slots(state)
            new_axis_flag = axis_flag ^ 1
            reverse_slots = axis_flag == 0 and new_axis_flag == 1
            if reverse_slots:
                slots = self._reverse_layer_slots(slots)
            states.append((new_axis_flag, slots[0], slots[1], slots[2]))
        self.layer_push_states = states

    def _brick_index_from_tag_id(self, tag_id: int) -> Optional[int]:
        if not is_block_tag_id(tag_id, self.recon_cfg):
            return None
        brick_id = (int(tag_id) - int(self.recon_cfg.start_id)) // 10
        if brick_id not in self.initial_brick_midpoints:
            return None
        return brick_id

    def _visible_initial_tags_by_brick(
        self, require_recorded_initial_tag: bool = True
    ) -> Dict[int, set[int]]:
        """Return visible initial tag ids grouped by brick, filtered by world z > 0."""
        visible: Dict[int, set[int]] = {}
        for st in self.cam_states:
            with st.lock:
                frame_bgr = (
                    None if st.latest_frame_bgr is None else st.latest_frame_bgr.copy()
                )
                p = None if st.stored_p is None else st.stored_p.copy()
                R = None if st.stored_R is None else st.stored_R.copy()
                K = st.K.copy()

            if frame_bgr is None or p is None or R is None:
                continue

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            dets = detect_apriltags_silent(
                self.recon_detector,
                gray,
                estimate_tag_pose=True,
                camera_params=(
                    float(K[0, 0]),
                    float(K[1, 1]),
                    float(K[0, 2]),
                    float(K[1, 2]),
                ),
                tag_size=float(self.recon_cfg.tag_size),
            )
            for det in dets:
                tid = int(getattr(det, "tag_id", -1))
                brick_id = self._brick_index_from_tag_id(tid)
                if brick_id is None:
                    continue
                if require_recorded_initial_tag:
                    recorded = self.initial_tag_ids_by_brick.get(brick_id, set())
                    if tid not in recorded:
                        continue

                pose = get_pupil_pose(det)
                if pose is None:
                    continue
                _R_c_t, t_c_t = pose
                p_est_w = cv_point_to_world(
                    t_c_t,
                    self.recon_cfg.S_cv2mj,
                    R,
                    p,
                )
                if float(p_est_w[2] * self.scale) > 0.0:
                    visible.setdefault(brick_id, set()).add(tid)

        return visible

    def _record_initial_visible_tags(self) -> None:
        self.initial_tag_ids_by_brick = self._visible_initial_tags_by_brick(
            require_recorded_initial_tag=False
        )

    def _visible_bricks_from_current_tags(self) -> set[int]:
        """Return bricks with any initially recorded tag visible at world z > 0."""
        visible_tags = self._visible_initial_tags_by_brick(
            require_recorded_initial_tag=True
        )
        return {brick_id for brick_id, tag_ids in visible_tags.items() if tag_ids}

    def _gone_bricks_from_current_visibility(self) -> set[int]:
        visible = self._visible_bricks_from_current_tags()
        return set(self.initial_brick_midpoints) - visible

    def _active_fitted(
        self,
        fitted: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        gone_ids: Optional[set[int]] = None,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        if not self.initial_brick_midpoints:
            return {} if fitted is None else fitted
        if fitted is None:
            fitted = self.initial_reconstruction
        if gone_ids is None:
            gone_ids = self._gone_bricks_from_current_visibility()
        return {
            bi: pose
            for bi, pose in fitted.items()
            if bi in self.initial_brick_midpoints and bi not in gone_ids
        }

    def _set_gone_bricks(
        self,
        gone_ids: set[int],
        fitted: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        self.gone_brick_ids = set(gone_ids)
        active = self._active_fitted(fitted, self.gone_brick_ids)
        self.last_brick_count = len(self.initial_brick_midpoints) - len(
            self.gone_brick_ids
        )
        self.last_reconstruction = active
        self.last_grid = bricks_to_grid(active, self.layer_count, self.bricks_per_layer)
        self._update_layer_push_states_from_gone()
        return active

    def _update_gone_from_visibility(
        self, phase_label: Optional[str] = None
    ) -> Tuple[set[int], set[int], set[int], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        gone_ids = self._gone_bricks_from_current_visibility()
        newly_gone = gone_ids - self.gone_brick_ids
        restored = self.gone_brick_ids - gone_ids
        active = self._set_gone_bricks(gone_ids)
        if self.xy_log:
            self._print_xy_log(phase_label)

        if phase_label is not None:
            if newly_gone:
                print(
                    f"{phase_label} Visibility update marked gone brick id(s): "
                    f"{sorted(newly_gone)}."
                )
            if restored:
                print(
                    f"{phase_label} Visibility update marked visible brick id(s): "
                    f"{sorted(restored)}."
                )
            if not newly_gone and not restored:
                print(f"{phase_label} Visibility update: no brick-state changes.")

        return gone_ids, newly_gone, restored, active

    def _print_xy_log(self, phase_label: Optional[str] = None) -> None:
        prefix = f"{phase_label} " if phase_label else ""
        tuples = ", ".join(str(state) for state in self.layer_push_states)
        print(f"{prefix}XY layer states: [{tuples}]")

    def _refresh_tower_from_vision(
        self, phase_label: str
    ) -> Tuple[
        Dict[int, Tuple[np.ndarray, np.ndarray]],
        set[int],
        Dict[int, Tuple[np.ndarray, np.ndarray]],
        GridResult,
    ]:
        gone_ids, _newly_gone, _restored, active = self._update_gone_from_visibility(
            phase_label
        )
        return self.last_reconstruction, gone_ids, active, self.last_grid

    def _enter_wait(self) -> None:
        self.player_wait_started_at = time.monotonic()
        self.player_wait_announced = False
        self.phase = Phase.WAIT

    def _end_game(self, outcome: str) -> None:
        self.game_outcome = outcome
        self.phase = Phase.COLLAPSE

    def _handle_successful_robot_push(
        self, phase_label: str, require_removed: bool = False
    ) -> bool:
        """Update from vision and return whether the push result is acceptable."""
        time.sleep(0.5)
        gone_ids, newly_gone, _restored, active = self._update_gone_from_visibility(
            phase_label
        )

        if len(newly_gone) >= 2:
            print(
                f"{phase_label} Robot push made {len(newly_gone)} brick(s) gone "
                "(no visible id with z > 0). "
                f"Machine failed. Active bricks: {len(active)}."
            )
            self._end_game(GAME_LOSE)
            return False

        if require_removed and not newly_gone:
            print(
                f"{phase_label} Tag visibility did not mark any new brick gone. "
                "Treating this push as failed."
            )
            return False

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
        if not self.tower_init_y_raised:
            print("[Phase TOWER_INIT] Moving Y up before tower reconstruction...")
            self._send_and_wait("MOVE Y 10000", timeout=2.0, ignore_errors=True)
            self._wait_for_done("Y")
            self.tower_init_y_raised = True

        if not self.tower_init_x_retracted:
            print("[Phase TOWER_INIT] Retracting X before tower reconstruction...")
            self._send_and_wait("MOVE X 10000", timeout=2.0, ignore_errors=True)
            self._wait_for_done("X")
            self.tower_init_x_retracted = True

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
                self._record_initial_brick_midpoints(fitted, self.last_grid)
                self._record_initial_visible_tags()
                print(
                    f"[Phase TOWER_INIT] Recorded {len(self.initial_brick_midpoints)} "
                    "initial brick midpoint(s) and "
                    f"{len(self.initial_layer_middle_brick_ids)} middle-brick id(s)."
                )
                print(
                    "[Phase TOWER_INIT] Recorded initially visible tag id(s) for "
                    f"{len(self.initial_tag_ids_by_brick)} brick(s)."
                )
                if self.xy_log:
                    self._print_xy_log("[Phase TOWER_INIT]")
                self._play_done_sound("TOWER_INIT sound")
                print("[Phase TOWER_INIT] Moving Y down after tower initialization...")
                self._send_and_wait("MOVE Y -10000", timeout=2.0, ignore_errors=True)
                self._wait_for_done("Y")
                self.current_layer_id = 0
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
                self._play_done_sound("A1 wait sound")
                self.player_wait_announced = True
            time.sleep(0.2)
            return
        if self.player_wait_announced:
            print("[Phase WAIT] Arduino A1 reads 0. Checking whether the player removed a brick...")
            self.player_wait_announced = False

        gone_ids, newly_gone, _restored, active = self._update_gone_from_visibility()

        if len(newly_gone) >= 2:
            print(
                f"[Phase WAIT] Player made {len(newly_gone)} brick(s) gone "
                "(no visible id with z > 0). "
                f"System wins. Active bricks: {len(active)}."
            )
            self._end_game(GAME_WIN)
            return

        if len(newly_gone) == 1:
            print(
                f"[Phase WAIT] Brick gone because no id is visible with z > 0. "
                f"Active bricks: {len(active)}."
            )
            self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
            self.phase = self.next_phase_after_wait
            return

        if self._check_collapse(len(active)):
            print(f"[Phase WAIT] COLLAPSE detected. Active bricks: {len(active)}")
            self._end_game(GAME_LOSE)
            return

        print("[Phase WAIT] No brick removed. Waiting for Arduino A1 to read 0 again.")
        self.player_wait_announced = False
        time.sleep(0.5)

    def _do_full_layer_1(self) -> None:
        print("[Phase FULL_LAYER_1] Selecting AUTO_PUSH layer from cached state...")
        active_count = sum(
            self._layer_active_count(state) for state in self.layer_push_states
        )
        if self._check_collapse(active_count):
            print(f"[Phase FULL_LAYER_1] COLLAPSE. Active bricks: {active_count}")
            self._end_game(GAME_LOSE)
            return

        # Find an AUTO_PUSH layer in sequential order. A cached layer is eligible
        # when the middle brick and at least one side brick are still present.
        target_layer: Optional[int] = None
        search_start = max(
            FULL_LAYER_SEARCH_FIRST_LAYER,
            min(self._full_layer_search_start, self.layer_count),
        )
        for y in range(search_start, self.layer_count):
            can_push, reason = self._layer_auto_push_candidate(y)
            if can_push:
                target_layer = y
                break
            state = self._layer_state(y)
            if self._layer_active_count(state):
                print(f"[Phase FULL_LAYER_1] Skipping layer {y}: {reason}.")

        if target_layer is None:
            print(
                "[Phase FULL_LAYER_1] No layer with middle plus side brick "
                "found. Randomly choosing a brick to move."
            )
            self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
            if self._try_random_cached_brick("[Phase FULL_LAYER_1 random]"):
                print("[Phase FULL_LAYER_1] Random push succeeded. Checking tower state.")
                if not self._handle_successful_robot_push(
                    "[Phase FULL_LAYER_1 random]", require_removed=True
                ):
                    if self.phase == Phase.COLLAPSE:
                        return
                    print(
                        "[Phase FULL_LAYER_1] Random push not confirmed by vision. "
                        "Retrying."
                    )
                    time.sleep(0.5)
                    return
                self._enter_wait()
                return
            print("[Phase FULL_LAYER_1] Random push failed. Retrying.")
            time.sleep(0.5)
            return

        can_push, reason = self._layer_auto_push_candidate(target_layer)
        if not can_push:
            print(
                f"[Phase FULL_LAYER_1] Skipping layer {target_layer} before AUTO_PUSH: "
                f"{reason}."
            )
            self._full_layer_search_start = target_layer + 1
            time.sleep(0.5)
            return

        state = self._layer_state(target_layer)
        push_mask = self._layer_push_mask(state)
        axis_flag = self._layer_axis_flag(state)
        if axis_flag == 0:
            print(
                f"[Phase FULL_LAYER_1] Layer {target_layer} is accessible. "
                f"Running AUTO_PUSH {push_mask:#05b} from state {state}."
            )
        else:
            print(
                f"[Phase FULL_LAYER_1] Layer {target_layer} has state "
                f"{state} but cached axis is X. Rotating tower before "
                "AUTO_PUSH."
            )
            try:
                self._rotate_z_with_x_retract(1)
            except (RuntimeError, TimeoutError) as exc:
                print(f"[Phase FULL_LAYER_1] Rotation failed: {exc}")
                self._full_layer_search_start = target_layer + 1
                time.sleep(0.5)
                return
            print("[Phase FULL_LAYER_1] DONE Z received. Using cached layer state.")

        can_push, reason = self._layer_auto_push_candidate(target_layer)
        if not can_push:
            print(
                f"[Phase FULL_LAYER_1] Skipping AUTO_PUSH on layer {target_layer}: "
                f"{reason}."
            )
            self._full_layer_search_start = target_layer + 1
            time.sleep(0.5)
            return

        state = self._layer_state(target_layer)
        push_mask = self._layer_push_mask(state)
        axis_flag = self._layer_axis_flag(state)
        if axis_flag != 0:
            print(
                f"[Phase FULL_LAYER_1] Layer {target_layer} is still cached as "
                "inaccessible after rotation. Retrying another layer."
            )
            self._full_layer_search_start = target_layer + 1
            time.sleep(0.5)
            return

        try:
            self._run_auto_push(target_layer, push_mask)
        except (RuntimeError, TimeoutError) as exc:
            print(
                f"[Phase FULL_LAYER_1] AUTO_PUSH failed on layer {target_layer}: {exc}. "
                "Retrying another layer."
            )
            self._advance_after_auto_push_failure(target_layer)
            time.sleep(0.5)
            return

        print("[Phase FULL_LAYER_1] AUTO_PUSH succeeded. Checking tower state.")
        if not self._handle_successful_robot_push(
            "[Phase FULL_LAYER_1]", require_removed=True
        ):
            if self.phase == Phase.COLLAPSE:
                return
            print(
                f"[Phase FULL_LAYER_1] AUTO_PUSH on layer {target_layer} was not "
                "confirmed by tag visibility. Retrying another layer."
            )
            self._full_layer_search_start = target_layer + 1
            time.sleep(0.5)
            return
        print("[Phase FULL_LAYER_1] Jumping to next phase.")
        self._full_layer_search_start = FULL_LAYER_SEARCH_FIRST_LAYER
        self._enter_wait()

    def _try_random_cached_brick(self, phase_label: str) -> bool:
        candidates: List[int] = []
        for y in range(0, self.layer_count):
            if y in self.na_layer:
                print(f"{phase_label} Skipping layer {y}: in na_layer.")
                continue
            state = self._layer_state(y)
            if self._layer_active_count(state):
                candidates.append(y)

        if not candidates:
            print(f"{phase_label} No cached bricks remain.")
            self._end_game(GAME_LOSE)
            return False

        target_y = random.choice(candidates)
        state = self._layer_state(target_y)
        axis_flag = self._layer_axis_flag(state)
        if axis_flag != 0:
            print(f"{phase_label} Layer {target_y} is cached as X-axis. Rotating first.")
            try:
                self._rotate_z_with_x_retract(1)
            except (RuntimeError, TimeoutError) as exc:
                print(f"{phase_label} Rotation failed: {exc}")
                return False
            state = self._layer_state(target_y)
            axis_flag = self._layer_axis_flag(state)

        if axis_flag != 0:
            print(f"{phase_label} Layer {target_y} is still not accessible.")
            return False

        slots_state = self._layer_slots(state)
        slots = [
            slot_id
            for slot_id in range(self.bricks_per_layer)
            if slots_state[slot_id] != EMPTY_BRICK_ID
        ]
        if not slots:
            print(f"{phase_label} Layer {target_y} has no cached bricks left.")
            return False

        target_x = random.choice(slots)
        brick_id = slots_state[target_x]
        print(
            f"{phase_label} Randomly pushing layer {target_y}, slot {target_x} "
            f"(brick {brick_id})."
        )
        try:
            self._execute_push(target_x, target_y)
        except (RuntimeError, TimeoutError) as exc:
            print(f"{phase_label} Random push error: {exc}")
            return False

        time.sleep(0.5)
        post_gone_ids = self._gone_bricks_from_current_visibility()
        if brick_id is None or brick_id in post_gone_ids:
            return True

        print(f"{phase_label} Brick {brick_id} still has a visible id with z > 0.")
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
        print("[Phase LASTLY] Updating cached grid from tag visibility...")
        _gone_ids, _newly_gone, _restored, _active = self._update_gone_from_visibility()
        grid_result = self.last_grid

        if self._check_collapse(grid_result.total):
            print(f"[Phase LASTLY] COLLAPSE. Active bricks: {grid_result.total}")
            self._end_game(GAME_LOSE)
            return

        pushed = self._try_random_cached_brick("[Phase LASTLY random]")

        if pushed:
            print("[Phase LASTLY] Push succeeded. Checking tower state.")
            if not self._handle_successful_robot_push(
                "[Phase LASTLY]", require_removed=True
            ):
                return
            self._enter_wait()
            return
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
    ap.add_argument(
        "--xy-log",
        action="store_true",
        help="Print cached (x, y) layer tuples after every tag visibility check.",
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
        xy_log=args.xy_log,
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
