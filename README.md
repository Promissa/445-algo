# MuJoCo AprilTag Calibration

Package-style transport of the camera calibration and reconstruction tools from the original MuJoCo project.

## Setup

```bash
cd /Users/pr/Documents/workspace/445-algo
uv sync
```

## OpenCV-only calibration

```bash
uv run view-camera-location-calibrated \
  --cams 0 1 2 3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-out calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --show-live-capture \
  --no-viewer
```

## MuJoCo viewer mode on macOS

```bash
mjpython -m mujoco_apriltag_calibration.view_camera_location_calibrated \
  --cams 0 1 2 3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-out calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz
```

## TODO

1. Action after all layers tried to be determined
2. Rasperry Pi 5 transplant