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

## Debian / Linux cameras

On Debian, numeric camera arguments are opened as V4L2 devices:

```bash
uv run view-camera-location-calibrated \
  --cams /dev/video0 /dev/video1 /dev/video2 /dev/video3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-out calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --show-live-capture \
  --no-viewer
```

`--cams 0 1 2 3` is equivalent on Linux. For stable multi-camera ordering, prefer
the symlinks shown by:

```bash
ls -l /dev/v4l/by-id/
```

If a camera does not support the default `MJPG` V4L2 format, pass
`--camera-fourcc none` or another format such as `YUYV`.

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
