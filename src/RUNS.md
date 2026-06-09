```bash
uv run mjpython view_camera_location_calibrated.py --start-id 0\
  --cams /dev/video0 /dev/video1 /dev/video2 /dev/video3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-in calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --bypass-calib
```

```
uv run jenga_controller.py --cams /dev/video0 /dev/video1 /dev/video2 /dev/video3 --bypass-calib --calib-in calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz
```
