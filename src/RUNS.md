```bash
uv run mjpython view_camera_location_calibrated.py --start-id 0\
  --cams 0 1 2 3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-in calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --bypass-calib
```

```
uv run jenga_controller.py --cams 0 1 2 3 --bypass-calib --calib-in calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz
```
