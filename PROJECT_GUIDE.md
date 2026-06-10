# MuJoCo AprilTag Calibration / Jenga Robot 项目说明书

> 本文档面向开发者与运维人员，概述系统架构、各模块工作原理、调试方法与预期能力。

---

## 一、项目概述

本项目是一个**基于 AprilTag 视觉的实体 Jenga（抽积木）机器人系统**。核心能力包括：

1. **多相机标定**：利用转盘上的 4 个 AprilTag（ID 300–303），通过 OpenCV `solvePnP` 求解最多 4 个相机在统一世界坐标系下的 3D 位姿。
2. **实时积木重建**：从已标定的相机中检测贴在积木上的 AprilTag，三角化其 3D 位置，并用加权 Kabsch 算法拟合每块积木的位姿。
3. **机器人控制**：通过自定义 Arduino 固件驱动三轴运动平台（X/Y 推杆 + Z 旋转）+ 直流推杆电机，实现自主抽积木。

系统遵循经典的 **感知–规划–执行（Perception-Planning-Actuation）** 闭环。

---

## 二、目录结构与主要文件

```
445-algo/
├── src/                                    # 主 Python 源码
│   ├── calibration_core.py                 # 标定数学核心
│   ├── reconstruction_core.py              # 多相机积木重建核心
│   ├── camera_protocol.py                  # 跨平台相机打开协议
│   ├── apriltag_utils.py                   # AprilTag 检测静音封装
│   ├── arduino_serial.py                   # Arduino 串口客户端
│   ├── view_camera_location.py             # MuJoCo 相机可视化 + 单次重建
│   ├── view_camera_location_calibrated.py  # 两阶段标定/可视化工具
│   ├── manual_camera_calibration.py        # 单相机手动标定工具
│   ├── jenga_controller.py                 # 顶层 Jenga 自动对局控制器
│   ├── assets/                             # MuJoCo 场景、网格、贴图
│   ├── calib_cam0.npz ... calib_cam3.npz   # 保存的相机标定结果
│   └── RUNS.md                             # 常用命令速查
├── arduino/                                # Arduino 固件
│   └── firmware_custom_pins_BTS7960_REN_LEN_VCC/
│       └── firmware_custom_pins_BTS7960_REN_LEN_VCC.ino
├── pyproject.toml                          # Python 包配置与入口点
├── requirements.txt
├── uv.lock
└── README.md
```

---

## 三、各模块详解

### 3.1 标定核心 `calibration_core.py`

- **职责**：相机内参计算、OpenCV↔MuJoCo 坐标系转换、基于 4 个标签中心的 PnP 位姿求解。
- **关键符号**：
  - `TURNTABLE_TAG_IDS = (300, 301, 302, 303)`：转盘 AprilTag 编号。
  - `S_CV2MJ = np.diag([1.0, -1.0, -1.0])`：OpenCV 相机坐标系到 MuJoCo 坐标系的对角变换（Y/Z 翻转）。
- **核心函数**：
  - `K_from_fovy_mujoco(fovy_deg, w, h)`：由垂直视场角构造内参矩阵 `K`。
  - `build_square_frame(world_points)`：用 4 个标签中心构建局部正交坐标系 G，输出 `R_w_g`、`t_w_g`。
  - `solve_camera_pose_from_square_centers(centers_uv, K, R_w_g, t_w_g)`：对 4 个标签中心做 PnP，优先用 `SOLVEPNP_IPPE`；对多解情况，按**朝向中心点的角度**与**相机高度**打分选优，返回 `R_w_c_mj`（MuJoCo 风格旋转）、`p_w_c`（世界位置）、重投影误差。

**调试提示**：若标定结果异常，检查标签顺序是否与 `--world-points` 一致；重投影误差 `reproj` 应小于 1–2 像素。

---

### 3.2 重建核心 `reconstruction_core.py`

- **职责**：接收多相机一帧观测，检测积木 AprilTag，估计每块积木的世界位姿。
- **数据结构**：
  - `CameraObsInput`：单帧输入（RGB、K、相机位姿）。
  - `ReconConfig`：标签族、标签尺寸、`S_cv2mj`、是否强制底面平行地面等。
  - `BlockFitResult`：单块积木的拟合结果（R、t、RMS、观测数等）。
- **核心算法**：
  - `weighted_kabsch(P, Q, w)`：带权重的 Kabsch 刚性配准，求解 `R, t` 使得 `R*P + t ≈ Q`，返回加权 RMS。
  - `enforce_one_face_parallel_ground(R)`：将积木旋转约束到“某一面平行地面”。Jenga 积木只能直立，此约束把局部 6 个面法向之一强行对齐世界 Z 轴。
  - `solve_two_points_with_flat(P2, Q2)`：仅看到 2 个标签时的闭式位姿解（枚举 6 个面 × 2 个符号）。
  - `tag_confidence(...)`：综合标签像素边长、倾斜角（法向与相机 Z 轴夹角）、图像中心距离、深度 4 项，输出 `[0,1]` 置信度，用于加权融合。
- **主入口**：
  - `reconstruct_blocks_multi_cam(cams, detector, tag_local, ...)`：逐相机检测 → 筛选积木标签（`start_id ≤ tid ≤ max_block_tag_id`）→ 计算各标签世界坐标 → 按 `brick_index = (tid - start_id) // 10` 分组 → 若观测点 ≥3 用 Kabsch；若 1/2 点且启用 `enforce_flat` 用降级策略 → 输出每块积木位姿。

**调试提示**：
- 重建失败常见原因：相机标定未锁定、标签尺寸 `--tag-size` 与实际不符、光照不足导致 `pupil-apriltags` 漏检。
- 开启 `--enforce-flat` 可在积木直立假设下，用极少标签（甚至 1 个）恢复大致位姿。

---

### 3.3 相机协议 `camera_protocol.py`

- **职责**：统一封装 Linux（Debian/V4L2）与 macOS/其他平台的相机打开逻辑。
- **关键行为**：
  - Linux 默认使用 `debian` 协议：数字相机参数 `0` 自动映射到 `/dev/video0`；支持 `v4l2://` 显式前缀。
  - 默认请求 `MJPG` FOURCC；若相机不支持，可用 `--camera-fourcc none` 或 `YUYV`。
  - 设置 `CAP_PROP_BUFFERSIZE = 1` 降低延迟。

---

### 3.4 AprilTag 静音封装 `apriltag_utils.py`

- **问题**：`pupil-apriltags` 底层 C 库会在 stdout 输出 `"Error, more than one new minima found."` 等噪声。
- **解决**：`detect_apriltags_silent()` 通过 `os.dup2` 重定向 stdout/stderr，过滤已知噪声行后回写，配合线程锁保证多线程安全。

---

### 3.5 Arduino 串口客户端 `arduino_serial.py`

- **职责**：与自定义 Arduino 固件通信。
- **自动发现**：遍历串口，按关键词（`arduino`、`usbmodem`、`ch340`、`cp210` 等）打分，选最高分端口；若只有一个端口则直接选用。
- **协议**：115200 baud，`\n` 结尾。
- **命令**：`INIT`、`MOVE`、`BM`（Block-wise Move）、`DC`（直流电机）、`STATUS?`、`AUTO_PUSH`。
- **响应解析**：`is_terminal_response()` 判定 `OK`、`PONG`、`INIT DONE`、`ERR ...`、`DONE ...`、`POS ...` 等终结行。

**调试提示**：若连接失败，先运行 `python src/arduino_serial.py ports` 查看可用端口；确认波特率与固件一致。

---

### 3.6 MuJoCo 可视化 `view_camera_location.py`

- **职责**：在 MuJoCo 3D 视窗中实时显示相机视锥与重建的积木。
- **交互**：
  - 后台线程持续对各相机做转盘标签检测并实时估计相机位姿。
  - **按 SPACE**：保存当前标定 + 执行一次多相机积木重建，结果以半透明青色方块画在场景中，并写入 `poses_live.txt`。
- **可视化元素**：
  - 实时位姿视锥（不透明）+ 已保存位姿视锥（半透明）。
  - 重建出的积木以 `BAR_HALF = (0.5, 0.5, 1.5)` 的半尺寸箱子绘制。

---

### 3.7 两阶段标定工具 `view_camera_location_calibrated.py`

- **职责**：更完整的标定与可视化流程。
- **Phase 1（标定）**：后台线程持续 PnP，OpenCV 窗口显示每相机的标签检测与距离校验（`world_dist` vs `image_dist`）。
  - **按 M**（或在 OpenCV 窗口按 Space/S/M）：冻结当前位姿，保存 `.npz`。
- **Phase 2（标签可视化）**：标定冻结后，忽略相机运动，将每帧检测到的 AprilTag 世界坐标以小球形式画在 MuJoCo 中；同一标签多相机观测取平均。
- **积木对齐**：`compute_aligned_lines()` 从可见标签拟合积木位姿，将长轴 snapped 到 ±X/±Y，中心 XY 取整、Z 取 `n+0.5`，并在场景中画出对齐后的圆柱与砖块模型。

**调试提示**：
- `--no-viewer` 模式仅使用 OpenCV 窗口，适合无显示器或远程运行。
- `--bypass-calib` 跳过实时标定，直接加载 `--calib-in` 的 `.npz`。

---

### 3.8 单相机手动标定 `manual_camera_calibration.py`

- **用途**：最简化的单路相机标定，无需 MuJoCo。
- **操作**：打开相机 → 检测 4 个转盘标签 → 实时显示 PnP 结果与距离校验 → **Space/S 保存** → **Q/Esc 退出**。
- **输出**：`calib_{name}.npz`，包含 `K`、`dist_coeffs`、`cam_pos_w`、`R_w_c`、重投影误差及距离对比表。

---

### 3.9 顶层控制器 `jenga_controller.py`

- **职责**：整合 4 相机视觉、Arduino 运动、层状态机，实现人机对局。
- **状态机**：

```
INIT -> TOWER_INIT -> WAIT -> FULL_LAYER_1 -> WAIT -> LASTLY -> WAIT ...
                                      |
                                 COLLAPSE（紧急停止）
```

- **各阶段说明**：
  1. **INIT**：发送 `INIT` 到 Arduino（回零）；等待所有相机完成 PnP 标定并冻结位姿。
  2. **TOWER_INIT**：Y 轴抬升、X 轴回退 → 反复执行多相机重建，直到连续 3 秒检测到 24 块积木（`layer_count * bricks_per_layer`）→ 记录初始中层点、每层的 slot 映射、各砖初始可见标签 → 进入等待。
  3. **WAIT**：读取 Arduino A1 引脚；当 A1 == 0 时认为玩家回合结束，通过**标签可见性**判断是否有砖被抽走（若某砖所有初始标签在 z>0 区域均不可见，则标记为 gone）。
     - 若一次性消失 ≥2 块 → 判定系统胜利（玩家碰倒）。
     - 若当前活跃砖 <4 → 判定倒塌失败。
  4. **FULL_LAYER_1**：机器人选择一层执行 `AUTO_PUSH`。
     - 优先寻找“中间砖仍在且至少一侧有砖”的完整层。
     - 若缓存层方向不是推杆方向（axis_flag != 0），先旋转 Z 轴 90°。
     - 发送 `AUTO_PUSH {mask}`，`mask` 按 slot 位掩码编码。
  5. **LASTLY**：当没有标准完整层可选时，随机选取缓存中仍存在的砖，执行基础推杆动作。
  6. **COLLAPSE**：停止所有运动、关闭 DC 电机、记录最终位姿到 `collapse_poses.txt`、释放相机并退出。

- **标签可见性判定**：`_visible_initial_tags_by_brick()` 遍历各相机当前帧，检测属于某砖的初始标签，计算其世界 z 坐标（经 `scale` 缩放后），若 z>0 则认为可见。这比纯重建更鲁棒，可检测“砖被抽走”这一离散事件。
- **层状态缓存**：`LayerState = (axis_flag, slot0, slot1, slot2)`。Z 每旋转一次，所有层的 `axis_flag` 翻转，slot 顺序反转。

**调试提示**：
- `--bypass-calib`：跳过 INIT 中的实时标定，直接加载已有的 `.npz`。
- `--live`：在 MuJoCo 视窗中实时显示当前重建的塔（更新率 4 Hz）。
- `--xy-log`：每次可见性更新后打印层状态元组，便于追踪逻辑。

---

### 3.10 Arduino 固件 `firmware_custom_pins_BTS7960_REN_LEN_VCC.ino`

- **硬件平台**：Arduino UNO R3。
- **轴定义**：
  - X / Y：外部 2D45A 步进驱动器，光耦共阳极接法（脉冲拉低）。
  - Z：板载 TMC2209，逻辑电平直驱（脉冲高）。
- **限位**：X/Y 各 2 路光电 NPN 开漏（INPUT_PULLUP），触发电平 LOW。
- **直流电机**：BTS7960 H 桥；因 D2 非硬件 PWM，固件使用 **Timer2 软件 PWM**（约 976 Hz）。步进调度由 **Timer1 ISR（12 kHz）**完成。
- **运动规划**：梯形加减速，定点数 `rate_fp` 累加发脉冲，ISR 中只判断 `phase_fp >= RATE_FP_ONE` 即发步进脉冲。
- **关键指令**：
  - `INIT`：X/Y 回负限位 → X 前进到 `INIT_X_OFFSET + 638`。
  - `AUTO_PUSH [mask]`：状态机自动流程——
    1. X 回零 → 前进到 offset。
    2. 按 mask 遍历 slot：每个 slot 先以低速 DC（duty=80）探测；若 A0 接地（analogRead==0）说明碰到砖，则反转 DC 退出；若 3.25s 内未接地，视为空槽，记录 success。
    3. 探测成功后以 duty=180 持续推 5s → 停 2s → 反转 5s 松脱 → 低速反转 2s → X 回零。
    4. 状态机结束时上报 `AUTO_PUSH SUCCESS <slot>` 或 `WARN: AUTO_PUSH FAILED`。
  - `BM X|Y|Z <blocks>`：Block-wise Move，X/Y 每 block = 638 步，Z 每 90° = 400 步。
  - `MOVE`：直接按步数运动，支持 `F`（速度）、`A`（对称加减速）、`AU/AD`（非对称加减速）。
  - `DC F/R/0`：直流电机正转/反转/停止。

---

## 四、快速开始

### 4.1 环境准备

```bash
cd /Users/pr/Documents/workspace/445-algo
uv sync          # 使用 uv 安装依赖并创建虚拟环境
```

依赖包括：MuJoCo 3.4.0、OpenCV 4.13、pupil-apriltags、numpy、pyserial、matplotlib 等。

### 4.2 仅 OpenCV 标定（无 MuJoCo  Viewer）

```bash
uv run view-camera-location-calibrated \
  --cams 0 1 2 3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-out calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --show-live-capture \
  --no-viewer
```

Linux/Debian 下推荐使用 V4L2 设备路径：

```bash
uv run view-camera-location-calibrated \
  --cams /dev/video0 /dev/video1 /dev/video2 /dev/video3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-out calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --show-live-capture \
  --no-viewer
```

若相机不支持 `MJPG`，加 `--camera-fourcc YUYV` 或 `--camera-fourcc none`。

### 4.3 macOS MuJoCo Viewer 模式

```bash
mjpython -m mujoco_apriltag_calibration.view_camera_location_calibrated \
  --cams 0 1 2 3 \
  --cam-names cam0 cam1 cam2 cam3 \
  --calib-out calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz
```

> 注：macOS 下 `cv2.imshow` 与 MuJoCo viewer 同时运行可能冲突，此时优先用 MuJoCo 的 `M` 键冻结标定。

### 4.4 运行自动对局

先完成标定并保存 4 个 `.npz`，然后：

```bash
uv run jenga_controller.py \
  --cams /dev/video0 /dev/video1 /dev/video2 /dev/video3 \
  --calib-in calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --bypass-calib
```

若想在 MuJoCo 中实时看塔：

```bash
uv run jenga_controller.py \
  --cams /dev/video0 /dev/video1 /dev/video2 /dev/video3 \
  --calib-in calib_cam0.npz calib_cam1.npz calib_cam2.npz calib_cam3.npz \
  --bypass-calib --live
```

---

## 五、调试指南

### 5.1 相机类问题

| 现象 | 排查方法 |
|------|----------|
| 相机打不开 | 检查 `ls -l /dev/v4l/by-id/` 中的稳定 symlink；确认无其他进程占用；尝试 `--camera-protocol auto`。 |
| 画面全绿/花屏 | FOURCC 不匹配，尝试 `--camera-fourcc YUYV` 或 `none`。 |
| 标签检测率低 | 检查光照（避免过曝/欠曝）；调整 `--decimate`（越小越精细但越慢）；确认 `--tag-size` 与实际打印尺寸一致。 |
| 标定重投影误差大 | 确认标签打印无拉伸；检查 `--world-points` 顺序与物理布置是否对应；确认内参 `K` 正确（或先用 `--fovy` 近似）。 |

### 5.2 重建类问题

| 现象 | 排查方法 |
|------|----------|
| 重建砖数远少于 24 | 检查相机是否全部标定锁定；确认积木标签 ID 范围（默认 `start-id=0`，`max_block_tag_id=239`）；在 `view_camera_location_calibrated.py` 中看标签世界坐标是否合理。 |
| 积木位姿漂移/倒伏 | 开启 `--enforce-flat`；检查 `scale = mujoco_tag_size / real_tag_size` 是否正确。 |
| Z 高度不准 | 受单目深度精度限制，属于正常现象；如仅需层判断，当前精度足够。 |

### 5.3 Arduino / 运动类问题

| 现象 | 排查方法 |
|------|----------|
| 串口无响应 | 确认波特率 115200；运行 `python src/arduino_serial.py ping` 测试；检查 USB 线是否仅供电而无数据。 |
| INIT 回零失败 | 检查限位接线（NPN 开漏需 INPUT_PULLUP）；运行 `LIMITS?` 查看当前限位状态；确认 `LSX1/LSY1` 在负方向。 |
| AUTO_PUSH 总是失败 | 检查 A0 模拟输入是否接地正确；调整 `AUTO_PUSH_DC_2S_MS` 等阈值；在固件中开启 `A0TEST 1` 实时打印 A0 值。 |
| X/Y 运动方向反了 | 修改固件中 `dir_positive_level` 或交换驱动器 DIR 接线。 |
| DC 电机不转 | 确认 BTS7960 供电足够（大电流）；检查 `PIN_RPWM/PIN_LPWM` 接线；确认 Timer2 软件 PWM 未被其他库覆盖。 |

### 5.4 对局逻辑类问题

| 现象 | 排查方法 |
|------|----------|
| 玩家抽砖后系统未检测到 | 检查标签可见性逻辑：被抽走的砖是否仍有标签残片留在视野内；调整判定阈值或要求更高 `z>0` 标准。 |
| 机器人连续推同一层失败 | 查看 `na_layer` 列表；若某层被标记为 `na`，系统会自动跳过；清除 `na_layer` 或检查物理原因。 |
| COLLAPSE 误触发 | 若砖只是倾斜但标签仍可见，不会误触发；若 `<4` 块活跃砖则触发，可调整 `LAYER_Z_THRESHOLD` 或 `layer_count`。 |

---

## 六、预期能力与限制

### 6.1 系统能做什么

1. **全自动标定**：在转盘标签可见的情况下，4 相机可在数秒内完成联合世界坐标标定。
2. **24 砖实时重建**：在良好光照与标定条件下，可稳定检测并重建 24 块 Jenga 积木的 3D 位姿。
3. **人机对局**：玩家回合结束后，系统自动识别缺失砖块，随后自主规划并执行推杆动作。
4. **跨平台**：支持 macOS（开发/可视化）与 Debian/Linux（部署/V4L2 相机）。
5. **模块化**：标定、重建、可视化、控制各模块可独立运行，便于分别调试。

### 6.2 已知限制

1. **单目深度精度**：`pupil-apriltags` 的 `pose_t` 来自单目 PnP，深度方向误差通常大于 XY；多相机融合可改善，但非结构光/双目精度。
2. **标签遮挡**：若某砖所有 10 个标签被完全遮挡，则只能依赖缓存状态推断其存在；系统不处理标签部分脱落。
3. **物理一致性**：`enforce_flat` 仅约束旋转，不约束积木之间不穿透；重建结果可能出现轻微重叠。
4. **Arduino UNO 资源**：Timer1/Timer2 已分别用于步进调度与 DC 软件 PWM，不可再使用 `Servo` 库等冲突资源。
5. **Z 轴无硬件限位**：旋转依赖开环步数，若失步需重新 `INIT`。

---

## 七、CLI 入口点速查

| 命令 | 说明 |
|------|------|
| `uv run view-camera-location` | MuJoCo 实时相机可视化 + Space 重建 |
| `uv run view-camera-location-calibrated` | 两阶段标定（M 冻结）+ 标签/砖块可视化 |
| `uv run jenga-controller` | 自动对局主程序 |
| `python src/manual_camera_calibration.py --cam 0` | 单相机手动标定 |
| `python src/arduino_serial.py ports` | 列出串口 |
| `python src/arduino_serial.py init` | 发送 INIT |
| `python src/arduino_serial.py command "MOVE X 1000 F 800 A 600"` | 发送任意运动指令 |

---

## 八、坐标系约定

- **世界坐标系**：MuJoCo 风格，Z 向上，单位为 MuJoCo 内部单位（由 `--mujoco-tag-size` 决定比例）。
- **OpenCV 相机坐标系**：Z 向前，X 向右，Y 向下。
- **转换**：`p_mujoco = S_CV2MJ @ p_cv`，其中 `S_CV2MJ = diag(1, -1, -1)`。
- **积木局部坐标**：长轴为局部 +Z，半尺寸 `(0.5, 0.5, 1.5)`；10 个 AprilTag 分布在两端与四个侧面的上下边缘。

---

> 文档版本：2026-06-10 | 对应分支：`raspberry-pi`
