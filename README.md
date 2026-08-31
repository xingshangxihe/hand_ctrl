# hand-ctrl：端侧隔空手势控制器（单指控制方案）

> Linux 端侧 C++17 纯离线隔空手势控制器，向系统注入鼠标事件。
> 当前版本：**v2.6.2（捏合单击/双击 + 食指小指右键 + 双指V拖拽方案）**——开掌锁定，掌心相对位移控制鼠标，捏合单击/双击，食指小指右键，双指V拖拽。
>
> v1.0（六手势方案）已废弃：握拳短按、OK、竖拇指、食指滑动等姿态手势。
> v2.6.2 保留：开掌锁定/解锁 + 掌心相对位移 + 捏合单击/双击 + 食指小指右键 + 双指V拖拽。

## 项目定位

- 不依赖创意新颖度，依靠**标准化工程架构、Linux 底层字符设备调用、端侧模型部署优化、时序防抖算法**构建差异化竞争力。
- 开发环境限定 PC / VMware Ubuntu 22.04（WSL 亦可，命令通用）。
- 推理层预留 RKNN 抽象接口，后续仅替换子类即可无缝移植 RV1106 瑞芯微开发板。

## 开源协议

[MIT License](LICENSE) —— 任何人可自由使用、修改、商用，只需保留版权声明。

## v2.6 交互模型（唯一基准，全部阈值外置 `config/gesture_config.json`）

**核心思想：单击/双击（捏合）、右键（食指小指）、拖拽（双指V）、移动（食指伸出）完全分离。**

| 动作 | 判定规则 | 输出事件 |
| ---- | -------- | -------- |
| 五指张开长按 1s | 5 根手指全部伸直（指尖/指根比值 > 阈值）持续 1s | 锁定/解锁切换（IDLE↔LOCKED） |
| 手移动 | 食指伸出（非捏合/非食指小指/非双指V），掌心帧间位移 | 鼠标相对位移（触控板式，可覆盖整个屏幕） |
| 捏合 tap | 拇指尖(4)-食指尖(8)距离 ÷ 中指根(9)-手腕(0)距离 < 阈值 | 左键单击 |
| 快速捏合两次 | 两次捏合 tap 间隔 < `double_click_interval_ms` | 左键双击 |
| 食指+小指伸出 | 食指(8)+小指(20)伸直、中指(12)+无名指(16)弯曲 | 右键单击 |
| 双指V手势 | 食指+中指伸直（比值>阈值）、无名指+小指弯曲（距离<阈值） | 左键按住（拖拽开始） |
| 双指V保持移动 | 左键按住时手移动 | 鼠标跟手移动（拖拽） |
| 解除双指V | 收手（不再满足双指V） | 释放左键（拖拽结束） |

> **手势设计说明**：
> - 锁定/解锁用「五指张开」——与握拳/捏合区分度极高。判定用"指尖到手腕 /
>   指根到手腕"比值法，对"手指朝屏幕"的透视投影缩短鲁棒。
> - **单击与拖拽用完全不同的两个动作**（v2.5 核心改进）——避免"用同一动作靠
>   时长区分"带来的模糊地带。
>   - 单击 = **捏合**（拇指+食指 tap）：拇指尖与食指尖距离判定，独立可靠；
>     瞬时动作不累，边沿触发防连发。
>   - 拖拽 = **双指V**（食指+中指伸直、无名指+小指弯曲）：显式手型，与移动
>     （单指/张手）和单击（捏合）完全不同，绝无冲突。摆V即按住，收手即释放。
> - 移动用「相对位移」（触控板式）而非绝对定位——手活动范围有限做不到绝对
>   定位全覆盖；相对位移可累计，手在小范围内来回移动多次即可到达屏幕任意位置。

## 差异化优势

- **纯 C++ 全链路实现**：采集→推理→FSM→uinput 全部 C++17，无 Python 依赖
- **完全离线运行**：无需网络，推理在本地 CPU 完成
- **单指控制**：只需记住一根手指，学习成本极低（v2 核心改进）
- **绝对定位**：食指指尖 = 鼠标指针，指哪打哪，无需记忆"手势与功能的映射"
- **开掌锁定防误触**：张开五指 1s 锁定后控制才生效，大幅降低误触率
- **端到端延迟 <50ms**：全链路 30fps 稳定
- **推理框架可插拔**：抽象基类 `InferBase`，ONNX/NCNN/RKNN 子类自由切换

## 目录结构

```
hand-ctrl/
├── CMakeLists.txt            # 全局编译规则：C++17、OpenCV、ONNX Runtime 依赖链接
├── LICENSE                   # MIT 开源协议
├── config/
│   └── gesture_config.json   # 全部阈值外置：屏幕分辨率/锁定/点击/拖拽参数
├── models/                   # 不入库（体积大），按下方"模型获取"章节自行生成
│   ├── hand_landmark.onnx    # 手部关键点模型（MediaPipe Hands 转换）
│   ├── palm_detection.onnx   # 手掌检测模型（过滤脸部/非手，双模型必需）
│   └── ncnn_model/           # 早期 NCNN 模型目录（现方案用 ONNX）
├── src/
│   ├── main.cpp              # 程序入口、多线程调度器、绝对定位实现
│   ├── camera_capture/       # V4L2 直读摄像头封装（YUYV 未压缩优先，MJPG 回退）
│   ├── infer_base/           # 抽象基类 InferBase + ONNX/NCNN/RKNN 三个子类
│   ├── gesture_fsm/          # 卡尔曼滤波 + v2 单指状态机
│   ├── linux_uinput/         # uinput 虚拟键鼠，RAII 自动释放
│   └── utils/common.h        # 通用常量/枚举/结构体收拢
└── README.md
```

## 开发进度

| 阶段 | 内容 | 状态 |
| ---- | ---- | ---- |
| 0-7 | 七阶段完整开发（v1.0 六手势方案） | ✅ 已完成 |
| 7+ | 阶段7收尾：按键映射自定义/配置热加载/窗口关闭检测 | ✅ 已完成 |
| 8 | v2 重构：单指控制方案（废弃六手势，改为食指绝对定位+点击/拖拽） | ✅ 已完成 |

## 编译与运行

### 依赖安装

```bash
# Ubuntu / Debian
sudo apt install -y build-essential cmake git libopencv-dev libeigen3-dev

# ONNX Runtime C++ 库（需头文件 + 库，见下文说明）
```

**ONNX Runtime C++ 头文件获取**（Ubuntu 无 apt 包时，实测版本 1.28.0）：
```bash
# 1. 安装 Python 包获取 C 库（含 libonnxruntime.so）
pip install --break-system-packages onnxruntime
# 库路径：~/.local/lib/python3.x/site-packages/onnxruntime/capi/libonnxruntime.so

# 2. C++ 头文件：pip 包内不带，需从 PyPI 源码包提取（无需外网，走 pip 源）
pip download onnxruntime==1.28.0 --no-binary :all: --no-deps -d /tmp/ort_src
cd /tmp/ort_src && tar -xzf onnxruntime-1.28.0.tar.gz
mkdir -p ~/onnxruntime_cpp/include/onnxruntime
cp /tmp/ort_src/onnxruntime-1.28.0/include/onnxruntime/onnxruntime_cxx_api.h   ~/onnxruntime_cpp/include/onnxruntime/
cp /tmp/ort_src/onnxruntime-1.28.0/include/onnxruntime/onnxruntime_cxx_inline.h ~/onnxruntime_cpp/include/onnxruntime/
cp /tmp/ort_src/onnxruntime-1.28.0/include/onnxruntime/onnxruntime_c_api.h     ~/onnxruntime_cpp/include/onnxruntime/

# 3. 创建符号链接（libonnxruntime.so → 具体版本号库），供链接器找到
ln -sf ~/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.28.0 \
       ~/.local/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so
```

> 早期尝试过 jsdelivr CDN / GitHub raw 下载头文件均失败（onnxruntime 仓库体积过大
> 被 CDN 拒绝，GitHub raw 在部分网络不可达），PyPI 源码包提取是最可靠方案。

### 模型获取

双模型均已验证可用：`hand_landmark.onnx`（手部关键点）+ `palm_detection.onnx`（手掌检测，用于过滤脸部/非手误检）。

```bash
# 从 pip mediapipe 包提取模型（无需外网）
pip download mediapipe==0.10.15 --no-deps -d /tmp/mp
cd /tmp/mp && mkdir -p ex && python3 -m zipfile -e mediapipe-*.whl ex/
mkdir -p ~/hand_ctrl/models
cp ex/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite ~/hand_ctrl/models/
cp ex/mediapipe/modules/palm_detection/palm_detection_lite.tflite ~/hand_ctrl/models/

# 转换 ONNX（tflite2onnx 轻量转换，无需 tensorflow；版本需 ≥1.16）
pip install --break-system-packages tflite2onnx
cd ~/hand_ctrl/models
python3 -c "import tflite2onnx; tflite2onnx.convert('hand_landmark_lite.tflite', 'hand_landmark.onnx')"
python3 -c "import tflite2onnx; tflite2onnx.convert('palm_detection_lite.tflite', 'palm_detection.onnx')"

# 校验：确认两个 onnx 文件均生成且大小 > 1MB
ls -la hand_landmark.onnx palm_detection.onnx
```

> 说明：早期 tflite2onnx 版本 palm_detection 转换会失败（输出布局 bug），
> 升级到新版后双模型转换均成功。若 palm_detection.onnx 缺失，程序自动
> 回退到"单模型模式"（用画面中央 60% 区域代替手掌框），功能受限但可运行。

### 编译

```bash
cd ~/hand_ctrl
mkdir -p build && cd build
cmake .. \
  -DONNXRUNTIME_INCLUDE_HINT=$HOME/onnxruntime_cpp/include/onnxruntime \
  -DONNXRUNTIME_LIB_HINT=$HOME/.local/lib/python3.12/site-packages/onnxruntime/capi
make -j$(nproc) hand_ctrl
```

> 头文件目录：含 `onnxruntime_cxx_api.h` 的目录（上文提取到 `~/onnxruntime_cpp/include/onnxruntime`）
> 库目录：含 `libonnxruntime.so` 符号链接的 capi 目录

### 运行（需 sudo，访问 /dev/uinput）

```bash
cd ~/hand_ctrl
sudo ./build/hand_ctrl config/gesture_config.json models 0        # 无 GUI，完整日志
sudo ./build/hand_ctrl config/gesture_config.json models 0 -q     # 无 GUI，静默（正式使用）
sudo ./build/hand_ctrl config/gesture_config.json models 0 -v     # 带实时画面
sudo ./build/hand_ctrl config/gesture_config.json models 0 -v -q  # 画面 + 静默
```

**首次使用**：先确认 `config/gesture_config.json` 的 `screen.screen_width/height`
与你的实际屏幕分辨率一致（绝对定位映射目标）。

### 使用步骤（v2.6.2 捏合单击/双击 + 食指小指右键 + 双指V拖拽）

1. **锁定**：对摄像头张开五指并保持 1s → 状态变 `LOCKED`
2. **移动鼠标**：伸出食指（或张开手），手移动 → 鼠标相对移动（触控板式，手可小范围来回划动，鼠标能到屏幕任意位置）
3. **单击**：拇指+食指快速捏合一下（tap，如点击屏幕上按钮）
4. **双击**：快速捏合两下（间隔 <0.5s，如双击打开文件/程序）
5. **右键**：食指+小指同时伸出（中指、无名指弯曲）——在食指控制时只需再伸出小指即可切换
6. **拖拽**：摆出"双指V"（食指+中指伸出比胜利）→ 按住左键；保持V手势移动 → 拖动窗口/文件；收手 → 释放
7. **解锁**：再次张开五指保持 1s → 状态回 `IDLE`

> **移动提示**：相对位移可累计——手移到画面边缘后，可收回手继续向同一方向划，
> 鼠标会持续移动。若感觉移动太快/太慢，调 `mouse.gain_x/gain_y`（默认 2.0）。

### 配置热加载（实时生效）

运行中直接编辑 `config/gesture_config.json`，约 1 秒后自动生效：
- `lock.*`/`open_palm.*`/`click.*`/`drag.*` 阈值 → FSM 立即采用新值
- `screen.screen_width/height`、`mouse.gain_x/gain_y` → 相对位移映射立即更新
无需重启程序，控制台会打印 `[GestureFSM] 配置热加载成功`

### 各模块独立测试

```bash
# 摄像头测试（保存帧验证，绕开 imshow 卡顿）
make camera_test && ./camera_test 0 640 480 -s

# 推理测试
make infer_test && ./infer_test models -img 测试图片.jpg

# FSM 测试（v2 状态机：锁定/定位/点击/拖拽事件，仅打印不注入）
make fsm_test && ./fsm_test config/gesture_config.json models 0

# uinput 测试
make uinput_test && sudo ./uinput_test
```

## 为什么需要 sudo

运行需要 root 权限访问 `/dev/uinput` 字符设备。可选授权方式（无需每次 sudo）：

```bash
sudo usermod -aG input $USER    # 加入 input 用户组
# 重新登录后生效
```

## 推理框架选型说明（NCNN → ONNX Runtime）

**原方案（NCNN）**：需求文档指定 NCNN，但开发中遇到障碍：
1. 新版 NCNN 已移除 `onnx2ncnn` 转换工具
2. 虚拟机无法访问 GitHub，转换工具链无法下载

**现方案（ONNX Runtime）**：
- 安装简单、MediaPipe ONNX 模型源更丰富
- **ONNX 是 RV1106 移植 RKNN 的官方最优输入格式**（RKNN-Toolkit2 原生支持 ONNX）
- 推理层抽象基类 `InferBase` 不变，仅替换子类，上层业务零改动

`src/infer_base/` 下三个平级子类：
- `onnx_infer.*`：**当前主推理实现**（ONNX Runtime）
- `ncnn_infer.*`：保留作对比/移植参考（不参与默认构建）
- `rknn_infer.*`：RV1106 移植预留空壳

## VMware 环境适配记录（踩坑汇总）

### 摄像头 MJPG 流损坏（画面卡死）
- **根因**：VMware USB 重定向传输的 MJPG 帧流在驱动层损坏（libjpeg 报
  `Corrupt JPEG data`），表现为解码出半张坏图 → palm 检测不到手 → 画面卡死。
  重开设备也拿不到干净流（损坏在 VMware/驱动层）。
- **解决**：优先改用 **YUYV 未压缩格式**（无 JPEG 解压环节，从源头消除损坏），
  V4L2 直读像素 + `cvtColor` 转 BGR；摄像头不支持 YUYV 时自动回退 MJPG
  （SOI/EOI 区间截取 + `imdecode` 宽容解码）
- **实测**：YUYV 模式下长时间运行不再卡死

### 摄像头间歇断流（兜底）
- VMware 传输不稳定，长时间运行会偶发读帧失败/无新帧
- **对策**：主线程看门狗检测"3 秒无新帧"→ 请求重开摄像头（采集线程独占
  执行 release/open，避免并发 EBUSY）；重开后丢弃前 10 帧启动坏帧
- **注意**：不能用"palm 持续无效"判花屏——手不在画面时检测不到手是正常的，
  据此重开会反复误触（实测"什么都不做时画面反复卡住"）

### 双模型坐标映射偏移（手框定位不准）
- **根因**：palm 输出中心坐标的 pad 映射比例错误（192 输入空间直接减 pad，未换算到原图像素）
- **解决**：对照 PINTO 官方 app.py 精确映射：`cx=center_x*640`，`cy=center_y*640-pad*(640/192)`
- **验证**：手腕坐标从画面边缘 `(78,430)` 修正到合理位置 `(176,383)`

### 握拳不触发锁定（v1 时代记录，v2 已改为开掌手势）
- **根因**：v1 的 `detectFistHold` 中"手指骨骼长度比例检查"基于张手解剖比例，
  但握拳时各指尖全部收拢，比例必然失配 → 任何握拳都被判 false
- **解决**：双模型下脸部已被 palm 过滤，删除该过严检查，仅保留核心判据
- **v2 演进**：单指控制时"食指伸直+其余弯曲"与握拳在摄像头视角下仍难区分，
  故 v2 将锁定/解锁手势改为「五指张开」（各手指伸直比值判定），彻底消除误触

### 脸部/耳朵偶发显示手掌骨架
- **根因**：palm 偶发把脸部/耳朵判为手掌（得分 0.6~0.7），landmark 在脸部推理出低置信度关键点
- **解决**：双层过滤——① palm 阈值 0.6→0.7；② landmark 置信度 <0.5 直接判为无手（与 FSM 一致）
- **验证**：真手（置信度 >0.9）完全不受影响，脸部误检不再显示骨架

## 性能指标（VMware 环境实测）

| 指标 | 实测值 | 目标 |
| ---- | ---- | ---- |
| 单帧推理耗时 | 20-30ms（双模型串行，虚拟机 CPU） | — |
| 端到端延迟 | <50ms | ≤50ms |
| 摄像头帧率 | 30fps（YUYV 未压缩，稳定不卡死） | 30fps |
| 开掌锁定计时 | 精准（实测多次 1s 触发） | 无偏差 |
| 配置热加载 | 修改 JSON 后约 1s 生效 | 实时 |

> 备注：推理耗时含 palm + landmark 两次推理。真机（非虚拟机）预计可降至 5-10ms。

## RV1106 移植规划（下一迭代）

**核心思路**：仅替换推理子类，上层业务零修改。

| 项 | 说明 |
| ---- | ---- |
| 推理层 | 填充 `rknn_infer.cpp`：`rknn_init` 加载 .rknn 模型 |
| 模型转换 | ONNX → RKNN（RKNN-Toolkit2 原生支持，已提前打通） |
| 关键点输出 | 21 个手部关键点，与现有 `HandKeypoints` 结构一致 |
| 摄像头 | 替换为 RK 平台 MIPI/USB 采集（保持 `Camera` 接口不变） |
| uinput | 保持（Linux 标准接口） |
| 编译 | 交叉编译链（aarch64-linux-gnu） |

## 已实现的功能细节（v2 单指方案）

- [x] **双模型精确手掌裁剪**：palm_detection 定位手掌框 → 裁剪 → hand_landmark 关键点
- [x] **脸部/耳朵误检过滤**：双层防线（palm 阈值 0.7 + landmark 置信度 0.5）
- [x] **单指相对位移**：触控板式，手可小范围反复划动，鼠标覆盖整个屏幕（增益可调）
- [x] **移动直线修正**：掌心坐标 EMA 低通滤波 + 死区，抑制手部高频抖动，路径更直更稳
- [x] **捏合单击/双击 + 食指小指右键 + 双指V拖拽**：各动作完全分离（右键与拖拽判据正交）
- [x] **拖拽画面保持**：拖拽期间画面窗口保持显示（曾尝试销毁/重建窗口避免鼠标误触，
  但 GTK 销毁/重建在 VMware 下不稳定会引发摄像头卡死，已移除该方案）
- [x] **开掌锁定/解锁**：唯一保留的姿态手势（5 指全伸直比值判定），1s 防误触
- [x] **配置热加载**：修改 JSON 约 1 秒内实时生效（无需重启程序）
- [x] **静默运行模式**：`-q` 参数关闭心跳日志，正式使用画面干净
- [x] **窗口关闭检测**：点击右上角 × 优雅退出（防抖防误判）
- [x] **虚拟鼠标位置维护**：uinput 相对位移设备下的软件绝对定位实现

## 后续迭代规划

- [ ] **坐标漂移校准**：软件虚拟位置与真实指针脱节时，支持热键/手势一键归中
- [ ] **点击灵敏度自适应**：根据手与摄像头的距离动态调整下压速度阈值
- [ ] RV1106 嵌入式板卡移植（填充 `rknn_infer.cpp`，交叉编译）
