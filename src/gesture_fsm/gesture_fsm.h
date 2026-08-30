#pragma once
// ============================================================================
// gesture_fsm/gesture_fsm.h
// 作用：手势有限状态机（v2.5 捏合单击 + 双指V拖拽方案核心模块）。
//
// v2.5 交互模型（单击/拖拽/移动三个动作完全分离）：
//   IDLE       空闲态：唯一手势 = 开掌长按锁定
//     └─开掌持续≥lock_hold_ms──→ LOCKED
//   LOCKED     锁定运行态：
//     ├─ 开掌长按≥lock_hold_ms──→ IDLE（解锁）
//     ├─ 食指伸直（非捏合/非双指V）──→ 每帧 kPointerMove（掌心相对位移移动鼠标）
//     ├─ 拇指+食指捏合（tap）──→ kClick（左键单击）
//     ├─ 双指V手势（食指+中指伸直、无名指+小指弯曲）──→ kDragStart（拖拽开始，左键按住）
//     ├─ 双指V保持中移动 ──→ kDragMove（左键保持 + 跟手）
//     └─ 收手解除双指V ──→ kDragEnd（拖拽结束，释放左键）
//
// 设计理由（v2.4 握拳/以往方案的问题）：
//   1. 握拳：远距离时伸食指会被误判为握拳（各指尖投影缩短），不可靠。
//   2. 单击与拖拽必须用两个完全不同的动作，避免"时长区分"的模糊地带。
//   3. 单击用「捏合」：拇指尖(4)与食指尖(8)距离判定，独立可靠；瞬时 tap 不累。
//   4. 拖拽用「双指V」：食指+中指伸直、无名指+小指弯曲的显式手型，与移动
//      （单指/张手）和单击（捏合）完全不同，绝无冲突。
//
// 事件输出：
//   kOpenPalmHold 开掌长按锁定/解锁开关（无 uinput 输出，仅状态切换）
//   kPointerMove  食指尖绝对定位移动（上层映射屏幕坐标）
//   kClick        左键单击
//   kDragStart    拖拽开始（左键按住）
//   kDragMove     拖拽移动（左键保持 + 跟手）
//   kDragEnd      拖拽结束（左键释放）
//
// 全部阈值来自 config/gesture_config.json，代码中禁止硬编码魔法数字。
//
// MediaPipe Hands 21 关键点索引参考：
//   0: WRIST（手腕）
//   1-4:  THUMB（拇指：CMC, MCP, IP, TIP）
//   5-8:  INDEX_FINGER（食指：MCP, PIP, DIP, TIP）
//   9-12: MIDDLE_FINGER（中指）
//   13-16: RING_FINGER（无名指）
//   17-20: PINKY（小指）
// ============================================================================

#include <string>
#include "utils/common.h"
#include "gesture_fsm/kalman_filter.h"

namespace hand_ctrl {

// 状态枚举（强类型，避免魔法数字）
enum class CtrlState {
    kIdle   = 0,   // 空闲：等待开掌长按触发锁定
    kLocked = 1,   // 锁定运行：响应单指控制（定位/点击/拖拽）
};

// 手势事件枚举：状态机对外输出的识别结果（v2.5 捏合+双指V方案）
enum class GestureEvent {
    kNone         = 0,   // 无动作
    kOpenPalmHold = 1,   // 开掌长按：锁定/解锁开关（≥lock_hold_ms）
    kPointerMove  = 2,   // 掌心定位移动（LOCKED 态非捏合/非双指V时每帧上报）
    kClick        = 3,   // 拇指+食指捏合 tap：左键单击
    kDragStart    = 4,   // 双指V手势出现：拖拽开始（左键按住）
    kDragMove     = 5,   // 双指V保持中移动：拖拽移动（左键保持按下）
    kDragEnd      = 6,   // 解除双指V：拖拽结束（释放左键）
};

// JSON 配置参数集合（运行时加载，所有阈值外置）
struct FsmConfig {
    // 锁定相关
    int   lockHoldMs = 1200;            // 开掌长按锁定/解锁时长阈值（毫秒）
    int   stateCooldownMs = 1500;       // 状态切换防抖冷却时间（毫秒）：锁定/解锁切换后
                                        // 此时间内不允许再次切换，防止误判导致状态疯狂跳动

    // 卡尔曼滤波
    float kalmanProcessNoise = 1.0f;    // 过程噪声
    float kalmanMeasureNoise = 4.0f;    // 测量噪声

    // 开掌判定（锁定/解锁手势）
    float openPalmRatio = 1.8f;         // 手指伸直比值阈值：每指指尖到手腕/该指根(MCP)到手腕 > 此值
                                        // 视为该手指伸直。5 指全部伸直 = 开掌。
                                        // 拇指阈值自动放宽（×0.8），因其根紧邻手腕比值天然小

    // 单击判定（捏合，v2.5.1）
    float pinchRatio = 0.55f;           // 捏合比例阈值：拇指尖(4)到食指尖(8)距离 ÷ 中指根(9)到手腕(0)
                                        // < 此值 = 捏合 → 单击。比例法对"手离摄像头远近"鲁棒

    // 拖拽判定（双指V，v2.5）
    int   dragFoldDistPx = 130;         // 双指V：无名指(16)/小指(20)到手腕距离 < 此值（弯曲）
                                        // 食指(8)/中指(12)伸直用 openPalmRatio 比值判定

    // 鼠标相对位移增益（触控板式，各方向独立）
    float mouseGainX = 2.0f;            // 水平增益：鼠标位移 = 指尖画面位移×(屏幕宽/画面宽)×gain
    float mouseGainY = 2.0f;            // 垂直增益：同上，独立调大可解决某方向覆盖不足

    // 图像尺寸（用于坐标映射比例计算）
    int   imageWidth  = 640;
    int   imageHeight = 480;

    // 目标屏幕分辨率（绝对定位映射目标）
    int   screenWidth  = 1920;
    int   screenHeight = 1080;
};

class GestureFSM {
public:
    GestureFSM();
    ~GestureFSM();

    // 加载 JSON 配置文件
    bool loadConfig(const std::string& configPath);

    // 配置热加载：检测配置文件 mtime 变化，变化则自动重新加载
    bool reloadIfChanged();

    // 处理一帧关键点数据，驱动状态机推进
    GestureEvent handleFrame(const HandKeypoints& kp, double dtMs);

    // 对 21 个关键点做卡尔曼平滑（handleFrame 内部调用）
    void smoothKeypoints(const HandKeypoints& in, HandKeypoints& out);

    // 查询当前状态机所处状态（调试/日志打印用）
    const char* currentStateName() const;

    // 获取当前状态（供上层决策使用）
    CtrlState currentState() const { return m_state; }

    // 获取平滑后的掌心（点9，中指根）坐标，供上层做鼠标相对位移
    // 说明：用掌心而非食指尖作为参考点——握拳拖拽时食指尖收拢不可用，
    //       掌心（中指根）在手部中央、始终可见且稳定。
    void lastPalmPoint(float& x, float& y) const;

    // 获取屏幕分辨率（绝对定位映射目标）
    int screenWidth() const  { return m_cfg.screenWidth; }
    int screenHeight() const { return m_cfg.screenHeight; }

    // 获取图像尺寸（供上层做坐标映射）
    int imageWidth() const  { return m_cfg.imageWidth; }
    int imageHeight() const { return m_cfg.imageHeight; }

    // 获取鼠标相对位移增益系数（各方向独立）
    float mouseGainX() const { return m_cfg.mouseGainX; }
    float mouseGainY() const { return m_cfg.mouseGainY; }

private:
    // --------------------------- 状态成员 ---------------------------
    CtrlState m_state = CtrlState::kIdle;   // 当前状态
    int       m_holdMs = 0;                 // 开掌（锁定手势）持续累计时长（毫秒）
    int       m_cooldownMs = 0;             // 状态切换防抖冷却计时器（毫秒）

    // 捏合/拖拽检测状态（v2.5 核心）
    bool  m_pinched = false;                // 捏合防连发：记录上一次是否捏合（边沿触发单击）
    bool  m_dragging = false;               // 是否正在拖拽（双指V保持，左键按住中）
    float m_smoothPalmX = -1.f;             // 平滑后掌心（点9，中指根）X：移动/拖拽参考点
    float m_smoothPalmY = -1.f;             // 平滑后掌心 Y

    // 卡尔曼滤波器数组：每个关键点一个
    KalmanFilter m_filters[kHandKeypointCount];

    // 配置参数
    FsmConfig m_cfg;
    bool      m_cfgLoaded = false;
    std::string m_configPath;               // 当前配置文件路径（热加载检测用）
    long      m_configMtime = 0;            // 配置文件 mtime（热加载检测用）

    // --------------------------- 手势判定函数 ---------------------------
    // 开掌判定：5 根手指全部伸直（指尖到手腕 / 指根到手腕 > openPalmRatio，拇指 ×0.8）
    bool detectOpenPalm(const HandKeypoints& kp);

    // 双指V判定（拖拽手势）：食指(8)+中指(12)伸直（比值 > openPalmRatio），
    // 无名指(16)+小指(20)弯曲（到手腕距离 < dragFoldDistPx）
    bool detectTwoFinger(const HandKeypoints& kp);

    // 工具：计算两点欧式距离
    static float distance(float x1, float y1, float x2, float y2);
};

} // namespace hand_ctrl
