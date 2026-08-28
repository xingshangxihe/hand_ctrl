#pragma once
// ============================================================================
// gesture_fsm/gesture_fsm.h
// 作用：手势有限状态机（v2.2 食指弯曲控制方案核心模块）。
//
// v2.2 交互模型（点击/移动完全正交，杜绝误判）：
//   IDLE       空闲态：唯一手势 = 开掌长按锁定
//     └─开掌持续≥lock_hold_ms──→ LOCKED
//   LOCKED     锁定运行态：
//     ├─ 开掌长按≥lock_hold_ms──→ IDLE（解锁）
//     ├─ 食指伸直（未弯曲）──→ 每帧 kPointerMove（食指尖绝对定位鼠标）
//     ├─ 快速弯曲→伸直 ──→ kClick（左键单击）
//     ├─ 弯曲保持 ≥bend_hold_ms ──→ kDragStart（拖拽开始，左键按住）
//     ├─ 拖拽中移动 ──→ kDragMove（左键保持 + 跟手）
//     └─ 伸直 ──→ kDragEnd（拖拽结束，释放左键）
//
// 为何用"食指弯曲"而不是"捏合"（v2.1 方案的问题）：
//   1. 捏合（拇指+食指闭合）动作不自然，长时间操作累。
//   2. 食指弯曲判定用"食指尖(8)到中指根(9)的距离"——伸直时距离大，
//      弯曲点击时食指尖折向中指根、距离骤减。与手平移完全正交：
//      平移时两指相对位置不变，移动永不误触发点击。
//   3. 弯曲动作类似按鼠标键，直觉自然，学习成本最低。
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

// 手势事件枚举：状态机对外输出的识别结果（v2.2 食指弯曲方案）
enum class GestureEvent {
    kNone         = 0,   // 无动作
    kOpenPalmHold = 1,   // 开掌长按：锁定/解锁开关（≥lock_hold_ms）
    kPointerMove  = 2,   // 食指尖定位移动（LOCKED 态食指伸直时每帧上报）
    kClick        = 3,   // 快速弯曲→伸直：左键单击
    kDragStart    = 4,   // 弯曲保持（超 bend_hold_ms）：拖拽开始
    kDragMove     = 5,   // 拖拽中移动（左键保持按下）
    kDragEnd      = 6,   // 伸直：拖拽结束（释放左键）
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

    // 食指弯曲判定（点击/拖拽动作，v2.2 核心）
    float bendDistThresholdPx = 50.f;   // 弯曲距离阈值（像素）：食指尖(8)到中指根(9)距离 < 此值 = 食指弯曲
    int   bendHoldMs = 350;             // 弯曲保持时长阈值（毫秒）：保持超过 = 拖拽开始；之前伸直 = 单击

    // 鼠标绝对定位映射缩放（各方向独立，解决覆盖范围不足）
    float mapScaleX = 1.0f;             // 水平缩放系数：屏幕X = 指尖X/画面宽×屏幕宽×系数
    float mapScaleY = 1.0f;             // 垂直缩放系数：屏幕Y = 指尖Y/画面高×屏幕高×系数

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

    // 获取平滑后的食指尖（点8）坐标，供上层做鼠标绝对定位
    void lastIndexTip(float& x, float& y) const;

    // 获取屏幕分辨率（绝对定位映射目标）
    int screenWidth() const  { return m_cfg.screenWidth; }
    int screenHeight() const { return m_cfg.screenHeight; }

    // 获取图像尺寸（供上层做坐标映射）
    int imageWidth() const  { return m_cfg.imageWidth; }
    int imageHeight() const { return m_cfg.imageHeight; }

    // 获取鼠标绝对定位映射缩放系数（各方向独立）
    float mapScaleX() const { return m_cfg.mapScaleX; }
    float mapScaleY() const { return m_cfg.mapScaleY; }

private:
    // --------------------------- 状态成员 ---------------------------
    CtrlState m_state = CtrlState::kIdle;   // 当前状态
    int       m_holdMs = 0;                 // 开掌（锁定手势）持续累计时长（毫秒）
    int       m_cooldownMs = 0;             // 状态切换防抖冷却计时器（毫秒）

    // 食指弯曲/拖拽检测状态（v2.2 核心）
    bool  m_bending = false;                // 是否处于弯曲中（等待区分单击/拖拽）
    int   m_bendHoldMs = 0;                 // 弯曲持续累计时长（毫秒）
    bool  m_dragging = false;               // 是否正在拖拽（左键按住中）
    float m_smoothTipX = -1.f;              // 平滑后食指尖 X（供上层绝对定位）
    float m_smoothTipY = -1.f;              // 平滑后食指尖 Y

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

    // 工具：计算两点欧式距离
    static float distance(float x1, float y1, float x2, float y2);
};

} // namespace hand_ctrl
