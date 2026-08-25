#pragma once
// ============================================================================
// gesture_fsm/gesture_fsm.h
// 作用：手势有限状态机（v2 单指控制方案核心模块）。
//
// v2 交互模型（方案A：单指绝对定位）：
//   IDLE       空闲态：唯一手势 = 握拳长按锁定
//     └─握拳持续≥lock_hold_ms──→ LOCKED
//   LOCKED     锁定运行态：
//     ├─ 握拳长按≥lock_hold_ms──→ IDLE（解锁）
//     ├─ 非握拳（食指伸出）──→ 每帧上报 kPointerMove（食指尖绝对定位鼠标）
//     ├─ 食指尖快速下压 ──→ 按下判定
//     │    ├─ 在 press_time_ms 内回弹 ──→ kClick（单击）
//     │    └─ 按住超过 press_time_ms ──→ kDragStart（拖拽开始）
//     ├─ 拖拽中移动 ──→ kDragMove（食指尖跟手移动）
//     └─ 拖拽中指尖抬起 ──→ kDragEnd（拖拽结束）
//
// 事件输出：
//   kFistHold     锁定/解锁开关（无 uinput 输出，仅状态切换）
//   kPointerMove  鼠标绝对定位移动（上层用食指尖坐标映射屏幕）
//   kClick        左键单击
//   kDragStart    拖拽开始（左键按住）
//   kDragMove     拖拽移动（左键保持 + 跟手）
//   kDragEnd      拖拽结束（左键释放）
//
// 判定阈值全部来自 config/gesture_config.json（click.press_speed_px 等），
// 代码中禁止硬编码魔法数字。
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
    kIdle   = 0,   // 空闲：等待握拳长按触发锁定
    kLocked = 1,   // 锁定运行：响应单指控制（定位/点击/拖拽）
};

// 手势事件枚举：状态机对外输出的识别结果（v2 单指方案）
enum class GestureEvent {
    kNone        = 0,   // 无动作
    kFistHold    = 1,   // 握拳长按：锁定/解锁开关（≥lock_hold_ms）
    kPointerMove = 2,   // 食指定位移动（LOCKED 态每帧上报，供绝对定位）
    kClick       = 3,   // 食指快速下点+回弹：左键单击
    kDragStart   = 4,   // 食指下点按住（超 press_time_ms）：拖拽开始
    kDragMove    = 5,   // 拖拽中移动（左键保持按下）
    kDragEnd     = 6,   // 指尖抬起：拖拽结束（释放左键）
};

// JSON 配置参数集合（运行时加载，所有阈值外置）
struct FsmConfig {
    // 锁定相关
    int   lockHoldMs = 1200;            // 握拳长按锁定/解锁时长阈值（毫秒）
    int   stateCooldownMs = 1500;       // 状态切换防抖冷却时间（毫秒）：锁定/解锁切换后
                                        // 此时间内不允许再次切换，防止脸部误判导致状态疯狂跳动

    // 卡尔曼滤波
    float kalmanProcessNoise = 1.0f;    // 过程噪声
    float kalmanMeasureNoise = 4.0f;    // 测量噪声

    // 握拳判定
    int   fistDistThreshold = 160;      // 握拳距离阈值（像素）：所有指尖到手腕距离均小于此值

    // 点击/拖拽判定（v2 核心）
    float clickPressSpeedPx = 60.f;     // 下压速度阈值（像素/秒）：食指尖 y 方向下落速度超此值判定"按下"
    int   clickPressTimeMs  = 250;      // 按下后回弹最大时长（毫秒）：在此时长内抬起 = 单击
    int   clickReleaseTimeMs = 250;     // 预留：回弹判定时长（当前与 press 共用）

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
    // @param configPath gesture_config.json 路径
    // @return true 加载成功，false 失败（内部输出中文错误日志）
    bool loadConfig(const std::string& configPath);

    // 配置热加载：检测配置文件 mtime 是否变化，变化则自动重新加载全部阈值
    // @return true 本次发生了重载（配置有变更），false 无变更
    bool reloadIfChanged();

    // 处理一帧关键点数据，驱动状态机推进
    // @param kp   当前帧的 21 个手部关键点（已由上层完成推理）
    // @param dtMs 与上一帧的时间间隔（毫秒），用于时长/速度类手势判定
    // @return 当前帧识别到的手势事件
    GestureEvent handleFrame(const HandKeypoints& kp, double dtMs);

    // 对 21 个关键点做卡尔曼平滑（在 handleFrame 内部调用，也可独立调用）
    void smoothKeypoints(const HandKeypoints& in, HandKeypoints& out);

    // 查询当前状态机所处状态（调试/日志打印用）
    const char* currentStateName() const;

    // 获取当前状态（供上层决策使用）
    CtrlState currentState() const { return m_state; }

    // 获取平滑后的食指尖（点8）坐标，供上层做鼠标绝对定位
    // @return 平滑后食指尖坐标；若未初始化返回 (-1,-1)
    void lastIndexTip(float& x, float& y) const;

    // 获取屏幕分辨率（绝对定位映射目标）
    int screenWidth() const  { return m_cfg.screenWidth; }
    int screenHeight() const { return m_cfg.screenHeight; }

    // 获取图像尺寸（供上层做坐标映射）
    int imageWidth() const  { return m_cfg.imageWidth; }
    int imageHeight() const { return m_cfg.imageHeight; }

private:
    // --------------------------- 状态成员 ---------------------------
    CtrlState m_state = CtrlState::kIdle;   // 当前状态
    int       m_fistHoldMs = 0;             // 握拳持续累计时长（毫秒）
    int       m_cooldownMs = 0;             // 状态切换防抖冷却计时器（毫秒）

    // 点击/拖拽检测状态（v2 核心）
    bool  m_pressActive = false;            // 是否处于"按下"判定中
    int   m_pressHoldMs = 0;                // 按下持续累计时长（毫秒）
    bool  m_dragging = false;               // 是否正在拖拽（左键按住中）
    float m_lastIndexTipX = -1.f;           // 上一帧食指尖 X（用于移动判定）
    float m_lastIndexTipY = -1.f;           // 上一帧食指尖 Y（用于下压速度判定）
    float m_smoothTipX = -1.f;              // 平滑后食指尖 X（供上层绝对定位）
    float m_smoothTipY = -1.f;              // 平滑后食指尖 Y

    // 卡尔曼滤波器数组：每个关键点一个
    KalmanFilter m_filters[kHandKeypointCount];

    // 配置参数
    FsmConfig m_cfg;
    bool      m_cfgLoaded = false;
    std::string m_configPath;               // 当前配置文件路径（热加载检测用）
    long      m_configMtime = 0;            // 配置文件的最后修改时间（mtime，热加载检测用）

    // --------------------------- 手势判定函数 ---------------------------
    // 握拳判定：所有指尖(4,8,12,16,20)到手腕(0)距离均 < fistDistThreshold
    bool detectFistHold(const HandKeypoints& kp);

    // 工具：计算两点欧式距离
    static float distance(float x1, float y1, float x2, float y2);
};

} // namespace hand_ctrl
