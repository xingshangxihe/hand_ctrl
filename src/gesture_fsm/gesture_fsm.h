#pragma once
// ============================================================================
// gesture_fsm/gesture_fsm.h
// 作用：手势有限状态机（项目核心模块）。
//
// 状态机设计：
//   IDLE       空闲态，等待握拳长按触发锁定
//     └─握拳持续≥lock_hold_time_ms──→ LOCK_WAIT
//   LOCK_WAIT   等待锁定倒计时（1.2s）
//     └─倒计时结束──→ LOCKED
//     └─中途松开──→ IDLE
//   LOCKED     锁定运行态，响应所有手势
//     └─握拳≥lock_hold_time_ms──→ IDLE（解锁）
//     └─任意手势触发──→ TRIGGER（瞬时态，输出事件后立即回 LOCKED）
//
// 6 种手势判定（对应 config/gesture_config.json 的 gesture 字段）：
//   1. fist_hold   握拳长按：锁定/解锁开关
//   2. open_palm   五指张开：跟随鼠标移动
//   3. fist_short  握拳短按：左键单击/拖拽
//   4. ok_gesture  OK 手势：右键单击
//   5. index_swipe 食指横向滑动：左右翻页
//   6. thumb_up    竖拇指：回车确认
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
    kIdle      = 0,   // 空闲：等待握拳长按触发锁定
    kLockWait  = 1,   // 等待锁定倒计时
    kLocked    = 2,   // 锁定运行：响应所有手势
    kTrigger   = 3,   // 手势触发（瞬时态，输出事件后回 LOCKED）
};

// 手势事件枚举：状态机对外输出的识别结果（与需求文档手势规则总表一一对应）
enum class GestureEvent {
    kNone          = 0,   // 未识别到任何手势
    kFistHold      = 1,   // 握拳长按：锁定/解锁开关（≥1.2s）
    kOpenPalm      = 2,   // 五指张开：跟随鼠标移动
    kFistShort     = 3,   // 握拳短按（0~0.5s 松开）：左键单击
    kOkGesture     = 4,   // OK手势：右键单击
    kIndexSwipe    = 5,   // 食指横向滑动：左右翻页
    kThumbUp       = 6,   // 竖拇指：回车确认
    kFistDragStart = 7,   // 握拳拖拽开始（按住 >0.5s）：按住左键
    kFistDragMove  = 8,   // 握拳拖拽中：每帧跟随移动鼠标
    kFistDragEnd   = 9,   // 握拳拖拽结束（松开）：释放左键
};

// JSON 配置参数集合（运行时加载，所有阈值外置）
struct FsmConfig {
    // 锁定相关
    int   lockHoldMs = 1200;            // 握拳长按锁定时长阈值（毫秒）
    int   stateCooldownMs = 1500;       // 状态切换防抖冷却时间（毫秒）：锁定/解锁切换后
                                        // 此时间内不允许再次切换，防止脸部误判导致状态疯狂跳动

    // 卡尔曼滤波
    float kalmanProcessNoise = 1.0f;    // 过程噪声
    float kalmanMeasureNoise = 4.0f;   // 测量噪声

    // 手势阈值（像素/比例）
    int   fistDistThreshold     = 60;  // 握拳距离阈值
    int   openPalmDistThreshold = 100; // 五指张开距离阈值
    int   fistShortMs           = 500; // 握拳短按判定时长
    int   okGestureGapPx         = 30; // OK 手势拇指食指间距
    int   swipeWristStablePx     = 15; // 滑动手腕静止阈值
    float swipeHorizontalRatio  = 0.3f;// 滑动横向位移比例（相对图像宽度）
    float thumbExtendedRatio    = 1.2f;// 拇指伸直判定比例
    int   otherFingersFoldPx    = 80;  // 竖拇指时其余四指折叠阈值

    // 图像尺寸（用于比例计算）
    int   imageWidth  = 640;
    int   imageHeight = 480;

    // 拖拽模式解锁判定：拖拽中若累计移动距离 < 此阈值（手静止握拳），
    // 且握拳总时长 ≥ lockHoldMs，才触发解锁（避免拖拽移动被误判为解锁）
    int   dragUnlockStaticPx = 80;

    // 鼠标跟随灵敏度：掌心帧间位移（像素）× 此系数 = 鼠标相对位移
    float mouseSensitivity = 0.8f;
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
    // 说明：主循环每约 1s 调用一次即可实现"改配置实时生效"（阶段7 验收项）。
    // @return true 本次发生了重载（配置有变更），false 无变更
    bool reloadIfChanged();

    // 处理一帧关键点数据，驱动状态机推进
    // @param kp   当前帧的 21 个手部关键点（已由上层完成推理）
    // @param dtMs 与上一帧的时间间隔（毫秒），用于时长类手势判定
    // @return 当前帧识别到的手势事件
    GestureEvent handleFrame(const HandKeypoints& kp, double dtMs);

    // 对 21 个关键点做卡尔曼平滑（在 handleFrame 内部调用，也可独立调用）
    // @param in  原始关键点
    // @param out 平滑后的关键点
    void smoothKeypoints(const HandKeypoints& in, HandKeypoints& out);

    // 查询当前状态机所处状态（调试/日志打印用）
    const char* currentStateName() const;

    // 获取当前状态（供上层决策使用）
    CtrlState currentState() const { return m_state; }

    // 获取锁定倒计时剩余毫秒（LOCK_WAIT 状态下有意义）
    int lockCountdownMs() const { return m_lockCountdownMs; }

    // 获取鼠标跟随灵敏度系数（供上层将掌心位移映射为鼠标位移）
    float mouseSensitivity() const { return m_cfg.mouseSensitivity; }

    // 获取图像尺寸（供上层做比例计算）
    int imageWidth() const  { return m_cfg.imageWidth; }
    int imageHeight() const { return m_cfg.imageHeight; }

private:
    // --------------------------- 状态成员 ---------------------------
    CtrlState m_state = CtrlState::kIdle;   // 当前状态
    int       m_lockCountdownMs = 0;        // 锁定倒计时剩余（LOCK_WAIT 态）
    int       m_fistHoldMs = 0;             // 握拳持续累计时长（毫秒）

    // 食指滑动检测：记录上一帧食指尖位置，用于计算横向位移
    float     m_lastIndexTipX = -1.f;
    float     m_lastIndexTipY = -1.f;
    float     m_lastWristX = -1.f;
    float     m_lastWristY = -1.f;

    // 状态切换防抖冷却计时器（毫秒）：状态迁移后置为 cooldown 值，每帧递减
    int m_cooldownMs = 0;

    // 握拳拖拽状态：false=未拖拽，true=正在拖拽（左键按住中）
    bool m_dragging = false;
    // 拖拽期间累计移动距离（像素）：用于"手静止握拳长按"解锁判定
    float m_dragMoveAccum = 0.f;
    // 拖拽期间上一帧掌心（点9）坐标：用于累计位移
    float m_lastDragPalmX = -1.f;
    float m_lastDragPalmY = -1.f;

    // 卡尔曼滤波器数组：每个关键点一个
    KalmanFilter m_filters[kHandKeypointCount];

    // 配置参数
    FsmConfig m_cfg;
    bool      m_cfgLoaded = false;
    std::string m_configPath;   // 当前配置文件路径（热加载检测用）
    long      m_configMtime = 0; // 配置文件的最后修改时间（mtime，热加载检测用）

    // --------------------------- 手势判定函数 ---------------------------
    // 各手势判定返回 true 表示当前帧满足该手势条件
    bool detectFistHold(const HandKeypoints& kp);
    bool detectOpenPalm(const HandKeypoints& kp);
    bool detectOkGesture(const HandKeypoints& kp);
    bool detectIndexSwipe(const HandKeypoints& kp, double dtMs);
    bool detectThumbUp(const HandKeypoints& kp);

    // 工具：计算两点欧式距离
    static float distance(float x1, float y1, float x2, float y2);
};

} // namespace hand_ctrl