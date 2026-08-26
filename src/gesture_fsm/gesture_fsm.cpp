// ============================================================================
// gesture_fsm/gesture_fsm.cpp
// 作用：GestureFSM 类实现（v2.1 捏合控制方案核心模块）。
// 职责：
//   1. 加载 JSON 配置（所有阈值外置）
//   2. 对 21 关键点做卡尔曼平滑
//   3. 依据 v2.1 状态机输出事件：开掌锁定/解锁、捏合单击/拖拽、食指尖定位
//
// v2.1 捏合检测原理：
//   点击动作 = "拇指+食指捏合"，与移动（手平移）完全正交：
//   - 捏合距离 = 拇指尖(4)与食指尖(8)的欧式距离
//   - 距离 < pinchDistThresholdPx → 捏合（按下）
//   - 捏合保持超过 pinchHoldMs → 拖拽；在保持期内松开 → 单击
//   - 平移手时捏合状态不变，移动永不误触发点击
// ============================================================================

#include "gesture_fsm/gesture_fsm.h"

#include <cstdio>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>   // stat：配置热加载需要读取文件 mtime
// 简易 JSON 解析（轻量依赖，避免引入 nlohmann/json）
#include <sstream>

namespace hand_ctrl {

// --------------------------- 简易 JSON 数值解析工具 ---------------------------
static bool jsonGetFloat(const std::string& json, const std::string& key, float& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) ++pos;
    try { out = std::stof(json.substr(pos)); } catch (...) { return false; }
    return true;
}
static bool jsonGetInt(const std::string& json, const std::string& key, int& out) {
    float f = 0.f;
    if (!jsonGetFloat(json, key, f)) return false;
    out = static_cast<int>(f);
    return true;
}

// --------------------------- 构造 / 析构 ---------------------------
GestureFSM::GestureFSM() = default;
GestureFSM::~GestureFSM() = default;

// --------------------------- 配置加载 ---------------------------
bool GestureFSM::loadConfig(const std::string& configPath) {
    std::ifstream f(configPath);
    if (!f.good()) {
        std::fprintf(stderr, "[GestureFSM] 配置文件打开失败: %s\n", configPath.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    m_configPath = configPath;
    struct stat st;
    if (::stat(configPath.c_str(), &st) == 0) {
        m_configMtime = static_cast<long>(st.st_mtime);
    }

    // 解析各字段（key 名与 gesture_config.json 一致）
    jsonGetInt  (json, "hold_time_ms",          m_cfg.lockHoldMs);
    jsonGetInt  (json, "state_cooldown_ms",     m_cfg.stateCooldownMs);
    jsonGetFloat(json, "process_noise",         m_cfg.kalmanProcessNoise);
    jsonGetFloat(json, "measure_noise",         m_cfg.kalmanMeasureNoise);
    jsonGetFloat(json, "extension_ratio",              m_cfg.openPalmRatio);
    jsonGetFloat(json, "pinch_distance_threshold_px", m_cfg.pinchDistThresholdPx);
    jsonGetInt  (json, "pinch_hold_ms",               m_cfg.pinchHoldMs);
    jsonGetFloat(json, "sensitivity",                 m_cfg.mouseSensitivity);
    jsonGetInt  (json, "width",                       m_cfg.imageWidth);
    jsonGetInt  (json, "height",                      m_cfg.imageHeight);
    jsonGetInt  (json, "screen_width",                m_cfg.screenWidth);
    jsonGetInt  (json, "screen_height",               m_cfg.screenHeight);

    // 用配置初始化卡尔曼滤波器
    for (int i = 0; i < kHandKeypointCount; ++i) {
        m_filters[i].init(m_cfg.kalmanProcessNoise, m_cfg.kalmanMeasureNoise);
    }

    m_cfgLoaded = true;
    std::printf("[GestureFSM] 配置加载成功(v2.1): %s\n", configPath.c_str());
    std::printf("[GestureFSM] 锁定阈值=%dms 屏幕=%dx%d 捏合距离=%.0fpx 灵敏度=%.2f\n",
                m_cfg.lockHoldMs, m_cfg.screenWidth, m_cfg.screenHeight,
                m_cfg.pinchDistThresholdPx, m_cfg.mouseSensitivity);
    return true;
}

// --------------------------- 配置热加载 ---------------------------
bool GestureFSM::reloadIfChanged() {
    if (m_configPath.empty()) return false;

    struct stat st;
    if (::stat(m_configPath.c_str(), &st) != 0) {
        return false;
    }
    long curMtime = static_cast<long>(st.st_mtime);
    if (curMtime == m_configMtime) {
        return false;
    }

    std::printf("[GestureFSM] 检测到配置变更，正在热加载...\n");
    bool ok = loadConfig(m_configPath);
    if (ok) {
        std::printf("[GestureFSM] 配置热加载成功（阈值已实时生效）\n");
    } else {
        std::fprintf(stderr, "[GestureFSM] 配置热加载失败，沿用上次有效配置\n");
    }
    return ok;
}

// --------------------------- 卡尔曼平滑 ---------------------------
void GestureFSM::smoothKeypoints(const HandKeypoints& in, HandKeypoints& out) {
    out = in;
    if (in.points.size() < kHandKeypointCount) {
        out.points = in.points;
        return;
    }
    out.points.resize(kHandKeypointCount);
    for (int i = 0; i < kHandKeypointCount; ++i) {
        float sx, sy;
        m_filters[i].update(in.points[i].x, in.points[i].y, sx, sy);
        out.points[i].x = sx;
        out.points[i].y = sy;
        out.points[i].score = in.points[i].score;
    }
}

// --------------------------- 工具函数 ---------------------------
float GestureFSM::distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// --------------------------- 开掌判定（锁定/解锁手势） ---------------------------
// 5 指全部伸直（指尖/指根比值 > 阈值），拇指阈值放宽×0.8（其根紧邻手腕）
bool GestureFSM::detectOpenPalm(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    const auto& w = kp.points[0];  // 手腕
    // [指尖, 指根 MCP]：拇指(4,2)、食指(8,5)、中指(12,9)、无名指(16,13)、小指(20,17)
    const int tipBase[][2] = {
        {4, 2}, {8, 5}, {12, 9}, {16, 13}, {20, 17}
    };

    for (int i = 0; i < 5; ++i) {
        int tip = tipBase[i][0], base = tipBase[i][1];
        float tipToWrist = distance(kp.points[tip].x, kp.points[tip].y, w.x, w.y);
        float baseToWrist = distance(kp.points[base].x, kp.points[base].y, w.x, w.y);
        if (baseToWrist < 1e-6f) return false;
        float ratio = tipToWrist / baseToWrist;
        float threshold = (i == 0) ? (m_cfg.openPalmRatio * 0.8f) : m_cfg.openPalmRatio;
        if (ratio < threshold) return false;
    }
    return true;  // 5 指全部伸直 = 开掌
}

// --------------------------- 状态机主循环 ---------------------------
GestureEvent GestureFSM::handleFrame(const HandKeypoints& kpRaw, double dtMs) {
    if (dtMs <= 0) dtMs = 1.0;

    if (!m_cfgLoaded) {
        for (int i = 0; i < kHandKeypointCount; ++i) {
            m_filters[i].init(m_cfg.kalmanProcessNoise, m_cfg.kalmanMeasureNoise);
        }
        m_cfgLoaded = true;
    }

    // 1. 关键点卡尔曼平滑
    HandKeypoints kp;
    smoothKeypoints(kpRaw, kp);

    // 2. 手不存在或置信度过低：重置全部状态，若拖拽中先强制结束
    bool lowConfidence = (kpRaw.confidence < 0.5f);
    if (!kp.valid || kp.points.size() < kHandKeypointCount || lowConfidence) {
        m_holdMs = 0;
        GestureEvent lostEvent = GestureEvent::kNone;
        if (m_dragging || m_pinching) {
            m_dragging = false;
            m_pinching = false;
            m_pinchHoldMs = 0;
            lostEvent = GestureEvent::kDragEnd;
            std::printf("[FSM] 手丢失，拖拽强制结束\n");
        }
        if (m_state == CtrlState::kLocked) {
            m_state = CtrlState::kIdle;  // 手丢失回到空闲，避免误操作
        }
        return lostEvent;
    }

    // 3. 冷却计时递减（每帧执行，防抖）
    if (m_cooldownMs > 0) {
        m_cooldownMs -= static_cast<int>(dtMs);
        if (m_cooldownMs < 0) m_cooldownMs = 0;
    }

    GestureEvent event = GestureEvent::kNone;
    switch (m_state) {
        case CtrlState::kIdle:
            // IDLE 态：开掌长按触发锁定
            if (detectOpenPalm(kp)) {
                m_holdMs += static_cast<int>(dtMs);
                if (m_holdMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    m_state = CtrlState::kLocked;
                    m_holdMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;
                    event = GestureEvent::kOpenPalmHold;
                    std::printf("[FSM] 状态迁移：IDLE → LOCKED（已锁定）\n");
                }
            } else {
                m_holdMs = (m_holdMs > 30) ? (m_holdMs - 30) : 0;
            }
            break;

        case CtrlState::kLocked:
            // LOCKED 态：开掌解锁 / 捏合控制（定位/单击/拖拽）
            if (detectOpenPalm(kp)) {
                // ---- 开掌：累计长按解锁 ----
                m_holdMs += static_cast<int>(dtMs);
                if (m_holdMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    m_state = CtrlState::kIdle;
                    m_holdMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;
                    m_pinching = false;
                    m_pinchHoldMs = 0;
                    m_dragging = false;
                    event = GestureEvent::kOpenPalmHold;
                    std::printf("[FSM] 状态迁移：LOCKED → IDLE（已解锁）\n");
                }
                break;  // 开掌期间不响应定位/点击
            }

            // ---- 非开掌：捏合控制 ----
            // 记录平滑后食指尖（点8）坐标供上层绝对定位
            m_smoothTipX = kp.points[8].x;
            m_smoothTipY = kp.points[8].y;

            // 捏合距离 = 拇指尖(4)与食指尖(8)的欧式距离
            float pinchDist = distance(kp.points[4].x, kp.points[4].y,
                                       kp.points[8].x, kp.points[8].y);
            bool pinched = (pinchDist < m_cfg.pinchDistThresholdPx);

            if (!m_pinching && !m_dragging) {
                // ---- 空闲：检测捏合（点击/拖拽的开始动作）----
                if (pinched) {
                    m_pinching = true;
                    m_pinchHoldMs = 0;
                    std::printf("[FSM] 捏合开始（距离=%.0fpx）\n", pinchDist);
                } else {
                    event = GestureEvent::kPointerMove;  // 未捏合：绝对定位移动
                }
            } else if (m_dragging) {
                // ---- 拖拽中：检测松开结束 ----
                if (!pinched) {
                    m_dragging = false;
                    m_pinching = false;
                    m_pinchHoldMs = 0;
                    event = GestureEvent::kDragEnd;
                    std::printf("[FSM] 拖拽结束（松开捏合）\n");
                } else {
                    event = GestureEvent::kDragMove;  // 拖拽移动：跟手
                }
            } else {
                // ---- 捏合判定中：区分单击与拖拽 ----
                m_pinchHoldMs += static_cast<int>(dtMs);
                if (!pinched) {
                    // 松开且未达拖拽阈值：单击
                    m_pinching = false;
                    m_pinchHoldMs = 0;
                    event = GestureEvent::kClick;
                    std::printf("[FSM] 单击（捏合 %dms）\n", m_pinchHoldMs);
                } else if (m_pinchHoldMs >= m_cfg.pinchHoldMs) {
                    // 捏合保持超时：进入拖拽
                    m_dragging = true;
                    m_pinching = false;
                    event = GestureEvent::kDragStart;
                    std::printf("[FSM] 拖拽开始（捏合 %dms）\n", m_pinchHoldMs);
                }
                // 其余情况：捏合中尚未定性，等待（不发移动，避免误动）
            }
            break;
    }

    return event;
}

// --------------------------- 状态查询 ---------------------------
const char* GestureFSM::currentStateName() const {
    switch (m_state) {
        case CtrlState::kIdle:   return "IDLE";
        case CtrlState::kLocked: return "LOCKED";
    }
    return "UNKNOWN";
}

void GestureFSM::lastIndexTip(float& x, float& y) const {
    x = m_smoothTipX;
    y = m_smoothTipY;
}

} // namespace hand_ctrl
