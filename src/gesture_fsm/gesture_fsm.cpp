// ============================================================================
// gesture_fsm/gesture_fsm.cpp
// 作用：GestureFSM 类实现（v2 单指控制方案核心模块）。
// 职责：
//   1. 加载 JSON 配置（所有阈值外置）
//   2. 对 21 关键点做卡尔曼平滑
//   3. 依据新状态机输出 v2 事件：锁定/解锁、绝对定位移动、单击、拖拽
//
// v2 点击/拖拽检测原理（基于食指尖 y 方向运动）：
//   图像坐标系 y 向下为正，食指尖"下点"时 y 增大、抬起时 y 减小。
//   - 下压速度 = 食指尖 y 的帧间变化率，超过 clickPressSpeedPx 判定为"按下"
//   - 按下后短时间内（clickPressTimeMs）回弹 → 单击
//   - 按下后持续按住 → 进入拖拽（左键保持），移动跟手，抬起结束
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
// 在 JSON 文本中查找 "key":数值 模式并返回数值
static bool jsonGetFloat(const std::string& json, const std::string& key, float& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    // 跳过空白
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
    // 1. 读取配置文件全部内容
    std::ifstream f(configPath);
    if (!f.good()) {
        std::fprintf(stderr, "[GestureFSM] 配置文件打开失败: %s\n", configPath.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    // 记录路径与 mtime（供热加载检测使用）
    m_configPath = configPath;
    struct stat st;
    if (::stat(configPath.c_str(), &st) == 0) {
        m_configMtime = static_cast<long>(st.st_mtime);
    }

    // 2. 解析各字段（key 名与 gesture_config.json 中一致）
    jsonGetInt  (json, "hold_time_ms",          m_cfg.lockHoldMs);
    jsonGetInt  (json, "state_cooldown_ms",     m_cfg.stateCooldownMs);
    jsonGetFloat(json, "process_noise",         m_cfg.kalmanProcessNoise);
    jsonGetFloat(json, "measure_noise",         m_cfg.kalmanMeasureNoise);
    jsonGetFloat(json, "extension_ratio",       m_cfg.openPalmRatio);
    jsonGetFloat(json, "press_speed_px",        m_cfg.clickPressSpeedPx);
    jsonGetInt  (json, "press_time_ms",         m_cfg.clickPressTimeMs);
    jsonGetInt  (json, "release_time_ms",       m_cfg.clickReleaseTimeMs);
    jsonGetFloat(json, "sensitivity",           m_cfg.mouseSensitivity);
    jsonGetInt  (json, "width",                 m_cfg.imageWidth);
    jsonGetInt  (json, "height",                m_cfg.imageHeight);
    // 屏幕分辨率在嵌套字段 screen.screen_width / screen.screen_height（轻量解析按 key 全局查找）
    jsonGetInt  (json, "screen_width",          m_cfg.screenWidth);
    jsonGetInt  (json, "screen_height",         m_cfg.screenHeight);

    // 3. 用配置初始化所有卡尔曼滤波器
    for (int i = 0; i < kHandKeypointCount; ++i) {
        m_filters[i].init(m_cfg.kalmanProcessNoise, m_cfg.kalmanMeasureNoise);
    }

    m_cfgLoaded = true;
    std::printf("[GestureFSM] 配置加载成功(v2): %s\n", configPath.c_str());
    std::printf("[GestureFSM] 锁定阈值=%dms 屏幕=%dx%d 下压速度=%.0fpx/s\n",
                m_cfg.lockHoldMs, m_cfg.screenWidth, m_cfg.screenHeight,
                m_cfg.clickPressSpeedPx);
    return true;
}

// --------------------------- 配置热加载 ---------------------------
bool GestureFSM::reloadIfChanged() {
    if (m_configPath.empty()) return false;

    struct stat st;
    if (::stat(m_configPath.c_str(), &st) != 0) {
        return false;  // 文件不可访问：不重载，保持现有配置
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
// 设计：v2 单指控制时手指常处于"食指伸直+其余弯曲"状态，与"握拳"在摄像头视角下
//       难以可靠区分（尤其手指朝屏幕时投影缩短）。改用「五指张开」作为锁定/解锁手势：
//       - 开掌 = 5 根手指全部伸直（各指尖到手腕距离 / 各指根 MCP 到手腕距离 > 阈值）
//       - 单指控制 = 仅食指伸直，其余弯曲 → 不满足"全部伸直" → 不会误触锁定/解锁
// 比值法对"手指朝屏幕"的透视投影等比缩短鲁棒（分子分母同缩，比值不变）。
bool GestureFSM::detectOpenPalm(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    const auto& w = kp.points[0];  // 手腕
    // 5 根手指的 [指尖, 指根 MCP] 索引对：
    //   拇指(2=MCP, 4=TIP)、食指(5,8)、中指(9,12)、无名指(13,16)、小指(17,20)
    const int tipBase[][2] = {
        {4, 2}, {8, 5}, {12, 9}, {16, 13}, {20, 17}
    };

    for (auto& tb : tipBase) {
        int tip = tb[0], base = tb[1];
        float tipToWrist = distance(kp.points[tip].x, kp.points[tip].y, w.x, w.y);
        float baseToWrist = distance(kp.points[base].x, kp.points[base].y, w.x, w.y);
        if (baseToWrist < 1e-6f) return false;  // 指根异常（退化保护）
        // 手指未伸直（弯曲）：指尖与指根接近，比值小 → 非开掌
        if (tipToWrist / baseToWrist < m_cfg.openPalmRatio) return false;
    }
    return true;  // 5 指全部伸直 = 开掌
}

// --------------------------- 状态机主循环 ---------------------------
GestureEvent GestureFSM::handleFrame(const HandKeypoints& kpRaw, double dtMs) {
    if (dtMs <= 0) dtMs = 1.0;  // 防御：避免除零

    // 未加载配置时使用默认参数（避免崩溃）
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
        m_fistHoldMs = 0;
        GestureEvent lostEvent = GestureEvent::kNone;
        if (m_dragging || m_pressActive) {
            m_dragging = false;
            m_pressActive = false;
            m_pressHoldMs = 0;
            lostEvent = GestureEvent::kDragEnd;
            std::printf("[FSM] 手丢失，拖拽强制结束\n");
        }
        m_lastIndexTipX = -1.f;
        m_lastIndexTipY = -1.f;
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
                m_fistHoldMs += static_cast<int>(dtMs);
                if (m_fistHoldMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    m_state = CtrlState::kLocked;
                    m_fistHoldMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;
                    event = GestureEvent::kOpenPalmHold;
                    std::printf("[FSM] 状态迁移：IDLE → LOCKED（已锁定）\n");
                }
            } else {
                // 衰减容忍：短暂变化（帧间抖动）只衰减 30ms 不清零
                m_fistHoldMs = (m_fistHoldMs > 30) ? (m_fistHoldMs - 30) : 0;
            }
            break;

        case CtrlState::kLocked:
            // LOCKED 态：开掌解锁 / 单指控制（定位/点击/拖拽）
            if (detectOpenPalm(kp)) {
                // ---- 开掌：累计长按解锁 ----
                m_fistHoldMs += static_cast<int>(dtMs);
                if (m_fistHoldMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    m_state = CtrlState::kIdle;
                    m_fistHoldMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;
                    // 若此前在按下/拖拽，先强制结束（防止左键卡死）
                    m_pressActive = false;
                    m_pressHoldMs = 0;
                    m_dragging = false;
                    event = GestureEvent::kOpenPalmHold;
                    std::printf("[FSM] 状态迁移：LOCKED → IDLE（已解锁）\n");
                }
                // 开掌期间不响应定位/点击（锁定/解锁手势生效中）
                break;
            }

            // ---- 非开掌（单指/其他状态）：单指控制 ----
            // 记录平滑后食指尖（点8）坐标供上层绝对定位
            m_smoothTipX = kp.points[8].x;
            m_smoothTipY = kp.points[8].y;

            // 食指尖 y 方向运动速度（图像 y 向下：下点为正）
            float velY = 0.f;
            if (m_lastIndexTipY >= 0) {
                velY = (m_smoothTipY - m_lastIndexTipY) * 1000.f / static_cast<float>(dtMs);
            }
            m_lastIndexTipX = m_smoothTipX;
            m_lastIndexTipY = m_smoothTipY;

            if (!m_pressActive && !m_dragging) {
                // ---- 空闲：检测下压（食指尖快速下点）----
                if (velY > m_cfg.clickPressSpeedPx) {
                    m_pressActive = true;
                    m_pressHoldMs = 0;
                    std::printf("[FSM] 按下开始（下压速度=%.0fpx/s）\n", velY);
                } else {
                    // 无操作：上报绝对定位移动
                    event = GestureEvent::kPointerMove;
                }
            } else if (m_dragging) {
                // ---- 拖拽中：检测抬起结束 ----
                if (velY < -m_cfg.clickPressSpeedPx * 0.5f) {
                    m_dragging = false;
                    m_pressActive = false;
                    m_pressHoldMs = 0;
                    event = GestureEvent::kDragEnd;
                    std::printf("[FSM] 拖拽结束（指尖抬起）\n");
                } else {
                    event = GestureEvent::kDragMove;  // 拖拽移动：跟手
                }
            } else {
                // ---- 按下判定中：区分单击与拖拽 ----
                m_pressHoldMs += static_cast<int>(dtMs);
                if (velY < -m_cfg.clickPressSpeedPx * 0.5f && m_pressHoldMs < m_cfg.clickPressTimeMs) {
                    // 快速回弹且未超时：单击
                    m_pressActive = false;
                    m_pressHoldMs = 0;
                    event = GestureEvent::kClick;
                    std::printf("[FSM] 单击\n");
                } else if (m_pressHoldMs >= m_cfg.clickPressTimeMs) {
                    // 按住超时：进入拖拽
                    m_dragging = true;
                    m_pressActive = false;
                    event = GestureEvent::kDragStart;
                    std::printf("[FSM] 拖拽开始（按住 %dms）\n", m_pressHoldMs);
                }
                // 其余情况：按下中尚未定性，保持等待（不发移动，避免点击时鼠标抖动）
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
