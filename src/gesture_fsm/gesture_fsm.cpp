// ============================================================================
// gesture_fsm/gesture_fsm.cpp
// 作用：GestureFSM 类实现（v2.3 食指弯曲控制方案核心模块）。
// 职责：
//   1. 加载 JSON 配置（所有阈值外置）
//   2. 对 21 关键点做卡尔曼平滑
//   3. 依据 v2.3 状态机输出事件：开掌锁定/解锁、食指弯曲单击/拖拽、食指尖定位
//
// v2.3 食指弯曲检测原理：
//   点击动作 = "食指弯曲"（食指尖折向掌心），与移动（手平移）完全正交：
//   - 弯曲比值 = 食指尖(8)到食指根(5) 距离 / 食指根(5)到手腕(0) 距离
//   - 比值 < bendRatio → 弯曲（按下）
//   - 弯曲保持超过 bendHoldMs → 拖拽；在保持期内伸直 → 单击
//   - 手指并拢（其他手指靠近食指）时食指仍伸直、比值不变 → 不会误判
//   - 平移手时食指形态不变，移动永不误触发点击
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
    jsonGetFloat(json, "pinch_ratio",                  m_cfg.pinchRatio);
    jsonGetInt  (json, "fold_distance_threshold_px",   m_cfg.dragFoldDistPx);
    jsonGetFloat(json, "gain_x",                       m_cfg.mouseGainX);
    jsonGetFloat(json, "gain_y",                       m_cfg.mouseGainY);
    jsonGetInt  (json, "width",                       m_cfg.imageWidth);
    jsonGetInt  (json, "height",                      m_cfg.imageHeight);
    jsonGetInt  (json, "screen_width",                m_cfg.screenWidth);
    jsonGetInt  (json, "screen_height",               m_cfg.screenHeight);

    // 用配置初始化卡尔曼滤波器
    for (int i = 0; i < kHandKeypointCount; ++i) {
        m_filters[i].init(m_cfg.kalmanProcessNoise, m_cfg.kalmanMeasureNoise);
    }

    m_cfgLoaded = true;
    std::printf("[GestureFSM] 配置加载成功(v2.5.1): %s\n", configPath.c_str());
    std::printf("[GestureFSM] 锁定阈值=%dms 屏幕=%dx%d 捏合比例=%.2f 双指V折叠=%dpx 增益=%.2fx%.2f\n",
                m_cfg.lockHoldMs, m_cfg.screenWidth, m_cfg.screenHeight,
                m_cfg.pinchRatio, m_cfg.dragFoldDistPx,
                m_cfg.mouseGainX, m_cfg.mouseGainY);
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

// --------------------------- 双指V判定（拖拽手势） ---------------------------
// 判定：食指(8)+中指(12)伸直（指尖到手腕/指根到手腕比值 > openPalmRatio），
//       无名指(16)+小指(20)弯曲（到手腕距离 < dragFoldDistPx）。
// 显式手型：与移动（单指/张手）、单击（捏合）完全不同，绝无冲突。
// 说明：摆出"V"字手势（两根手指比胜利）即进入拖拽，收手即释放。
bool GestureFSM::detectTwoFinger(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    const auto& w = kp.points[0];  // 手腕

    // 食指(8)/中指(12)必须伸直：指尖/指根 比值 > 阈值
    {
        float tip = distance(kp.points[8].x, kp.points[8].y, w.x, w.y);
        float root = distance(kp.points[5].x, kp.points[5].y, w.x, w.y);
        if (root < 1e-6f || tip / root < m_cfg.openPalmRatio) return false;
    }
    {
        float tip = distance(kp.points[12].x, kp.points[12].y, w.x, w.y);
        float root = distance(kp.points[9].x, kp.points[9].y, w.x, w.y);
        if (root < 1e-6f || tip / root < m_cfg.openPalmRatio) return false;
    }
    // 无名指(16)/小指(20)必须弯曲：到手腕距离 < 阈值
    if (distance(kp.points[16].x, kp.points[16].y, w.x, w.y) >= m_cfg.dragFoldDistPx) return false;
    if (distance(kp.points[20].x, kp.points[20].y, w.x, w.y) >= m_cfg.dragFoldDistPx) return false;
    return true;  // 双指V成立
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

    // 2. 手不存在或置信度过低：重置手势状态，若拖拽中先强制结束。
    //    注意：不再将 LOCKED 回退到 IDLE——手短暂离开画面应保持锁定状态，
    //    否则手一离开画面就解锁，用户体验极差（保持 LOCKED 直到用户主动开掌解锁）。
    bool lowConfidence = (kpRaw.confidence < 0.5f);
    if (!kp.valid || kp.points.size() < kHandKeypointCount || lowConfidence) {
        m_holdMs = 0;
        GestureEvent lostEvent = GestureEvent::kNone;
        if (m_dragging || m_pinched) {
            m_dragging = false;
            m_pinched = false;
            lostEvent = GestureEvent::kDragEnd;
            std::printf("[FSM] 手丢失，拖拽强制结束\n");
        }
        // 保持当前状态不变（IDLE 保持 IDLE，LOCKED 保持 LOCKED）
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
            // LOCKED 态：捏合单击 + 双指V拖拽 + 开掌解锁 + 食指伸出移动
            // 记录平滑后掌心（点9，中指根）坐标供上层做相对位移
            m_smoothPalmX = kp.points[9].x;
            m_smoothPalmY = kp.points[9].y;

            // 捏合判定：拇指尖(4)到食指尖(8)距离 / 中指根(9)到手腕(0)距离 < 阈值
            // 比例法对"手离摄像头远近"鲁棒。注意：必须放在开掌判定之前！
            //   原因：捏合时其余手指可能仍伸直（OK 手势），会被 detectOpenPalm
            //   误判为开掌 → 走解锁分支 → 捏合永不触发（用户反馈"完全没反应"）。
            float pinchDist = distance(kp.points[4].x, kp.points[4].y,
                                       kp.points[8].x, kp.points[8].y);
            float handSize = distance(kp.points[9].x, kp.points[9].y,
                                      kp.points[0].x, kp.points[0].y);
            float pinchRatioNow = (handSize > 1e-6f) ? (pinchDist / handSize) : 1.f;
            bool pinched = (pinchRatioNow < m_cfg.pinchRatio);

            // 诊断：每 30 帧打印一次当前捏合比例（无论是否触发），
            // 便于观察捏合动作时的实际比例值，调整 pinch_ratio 阈值。
            static int pinchDiagCnt = 0;
            if (++pinchDiagCnt % 30 == 1) {
                std::printf("[FSM] 捏合比例=%.2f (阈值 %.2f) dist=%.0fpx handSize=%.0fpx%s\n",
                            pinchRatioNow, m_cfg.pinchRatio, pinchDist, handSize,
                            pinched ? " → 捏合" : "");
            }

            // 拖拽判定：双指V手势（食指+中指伸直、无名指+小指弯曲）
            bool twoFinger = detectTwoFinger(kp);

            // ---- 捏合单击（最高优先级，边沿触发防连发）----
            if (pinched) {
                if (!m_pinched) {
                    m_pinched = true;
                    m_holdMs = 0;  // 捏合中断开掌解锁计时
                    event = GestureEvent::kClick;
                    std::printf("[FSM] 单击（捏合 tap, 比例=%.2f）\n", pinchRatioNow);
                }
                // 捏合保持中：不发移动，避免误动
                break;
            }
            m_pinched = false;  // 解除捏合，允许下次边沿触发

            // ---- 双指V拖拽：显式手势 ----
            if (twoFinger) {
                if (!m_dragging) {
                    m_dragging = true;
                    event = GestureEvent::kDragStart;
                    std::printf("[FSM] 拖拽开始（双指V）\n");
                } else {
                    event = GestureEvent::kDragMove;  // 双指V保持：拖拽移动
                }
                break;  // 双指V期间不处理移动/开掌解锁
            }

            // ---- 拖拽结束：双指V解除 ----
            if (m_dragging) {
                m_dragging = false;
                event = GestureEvent::kDragEnd;
                std::printf("[FSM] 拖拽结束（解除双指V）\n");
                break;
            }

            // ---- 开掌解锁（非捏合、非双指V时才判定）----
            if (detectOpenPalm(kp)) {
                m_holdMs += static_cast<int>(dtMs);
                if (m_holdMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    m_state = CtrlState::kIdle;
                    m_holdMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;
                    m_dragging = false;
                    event = GestureEvent::kOpenPalmHold;
                    std::printf("[FSM] 状态迁移：LOCKED → IDLE（已解锁）\n");
                }
                break;  // 开掌期间不响应移动
            }
            m_holdMs = (m_holdMs > 30) ? (m_holdMs - 30) : 0;

            // ---- 食指伸出：相对位移移动 ----
            event = GestureEvent::kPointerMove;
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

void GestureFSM::lastPalmPoint(float& x, float& y) const {
    x = m_smoothPalmX;
    y = m_smoothPalmY;
}

} // namespace hand_ctrl
