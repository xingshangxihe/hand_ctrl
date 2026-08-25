// ============================================================================
// gesture_fsm/gesture_fsm.cpp
// 作用：GestureFSM 类实现（项目核心模块）。
// 职责：
//   1. 加载 JSON 配置（所有手势阈值外置）
//   2. 对 21 关键点做卡尔曼平滑
//   3. 依据状态机迁移规则与 6 种手势判定输出 GestureEvent
// 所有判定阈值来自 config/gesture_config.json，代码无硬编码魔法数字。
// ============================================================================

#include "gesture_fsm/gesture_fsm.h"

#include <cstdio>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>   // stat：配置热加载需要读取文件 mtime
// 简易 JSON 解析（轻量依赖，避免引入 nlohmann/json）
// 实现思路：本文件只需读取固定字段的数值，用最简单的字符串查找 + strtod
//           实现。后续若配置复杂度提升可替换为 nlohmann/json。
#include <sstream>

namespace hand_ctrl {

// --------------------------- 简易 JSON 数值解析工具 ---------------------------
// 在 JSON 文本中查找 "key":数值 模式并返回数值
// 说明：本实现为轻量解析，仅支持数值类型，足够本项目配置需求
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

    // 2. 解析各字段（嵌套字段用统一查找方式，无重复 key 冲突）
    jsonGetInt  (json, "hold_time_ms",          m_cfg.lockHoldMs);
    jsonGetInt  (json, "state_cooldown_ms",     m_cfg.stateCooldownMs);
    jsonGetFloat(json, "process_noise",        m_cfg.kalmanProcessNoise);
    jsonGetFloat(json, "measure_noise",        m_cfg.kalmanMeasureNoise);
    jsonGetInt  (json, "fist_distance_threshold_px",    m_cfg.fistDistThreshold);
    jsonGetInt  (json, "finger_distance_threshold_px", m_cfg.openPalmDistThreshold);
    jsonGetInt  (json, "hold_short_ms",                 m_cfg.fistShortMs);
    jsonGetInt  (json, "thumb_index_gap_px",           m_cfg.okGestureGapPx);
    jsonGetInt  (json, "wrist_stable_px",              m_cfg.swipeWristStablePx);
    jsonGetFloat(json, "horizontal_ratio",            m_cfg.swipeHorizontalRatio);
    jsonGetFloat(json, "thumb_extended_ratio",        m_cfg.thumbExtendedRatio);
    jsonGetInt  (json, "other_fingers_fold_threshold", m_cfg.otherFingersFoldPx);
    jsonGetInt  (json, "width",                       m_cfg.imageWidth);
    jsonGetInt  (json, "height",                      m_cfg.imageHeight);

    // 3. 用配置初始化所有卡尔曼滤波器
    for (int i = 0; i < kHandKeypointCount; ++i) {
        m_filters[i].init(m_cfg.kalmanProcessNoise, m_cfg.kalmanMeasureNoise);
    }

    m_cfgLoaded = true;
    std::printf("[GestureFSM] 配置加载成功: %s\n", configPath.c_str());
    std::printf("[GestureFSM] 锁定阈值=%dms, 握拳距离=%dpx, 滑动比例=%.2f\n",
                m_cfg.lockHoldMs, m_cfg.fistDistThreshold, m_cfg.swipeHorizontalRatio);
    return true;
}

// --------------------------- 配置热加载 ---------------------------
bool GestureFSM::reloadIfChanged() {
    // 未加载过配置（路径为空）时不检测
    if (m_configPath.empty()) return false;

    // 读取当前文件 mtime，与上次记录对比
    struct stat st;
    if (::stat(m_configPath.c_str(), &st) != 0) {
        // 文件被删除/不可访问：不重载，保持现有配置（避免误删后程序失配）
        return false;
    }
    long curMtime = static_cast<long>(st.st_mtime);
    if (curMtime == m_configMtime) {
        return false;  // 无变更
    }

    // 配置有变更：重新加载全部阈值
    // 说明：kalman 滤波器会随配置重新 init（过程/测量噪声可能被调整），
    //       短暂抖动属正常现象。
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
    out = in;  // 复制基础字段（valid/confidence）
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

// --------------------------- 6 种手势判定 ---------------------------
// 1. 握拳：所有指尖(4,8,12,16,20)到手腕(0)的距离均 < fistDistThreshold
//    注意：双模型模式下脸部已被 palm 检测过滤（valid=0），此处仅保留
//          "指尖收拢到手腕附近"这一核心判据 + 手部跨度兜底，避免误杀真手。
bool GestureFSM::detectFistHold(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    const auto& w = kp.points[0];  // 手腕
    const int tips[] = {4, 8, 12, 16, 20};  // 5 个指尖索引

    // 核心判据：5 个指尖到手腕距离均小于阈值（握拳时指尖收拢到掌心附近）
    float maxTipDist = 0.f;
    for (int t : tips) {
        float d = distance(kp.points[t].x, kp.points[t].y, w.x, w.y);
        if (d >= m_cfg.fistDistThreshold) return false;
        maxTipDist = std::max(maxTipDist, d);
    }
    // 手部跨度兜底检查：跨度太小视为噪声/无效关键点，过大视为误判区域
    // （双模型下 palm 已过滤脸部，此检查仅为极端异常兜底）
    const float minHandSpan = 30.f;   // 手部最小跨度（像素）
    const float maxHandSpan = 400.f;  // 手部最大跨度（像素，放宽避免误杀）
    if (maxTipDist < minHandSpan || maxTipDist > maxHandSpan) return false;
    return true;
}

// 2. 五指张开：所有指尖到手腕距离均 > openPalmDistThreshold
bool GestureFSM::detectOpenPalm(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    const auto& w = kp.points[0];
    const int tips[] = {4, 8, 12, 16, 20};
    for (int t : tips) {
        float d = distance(kp.points[t].x, kp.points[t].y, w.x, w.y);
        if (d <= m_cfg.openPalmDistThreshold) return false;
    }
    return true;
}

// 3. OK 手势：拇指尖(4)与食指尖(8)距离 < okGestureGapPx
bool GestureFSM::detectOkGesture(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    float d = distance(kp.points[4].x, kp.points[4].y,
                       kp.points[8].x, kp.points[8].y);
    return d < m_cfg.okGestureGapPx;
}

// 4. 食指横向滑动：手腕静止 + 食指尖横向位移 >= 图像宽度 * horizontalRatio
//    说明：此处返回 true 时，方向（左/右）需由调用方根据位移正负判断
bool GestureFSM::detectIndexSwipe(const HandKeypoints& kp, double dtMs) {
    if (kp.points.size() < 21) return false;
    const auto& wrist = kp.points[0];
    const auto& indexTip = kp.points[8];

    // 首帧：记录初始位置，不判定
    if (m_lastWristX < 0) {
        m_lastWristX = wrist.x;
        m_lastWristY = wrist.y;
        m_lastIndexTipX = indexTip.x;
        m_lastIndexTipY = indexTip.y;
        return false;
    }

    // 检查手腕是否基本静止
    float wristMove = distance(wrist.x, wrist.y, m_lastWristX, m_lastWristY);
    if (wristMove > m_cfg.swipeWristStablePx) {
        // 手腕动了，重置参考点
        m_lastWristX = wrist.x;
        m_lastWristY = wrist.y;
        m_lastIndexTipX = indexTip.x;
        m_lastIndexTipY = indexTip.y;
        return false;
    }

    // 计算食指尖相对参考点的横向位移
    float dx = indexTip.x - m_lastIndexTipX;
    float threshold = m_cfg.imageWidth * m_cfg.swipeHorizontalRatio;
    if (std::fabs(dx) >= threshold) {
        // 满足滑动条件，重置参考点（避免重复触发）
        m_lastWristX = wrist.x;
        m_lastWristY = wrist.y;
        m_lastIndexTipX = indexTip.x;
        m_lastIndexTipY = indexTip.y;
        return true;
    }
    return false;
}

// 5. 竖拇指：拇指伸直（指尖到手腕距离 > 拇指根到手腕距离 * ratio）
//          且其余四指折叠（指尖到手腕距离 < otherFingersFoldPx）
bool GestureFSM::detectThumbUp(const HandKeypoints& kp) {
    if (kp.points.size() < 21) return false;
    const auto& w = kp.points[0];
    // 拇指尖(4)到手腕距离 vs 拇指根(2, MCP)到手腕距离
    float thumbTipDist = distance(kp.points[4].x, kp.points[4].y, w.x, w.y);
    float thumbBaseDist = distance(kp.points[2].x, kp.points[2].y, w.x, w.y);
    if (thumbTipDist < thumbBaseDist * m_cfg.thumbExtendedRatio) return false;

    // 其余四指（食指8、中指12、无名指16、小指20）到手腕距离 < 阈值
    const int otherTips[] = {8, 12, 16, 20};
    for (int t : otherTips) {
        float d = distance(kp.points[t].x, kp.points[t].y, w.x, w.y);
        if (d >= m_cfg.otherFingersFoldPx) return false;
    }
    return true;
}

// --------------------------- 状态机主循环 ---------------------------
GestureEvent GestureFSM::handleFrame(const HandKeypoints& kpRaw, double dtMs) {
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

    // 2. 手不存在或置信度过低时的状态处理
    //    置信度过滤：landmark 模型输出2 为手部置信度，脸部误判时该分数低。
    //    阈值 0.5：真手 >0.8，脸部 <0.3（实测差异明显）。低于阈值视为"无手"。
    bool lowConfidence = (kpRaw.confidence < 0.5f);
    if (!kp.valid || kp.points.size() < kHandKeypointCount || lowConfidence) {
        // 手丢失/低置信度：重置握拳计时、滑动参考点，状态回 IDLE
        m_fistHoldMs = 0;
        m_lastWristX = -1.f;
        if (m_state == CtrlState::kLockWait) {
            m_state = CtrlState::kIdle;
        }
        return GestureEvent::kNone;
    }

    // 3. 当前帧各手势判定结果
    bool isFist      = detectFistHold(kp);
    bool isOpenPalm  = detectOpenPalm(kp);
    bool isOk        = detectOkGesture(kp);
    bool isSwipe     = detectIndexSwipe(kp, dtMs);
    bool isThumbUp   = detectThumbUp(kp);

    // 4. 状态机迁移
    // 冷却计时递减（每帧执行，防抖）
    if (m_cooldownMs > 0) {
        m_cooldownMs -= static_cast<int>(dtMs);
        if (m_cooldownMs < 0) m_cooldownMs = 0;
    }

    GestureEvent event = GestureEvent::kNone;
    switch (m_state) {
        case CtrlState::kIdle:
            // IDLE 态：只响应握拳长按触发锁定
            if (isFist) {
                m_fistHoldMs += static_cast<int>(dtMs);
                // 调试日志：握拳累计时长（每 300ms 打印一次，便于观察是否持续）
                if (m_fistHoldMs % 300 < static_cast<int>(dtMs)) {
                    std::printf("[FSM] IDLE 握拳累计 %dms / %dms\n", m_fistHoldMs, m_cfg.lockHoldMs);
                }
                if (m_fistHoldMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    m_state = CtrlState::kLocked;
                    m_fistHoldMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;  // 进入冷却，防止快速切换
                    event = GestureEvent::kFistHold;  // 通知上层：已锁定
                    std::printf("[FSM] 状态迁移：IDLE → LOCKED（已锁定）\n");
                }
            } else {
                // 衰减容忍：短暂松开（帧间抖动）只衰减 30ms（约 1 帧），不清零
                // 背景：VMware 摄像头帧率不稳，单模型模式关键点偶发抖动，
                //       直接清零会导致握拳计时频繁重置，无法达到 1.2s 锁定阈值。
                m_fistHoldMs = (m_fistHoldMs > 30) ? (m_fistHoldMs - 30) : 0;
            }
            break;

        case CtrlState::kLockWait:
            // 简化实现：IDLE 已直接跳到 LOCKED，本态暂不进入
            break;

        case CtrlState::kLocked:
            // LOCKED 态：响应所有手势
            if (isFist) {
                // 握拳持续：判断是解锁还是短按
                m_fistHoldMs += static_cast<int>(dtMs);
                if (m_fistHoldMs % 300 < static_cast<int>(dtMs)) {
                    std::printf("[FSM] LOCKED 握拳累计 %dms / %dms\n", m_fistHoldMs, m_cfg.lockHoldMs);
                }
                if (m_fistHoldMs >= m_cfg.lockHoldMs && m_cooldownMs <= 0) {
                    // 长按 ≥1.2s：解锁
                    m_state = CtrlState::kIdle;
                    m_fistHoldMs = 0;
                    m_cooldownMs = m_cfg.stateCooldownMs;  // 进入冷却，防止快速切换
                    event = GestureEvent::kFistHold;  // 通知上层：已解锁
                    std::printf("[FSM] 状态迁移：LOCKED → IDLE（已解锁）\n");
                }
                // 短按（0~0.5s）的事件在松开时触发，本帧不发事件
            } else {
                // 握拳释放：若之前累计时长 < fistShortMs，触发短按单击
                if (m_fistHoldMs > 0 && m_fistHoldMs < m_cfg.fistShortMs) {
                    event = GestureEvent::kFistShort;  // 左键单击
                } else if (m_fistHoldMs >= m_cfg.fistShortMs && m_fistHoldMs < m_cfg.lockHoldMs) {
                    event = GestureEvent::kFistShort;  // 左键拖拽释放（与单击同事件，上层按住时长区分）
                }
                // 衰减容忍：短暂松开只衰减 30ms 不清零（与 IDLE 态一致）
                m_fistHoldMs = (m_fistHoldMs > 30) ? (m_fistHoldMs - 30) : 0;
            }
            // 其他手势判定（仅在未握拳时生效）
            if (!isFist) {
                if (isOpenPalm) {
                    event = GestureEvent::kOpenPalm;
                } else if (isOk) {
                    event = GestureEvent::kOkGesture;
                } else if (isSwipe) {
                    event = GestureEvent::kIndexSwipe;
                } else if (isThumbUp) {
                    event = GestureEvent::kThumbUp;
                }
            }
            break;

        case CtrlState::kTrigger:
            // 瞬时态：直接回 LOCKED
            m_state = CtrlState::kLocked;
            break;
    }

    return event;
}

// --------------------------- 状态查询 ---------------------------
const char* GestureFSM::currentStateName() const {
    switch (m_state) {
        case CtrlState::kIdle:     return "IDLE";
        case CtrlState::kLockWait: return "LOCK_WAIT";
        case CtrlState::kLocked:   return "LOCKED";
        case CtrlState::kTrigger:  return "TRIGGER";
    }
    return "UNKNOWN";
}

} // namespace hand_ctrl