// ============================================================================
// gesture_fsm/fsm_test.cpp
// 作用：手势 FSM 模块【独立验收测试程序】（v2 单指方案）。
// 行为：
//   - 加载 gesture_config.json
//   - 从摄像头读取帧 → ONNX 推理 → 卡尔曼平滑 → FSM 推进 → 打印状态/事件
// 用法：
//   ./fsm_test [配置文件] [模型目录] [设备索引]
//   ./fsm_test config/gesture_config.json models 0
// 说明：验证 1.2s 开掌锁定倒计时、单指定位/点击/拖拽事件，无 uinput 输出（仅打印）。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <csignal>
#include <string>
#include <opencv2/opencv.hpp>
#include "gesture_fsm/gesture_fsm.h"
#include "infer_base/infer_base.h"
#include "infer_base/onnx_infer.h"
#include "camera_capture/camera.h"

using namespace hand_ctrl;

// 全局退出标志：Ctrl+C 触发，主循环检测后优雅退出
static volatile sig_atomic_t g_shouldExit = 0;
static void onSignal(int) { g_shouldExit = 1; }

// 手势事件名（用于日志打印，v2 单指方案）
static const char* eventToString(GestureEvent e) {
    switch (e) {
        case GestureEvent::kNone:        return "无";
        case GestureEvent::kOpenPalmHold: return "开掌长按(锁定/解锁)";
        case GestureEvent::kPointerMove: return "食指定位移动";
        case GestureEvent::kClick:       return "食指点击(左键)";
        case GestureEvent::kDragStart:   return "拖拽开始";
        case GestureEvent::kDragMove:    return "拖拽移动";
        case GestureEvent::kDragEnd:     return "拖拽结束";
    }
    return "未知";
}

int main(int argc, char* argv[]) {
    // 参数解析：[配置文件] [模型目录] [设备索引] [-v 显示画面]
    std::string configPath = (argc > 1) ? argv[1] : "config/gesture_config.json";
    std::string modelDir    = (argc > 2) ? argv[2] : "models";
    int deviceIndex         = (argc > 3) ? std::atoi(argv[3]) : 0;
    bool showGui = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) showGui = true;
    }
    std::printf("[fsm_test] 配置=%s 模型=%s 摄像头=%d 画面=%s\n",
                configPath.c_str(), modelDir.c_str(), deviceIndex, showGui ? "ON" : "OFF");

    // 注册信号处理：Ctrl+C 优雅退出（避免 ^C 后卡在资源清理）
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // 1. 加载手势配置
    GestureFSM fsm;
    if (!fsm.loadConfig(configPath)) {
        std::printf("[fsm_test] 配置加载失败，使用默认参数。\n");
    }

    // 2. 加载推理模型
    ONNXInfer infer;
    ONNXModelPaths paths;
    paths.palmModel     = modelDir + "/palm_detection.onnx";
    paths.landmarkModel = modelDir + "/hand_landmark.onnx";
    if (!infer.loadModels(paths)) {
        std::printf("[fsm_test] 模型加载失败。\n");
        return 1;
    }

    // 3. 打开摄像头
    Camera camera;
    if (!camera.open(deviceIndex)) {
        std::printf("[fsm_test] 摄像头打开失败。\n");
        return 1;
    }

    // 4. 主循环：采集 → 推理 → FSM（最长运行 60 秒自动退出，避免无限卡住）
    cv::Mat frame;
    HandKeypoints kp;
    auto lastTime = std::chrono::steady_clock::now();
    auto startTime = lastTime;
    int frameCnt = 0;
    int eventCnt[7] = {0};  // v2 共 7 个事件（kNone=0 ~ kDragEnd=6）

    while (true) {
        // 信号检测：Ctrl+C 优雅退出
        if (g_shouldExit) {
            std::printf("[fsm_test] 收到退出信号，正在退出...\n");
            break;
        }
        // 60 秒超时保护
        auto now = std::chrono::steady_clock::now();
        double totalSec = std::chrono::duration<double>(now - startTime).count();
        if (totalSec > 60.0) {
            std::printf("[fsm_test] 60秒超时，自动退出。\n");
            break;
        }
        double dtMs = std::chrono::duration<double, std::milli>(now - lastTime).count();
        lastTime = now;

        if (!camera.readFrame(frame)) {
            continue;
        }

        if (infer.infer(frame, kp) && kp.valid) {
            // 推进状态机
            GestureEvent e = fsm.handleFrame(kp, dtMs);
            ++frameCnt;

            // 调试：每 30 帧打印关键诊断信息（手腕位置 + 握拳距离 + 状态）
            if (frameCnt % 30 == 0) {
                const auto& w = kp.points[0];
                float maxDist = 0;
                const int tips[] = {4, 8, 12, 16, 20};
                for (int t : tips) {
                    float dx = kp.points[t].x - w.x;
                    float dy = kp.points[t].y - w.y;
                    float d = std::sqrt(dx*dx + dy*dy);
                    if (d > maxDist) maxDist = d;
                }
                std::printf("[fsm_test] 帧#%d 状态=%s 手腕=(%.0f,%.0f) 指尖最大距离=%.1fpx (阈值110) dtMs=%.0f\n",
                            frameCnt, fsm.currentStateName(), w.x, w.y, maxDist, dtMs);
            }
            // 有事件时打印
            if (e != GestureEvent::kNone) {
                ++eventCnt[static_cast<int>(e)];
                std::printf("[fsm_test] 事件=%s (状态=%s)\n",
                            eventToString(e), fsm.currentStateName());
            }

            // 可视化：默认关闭 GUI（VMware 无 GPU 加速下 imshow 渲染会卡死采集循环）
            //          如需观察画面，加 -v 参数启动
            if (showGui && frameCnt % 2 == 0) {
                cv::Mat vis = frame.clone();
                const int bones[][2] = {
                    {0,1},{1,2},{2,3},{3,4},
                    {0,5},{5,6},{6,7},{7,8},
                    {5,9},{9,10},{10,11},{11,12},
                    {9,13},{13,14},{14,15},{15,16},
                    {13,17},{17,18},{18,19},{19,20},
                    {0,17}
                };
                for (auto& b : bones) {
                    cv::line(vis,
                        cv::Point(static_cast<int>(kp.points[b[0]].x), static_cast<int>(kp.points[b[0]].y)),
                        cv::Point(static_cast<int>(kp.points[b[1]].x), static_cast<int>(kp.points[b[1]].y)),
                        cv::Scalar(0, 255, 0), 2);
                }
                for (int i = 0; i < 21; ++i) {
                    cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                    cv::circle(vis,
                        cv::Point(static_cast<int>(kp.points[i].x), static_cast<int>(kp.points[i].y)),
                        4, color, -1);
                }
                std::string stateText = std::string("State: ") + fsm.currentStateName();
                cv::putText(vis, stateText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
                cv::imshow("hand-ctrl fsm test (ESC to exit)", vis);
            }
        }

        // ESC 退出：仅在 GUI 模式下接收按键；非 GUI 模式用 Ctrl+C 终止
        if (showGui) {
            int key = cv::waitKey(1);
            if (key == 27) break;
        }
    }

    camera.release();
    cv::destroyAllWindows();
    std::printf("[fsm_test] 结束。共 %d 帧，事件统计:\n", frameCnt);
    for (int i = 1; i <= 6; ++i) {  // kOpenPalmHold(1) ~ kDragEnd(6)
        std::printf("  %-20s: %d 次\n", eventToString(static_cast<GestureEvent>(i)), eventCnt[i]);
    }
    return 0;
}
