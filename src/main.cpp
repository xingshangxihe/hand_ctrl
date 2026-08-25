// ============================================================================
// src/main.cpp
// 作用：程序入口 + 多线程调度器（v2 单指控制方案）。
//
// 线程架构（三线程解耦，避免阻塞卡顿）：
//   采集线程:   Camera::readFrame → 帧队列
//   推理线程:   从帧队列取帧 → ONNXInfer::infer → 关键点队列
//   主线程  :   从关键点队列取 → GestureFSM::handleFrame → UInputManager 输出
//
// v2 交互模型（方案A：单指绝对定位）：
//   锁定后食指指尖 = 鼠标指针（画面坐标等比映射屏幕坐标）
//   快速下点 = 单击；下点按住 = 拖拽；握拳1.2s = 锁定/解锁
//
// 绝对定位实现要点：
//   uinput 是相对位移设备，需软件维护"虚拟鼠标位置"：
//   每次计算 目标位置 - 虚拟位置 = 相对位移，注入后更新虚拟位置。
//   （若虚拟位置与真实指针脱节，可用物理鼠标微调，或重新锁定/解锁一次校准）
//
// 线程间通信：std::mutex + std::condition_variable + 共享队列（生产-消费者模型）
// 信号处理：Ctrl+C 优雅退出，析构释放所有资源（RAII）
//
// 用法：
//   sudo ./hand_ctrl [配置文件] [模型目录] [设备索引] [-v] [-q]
// ============================================================================

#include <cstdio>
#include <cstring>
#include <csignal>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include "utils/common.h"
#include "camera_capture/camera.h"
#include "infer_base/infer_base.h"
#include "infer_base/onnx_infer.h"
#include "gesture_fsm/gesture_fsm.h"
#include "linux_uinput/uinput_manager.h"

using namespace hand_ctrl;

// 前向声明：事件转字符串（供 main 函数日志打印）
static const char* eventToString(GestureEvent e);

// --------------------------- 全局退出标志 ---------------------------
static std::atomic<bool> g_shouldExit{false};

// 全局静默开关：-q 参数启用后，关闭推理/采集心跳日志，只保留事件触发与错误日志
// 说明：正式使用（上架演示）时画面更干净，调试时仍可看到完整流程
static std::atomic<bool> g_quiet{false};

static void onSignal(int) {
    g_shouldExit.store(true);
    std::printf("\n[main] 收到退出信号，正在停止线程...\n");
}

// --------------------------- 线程安全队列 ---------------------------
// 生产-消费者模型：采集→推理、推理→FSM 之间各一个
template <typename T>
class ThreadSafeQueue {
public:
    // 推入元素（生产者调用）
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // 限制队列长度，避免积压导致延迟（最新帧覆盖旧帧）
            while (m_queue.size() >= m_maxSize) {
                m_queue.pop();
            }
            m_queue.push(std::move(item));
        }
        m_cv.notify_one();
    }

    // 取出元素（消费者调用，阻塞直到有数据或退出）
    bool pop(T& item, int timeoutMs = 100) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_cv.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                           [this] { return !m_queue.empty(); }) ) {
            return false;  // 超时
        }
        item = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    void setMaxSize(size_t n) { m_maxSize = n; }

    // 查询当前队列长度（仅供可视化参考，非精确同步）
    size_t size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

private:
    std::queue<T> m_queue;
    std::mutex    m_mutex;
    std::condition_variable m_cv;
    size_t m_maxSize = 2;  // 队列最大长度（保持低延迟，旧帧被丢弃）
};

// --------------------------- 主函数 ---------------------------
int main(int argc, char* argv[]) {
    // 参数解析：[配置文件] [模型目录] [设备索引] [-v 显示画面] [-s 保存截图] [-q 静默]
    std::string configPath = (argc > 1) ? argv[1] : "config/gesture_config.json";
    std::string modelDir    = (argc > 2) ? argv[2] : "models";
    int deviceIndex         = (argc > 3) ? std::atoi(argv[3]) : 0;
    bool showGui = false;
    bool saveShot = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) showGui = true;
        if (std::strcmp(argv[i], "-s") == 0) saveShot = true;
        if (std::strcmp(argv[i], "-q") == 0) g_quiet.store(true);
    }

    std::printf("[main] hand-ctrl v%s 启动\n", kProjectVersion);
    std::printf("[main] 配置=%s 模型=%s 摄像头=%d 画面=%s 静默=%s\n",
                configPath.c_str(), modelDir.c_str(), deviceIndex, showGui ? "ON" : "OFF",
                g_quiet.load() ? "ON" : "OFF");

    // 关键修复：stdout 设为无缓冲，避免 printf 输出积压导致"假卡死"
    // 说明：默认 stdout 在非交互式或重定向时是全缓冲，会积压大量日志不显示
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    // 1. 注册信号处理
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    // 2. 加载手势 FSM 配置（v2 单指方案）
    GestureFSM fsm;
    if (!fsm.loadConfig(configPath)) {
        std::fprintf(stderr, "[main] FSM 配置加载失败，使用默认参数\n");
    }

    // 3. 加载推理模型
    ONNXInfer infer;
    ONNXModelPaths paths;
    paths.palmModel     = modelDir + "/palm_detection.onnx";
    paths.landmarkModel = modelDir + "/hand_landmark.onnx";
    if (!infer.loadModels(paths)) {
        std::fprintf(stderr, "[main] 推理模型加载失败，程序退出\n");
        return static_cast<int>(ErrorCode::kModelLoadFailed);
    }

    // 4. 打开摄像头
    Camera camera;
    if (!camera.open(deviceIndex)) {
        std::fprintf(stderr, "[main] 摄像头打开失败，程序退出\n");
        return static_cast<int>(ErrorCode::kCameraOpenFailed);
    }

    // 5. 创建虚拟键鼠
    UInputManager uinput;
    if (!uinput.createDevices()) {
        std::fprintf(stderr, "[main] uinput 设备创建失败，程序退出\n");
        return static_cast<int>(ErrorCode::kUInputCreateFailed);
    }

    // 6. 线程间队列：采集→推理、推理→FSM
    ThreadSafeQueue<cv::Mat> frameQueue;       // 帧队列
    ThreadSafeQueue<HandKeypoints> kpQueue;     // 关键点队列
    frameQueue.setMaxSize(1);  // 只保留最新帧（保证低延迟，旧帧丢弃）
    kpQueue.setMaxSize(1);

    // 共享最新帧：供主线程可视化使用（采集线程更新，主线程读取）
    std::mutex frameMutex;
    cv::Mat latestFrame;

    // 共享最新关键点：供独立显示线程绘制（主线程更新，显示线程读取）
    std::mutex kpMutex;
    HandKeypoints latestKp;

    // 7. 采集线程：持续读帧（失败连续超过阈值自动重开摄像头）
    std::thread captureThread([&]() {
        cv::Mat frame;
        int capCnt = 0;
        int failCnt = 0;
        while (!g_shouldExit.load()) {
            if (camera.readFrame(frame)) {
                frameQueue.push(frame.clone());  // clone 避免 Camera 缓冲被覆盖
                // 更新共享最新帧（供主线程可视化）
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    frame.copyTo(latestFrame);
                }
                ++capCnt;
                failCnt = 0;
                // 心跳：每 60 帧打印一次采集状态（-q 静默模式下不打印）
                if (capCnt % 60 == 0 && !g_quiet.load()) {
                    std::printf("[capture] 已采集 %d 帧\n", capCnt);
                }
            } else {
                // 采集失败：计数并定期打印日志（避免静默卡死）
                ++failCnt;
                if (failCnt % 10 == 1) {
                    std::printf("[capture] 读帧失败累计=%d\n", failCnt);
                }
                // 连续失败过多（约 5 秒无帧）：自动重开摄像头恢复
                if (failCnt == 50) {
                    std::printf("[capture] 连续失败过多，自动重开摄像头...\n");
                    camera.release();
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    camera.open(deviceIndex);
                    failCnt = 0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        std::printf("[capture] 线程退出 (成功=%d 失败=%d)\n", capCnt, failCnt);
    });

    // 8. 推理线程：从帧队列取帧 → 推理 → 推入关键点队列
    std::thread inferThread([&]() {
        cv::Mat frame;
        HandKeypoints kp;
        int inferCnt = 0;
        int popFailCnt = 0;
        int inferFailCnt = 0;
        while (!g_shouldExit.load()) {
            if (!frameQueue.pop(frame, 100)) {
                // 帧队列空：累计计数
                ++popFailCnt;
                if (popFailCnt % 10 == 0) {
                    std::printf("[infer] 帧队列空，已等待 %d 次\n", popFailCnt);
                }
                continue;
            }
            popFailCnt = 0;

            auto t0 = std::chrono::steady_clock::now();
            bool ok = false;
            try {
                ok = infer.infer(frame, kp);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[infer] 异常：%s\n", e.what());
                ++inferFailCnt;
                continue;
            } catch (...) {
                std::fprintf(stderr, "[infer] 未知异常\n");
                ++inferFailCnt;
                continue;
            }
            auto t1 = std::chrono::steady_clock::now();
            double inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (ok) {
                kpQueue.push(kp);
                ++inferCnt;
                // 每 10 次打印推理状态（含是否检测到手；-q 静默模式下不打印）
                if (inferCnt % 10 == 0 && !g_quiet.load()) {
                    std::printf("[infer] #=%d 耗时=%.1fms valid=%d\n",
                                inferCnt, inferMs, (int)kp.valid);
                }
            } else {
                ++inferFailCnt;
                if (inferFailCnt % 10 == 0) {
                    std::printf("[infer] 推理失败累计=%d\n", inferFailCnt);
                }
            }
        }
        std::printf("[infer] 线程退出 (总成功=%d 失败=%d)\n", inferCnt, inferFailCnt);
    });

    // 9. 主线程：从关键点队列取 → FSM 推进 → uinput 输出 + 时间驱动画面显示
    //    注意：OpenCV highgui 的 GTK 后端要求 imshow/waitKey 在主线程调用，
    //          不能放独立线程（否则窗口不创建）。故显示放主线程，
    //          用时间驱动（每 40ms 刷新）避免每帧渲染拖慢链路。
    HandKeypoints kp;
    auto lastTime = std::chrono::steady_clock::now();
    auto lastDisplay = std::chrono::steady_clock::now();
    int eventCnt = 0;
    int fsmCnt = 0;
    int waitLogCnt = 0;

    // v2 绝对定位：软件维护虚拟鼠标位置（相对位移设备无法直接"跳到"目标坐标）
    // 映射公式：screenPos = fingerPos / imageSize * screenSize
    // 每次注入 (虚拟目标 - 当前虚拟位置) 的相对位移，并更新虚拟位置。
    // 锁定瞬间虚拟位置初始化为屏幕中心（等价于手指落在画面中心时指针在屏幕中心）。
    const int scrW = fsm.screenWidth();
    const int scrH = fsm.screenHeight();
    const int imgW = fsm.imageWidth();
    const int imgH = fsm.imageHeight();
    float virtualMouseX = scrW / 2.f;   // 虚拟鼠标当前位置（屏幕坐标）
    float virtualMouseY = scrH / 2.f;
    bool  virtualInit = false;          // 锁定后首帧需初始化虚拟位置
    while (!g_shouldExit.load()) {
        // ---- 画面显示：放在循环最前，不依赖关键点队列（摄像头断流时也持续刷新）----
        // 说明：imshow/waitKey 必须在主线程调用（GTK 限制）；时间驱动每 40ms 刷新。
        if (showGui) {
            auto nowD = std::chrono::steady_clock::now();
            double dispInterval = std::chrono::duration<double, std::milli>(nowD - lastDisplay).count();
            if (dispInterval >= 100.0) {  // 10fps，降低 VMware 渲染压力
                lastDisplay = nowD;
                cv::Mat vis;
                bool frameOk = false;
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    if (!latestFrame.empty()) {
                        latestFrame.copyTo(vis);
                        frameOk = true;
                    }
                }
                if (vis.empty()) vis = cv::Mat::zeros(480, 640, CV_8UC3);
                // 用共享最新关键点绘制（latestKp 可能为空则只显示画面）
                HandKeypoints dispKp;
                {
                    std::lock_guard<std::mutex> lock(kpMutex);
                    dispKp = latestKp;
                }
                if (dispKp.valid && dispKp.points.size() >= 21) {
                    const int bones[][2] = {
                        {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
                        {5,9},{9,10},{10,11},{11,12},{9,13},{13,14},{14,15},{15,16},
                        {13,17},{17,18},{18,19},{19,20},{0,17}
                    };
                    for (auto& b : bones) {
                        if (b[0] < (int)dispKp.points.size() && b[1] < (int)dispKp.points.size()) {
                            cv::line(vis,
                                cv::Point((int)dispKp.points[b[0]].x, (int)dispKp.points[b[0]].y),
                                cv::Point((int)dispKp.points[b[1]].x, (int)dispKp.points[b[1]].y),
                                cv::Scalar(0, 255, 0), 2);
                        }
                    }
                    for (int i = 0; i < (int)dispKp.points.size(); ++i) {
                        cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                        cv::circle(vis, cv::Point((int)dispKp.points[i].x, (int)dispKp.points[i].y), 4, color, -1);
                    }
                }
                // 注意：OpenCV putText 只支持 ASCII（Hershey 矢量字体），中文会乱码，故用英文
                std::string stateText = std::string("State: ") + fsm.currentStateName() + "  (fist 1.2s to lock)";
                cv::putText(vis, stateText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                // 使用 WINDOW_NORMAL 允许调整窗口大小（AUTOSIZE 下 resizeWindow 无效）
                cv::namedWindow("hand-ctrl", cv::WINDOW_NORMAL);
                cv::imshow("hand-ctrl", vis);
                cv::resizeWindow("hand-ctrl", 960, 720);
                int key = cv::waitKey(10);  // 10ms 让 GTK 有时间处理窗口绘制
                if (key == 27) g_shouldExit.store(true);  // ESC 键退出
                // 点击窗口右上角 × 时：窗口被销毁，WND_PROP_VISIBLE 变为 0。
                // 注意：GTK/VMware 下窗口被遮挡/最小化/焦点变化时该值可能瞬时为 0，
                //       若立即退出会误判。故采用"连续 N 次不可见才退出"的防抖策略。
                static int invisibleCnt = 0;
                if (cv::getWindowProperty("hand-ctrl", cv::WND_PROP_VISIBLE) < 1) {
                    if (++invisibleCnt >= 5) {  // 约 0.5s 持续不可见才视为已关闭
                        std::printf("[main] 画面窗口已关闭，程序退出\n");
                        g_shouldExit.store(true);
                    }
                } else {
                    invisibleCnt = 0;  // 窗口可见，重置计数器
                }
                // 诊断：每 5 秒打印一次帧状态（仅调试，帮助定位黑屏原因）
                static int diagCnt = 0;
                if (++diagCnt % 125 == 0) {
                    std::printf("[main] 显示诊断: frameOk=%d visSize=%dx%d kp.valid=%d\n",
                                (int)frameOk, vis.cols, vis.rows, (int)dispKp.valid);
                }
            }
        }

        // ---- 取关键点 ----
        if (!kpQueue.pop(kp, 100)) {
            // 队列超时：仅每 10 次打印一次心跳（避免刷屏；-q 静默模式下不打印）
            ++waitLogCnt;
            if (waitLogCnt % 10 == 1 && !g_quiet.load()) {
                std::printf("[main] 等待关键点... (状态=%s)\n", fsm.currentStateName());
            }
            continue;  // continue 回到循环开头，仍会刷新显示 ✓
        }
        waitLogCnt = 0;

        auto now = std::chrono::steady_clock::now();
        double dtMs = std::chrono::duration<double, std::milli>(now - lastTime).count();
        lastTime = now;

        // FSM 推进
        GestureEvent e = fsm.handleFrame(kp, dtMs);
        ++fsmCnt;
        // 配置热加载：每 60 帧（约 1 秒）检测配置文件变更，实时生效阈值
        if (fsmCnt % 60 == 0) {
            if (fsm.reloadIfChanged()) {
                std::printf("[main] 配置已热加载，屏幕=%dx%d\n",
                            fsm.screenWidth(), fsm.screenHeight());
            }
            if (!g_quiet.load()) {
                const auto& w = kp.points.empty() ? HandPoint{} : kp.points[0];
                std::printf("[main] FSM#%d 状态=%s kp.valid=%d 手腕=(%.0f,%.0f)\n",
                            fsmCnt, fsm.currentStateName(), (int)kp.valid, w.x, w.y);
            }
        }

        // 更新共享最新关键点（供显示/截图共用）
        {
            std::lock_guard<std::mutex> lock(kpMutex);
            latestKp = kp;
        }

        // 保存截图：每 120 帧保存一张叠加关键点的帧到 /tmp（绕开 imshow，VMware 可靠）
        if (saveShot && fsmCnt % 120 == 0 && kp.valid) {
            cv::Mat vis;
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                if (!latestFrame.empty()) latestFrame.copyTo(vis);
            }
            if (!vis.empty()) {
                const int bones[][2] = {
                    {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
                    {5,9},{9,10},{10,11},{11,12},{9,13},{13,14},{14,15},{15,16},
                    {13,17},{17,18},{18,19},{19,20},{0,17}
                };
                for (auto& b : bones) {
                    if (b[0] < (int)kp.points.size() && b[1] < (int)kp.points.size()) {
                        cv::line(vis,
                            cv::Point((int)kp.points[b[0]].x, (int)kp.points[b[0]].y),
                            cv::Point((int)kp.points[b[1]].x, (int)kp.points[b[1]].y),
                            cv::Scalar(0, 255, 0), 2);
                    }
                }
                for (int i = 0; i < (int)kp.points.size(); ++i) {
                    cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                    cv::circle(vis, cv::Point((int)kp.points[i].x, (int)kp.points[i].y), 4, color, -1);
                }
                std::string stateText = std::string("State: ") + fsm.currentStateName();
                cv::putText(vis, stateText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                char path[128];
                std::snprintf(path, sizeof(path), "/tmp/hand_shot_%04d.jpg", fsmCnt);
                cv::imwrite(path, vis);
                std::printf("[main] 已保存截图: %s\n", path);
            }
        }

        // 事件映射到 uinput 输出（v2 单指方案）
        switch (e) {
            case GestureEvent::kFistHold:
                // 锁定/解锁切换：无 uinput 输出（仅状态切换）
                std::printf("[main] 状态切换: %s\n", fsm.currentStateName());
                // 安全释放左键：若此前在拖拽/按下中，必须释放避免鼠标卡住
                uinput.releaseLeft();
                // 锁定瞬间初始化虚拟鼠标位置（锁定后手指落在画面何处，指针从屏幕中心起算）
                virtualInit = false;
                break;

            case GestureEvent::kPointerMove:
            case GestureEvent::kDragMove:
                // 绝对定位移动（锁定定位 / 拖拽中跟手）：食指指尖坐标 → 屏幕坐标
                {
                    float tipX, tipY;
                    fsm.lastIndexTip(tipX, tipY);
                    if (tipX < 0 || tipY < 0) break;  // 尚未初始化
                    // 画面坐标 → 屏幕坐标等比映射
                    float targetX = tipX / imgW * scrW;
                    float targetY = tipY / imgH * scrH;
                    // 首帧校准：以当前虚拟位置为基准，避免首帧跳变
                    if (!virtualInit) {
                        virtualMouseX = targetX;
                        virtualMouseY = targetY;
                        virtualInit = true;
                    }
                    // 相对位移 = 目标 - 虚拟位置（uinput 仅支持相对位移）
                    int dx = static_cast<int>(targetX - virtualMouseX);
                    int dy = static_cast<int>(targetY - virtualMouseY);
                    if (dx != 0 || dy != 0) {
                        uinput.moveMouse(dx, dy);
                        virtualMouseX += dx;
                        virtualMouseY += dy;
                    }
                }
                break;

            case GestureEvent::kClick:
                uinput.clickLeft();  // 左键单击
                break;

            case GestureEvent::kDragStart:
                uinput.pressLeft();  // 拖拽开始：按住左键
                break;

            case GestureEvent::kDragEnd:
                uinput.releaseLeft();  // 拖拽结束：释放左键
                break;

            case GestureEvent::kNone:
            default:
                break;
        }

        if (e != GestureEvent::kNone) {
            ++eventCnt;
            // 持续类事件（定位移动/拖拽移动）每帧触发，逐帧打印会刷屏拖垮主循环
            const bool isContinuous =
                (e == GestureEvent::kPointerMove || e == GestureEvent::kDragMove);
            if (!isContinuous) {
                std::printf("[main] 事件触发: %s (状态=%s)\n",
                            eventToString(e), fsm.currentStateName());
            }
        }
    }

    // 10. 等待线程退出
    std::printf("[main] 等待子线程退出...\n");
    captureThread.join();
    inferThread.join();

    // 11. 资源释放（RAII）
    camera.release();
    // uinput 由析构自动销毁
    cv::destroyAllWindows();

    std::printf("[main] 程序退出，共触发 %d 个事件\n", eventCnt);
    return 0;
}

// 事件转字符串（供日志打印，v2 单指方案）
static const char* eventToString(GestureEvent e) {
    switch (e) {
        case GestureEvent::kNone:        return "无";
        case GestureEvent::kFistHold:    return "握拳长按(锁定/解锁)";
        case GestureEvent::kPointerMove: return "食指定位移动";
        case GestureEvent::kClick:       return "食指点击(左键)";
        case GestureEvent::kDragStart:   return "拖拽开始";
        case GestureEvent::kDragMove:    return "拖拽移动";
        case GestureEvent::kDragEnd:     return "拖拽结束";
    }
    return "未知";
}
