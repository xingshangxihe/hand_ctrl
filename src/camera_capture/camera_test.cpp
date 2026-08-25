// ============================================================================
// camera_capture/camera_test.cpp
// 作用：摄像头模块【独立验收测试程序】（V4L2 直读 + imdecode 实现）。
// 用法：
//   ./camera_test [设备索引]              # 显示模式：实时显示画面
//   ./camera_test 0 640 480 -s           # 保存模式：保存帧到 /tmp，绕开 imshow
// 说明：
//   - 显示模式用于人工目测画面；
//   - 保存模式将帧直接写入磁盘，用于验证"采集链路完全流畅"（排除 VM 无 GPU
//     环境下 imshow 渲染卡顿的干扰）。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "camera_capture/camera.h"

using namespace hand_ctrl;

int main(int argc, char* argv[]) {
    // 参数解析：./camera_test [设备索引] [宽度] [高度] [-s]
    int deviceIndex = (argc > 1) ? std::atoi(argv[1]) : 0;
    int width  = (argc > 2) ? std::atoi(argv[2]) : 640;
    int height = (argc > 3) ? std::atoi(argv[3]) : 480;
    bool saveMode = false;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "-s") == 0) {
            saveMode = true;
        }
    }
    std::printf("[camera_test] 开始测试，设备索引=%d 分辨率=%dx%d 模式=%s\n",
                deviceIndex, width, height, saveMode ? "SAVE" : "DISPLAY");

    Camera camera;
    if (!camera.open(deviceIndex, width, height)) {
        std::printf("[camera_test] 摄像头打开失败，程序退出。\n");
        return 1;
    }

    cv::Mat frame;
    int okCount = 0, failCount = 0;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed > 20.0) {
            std::printf("[camera_test] 20秒诊断窗口结束。成功=%d 失败=%d\n", okCount, failCount);
            break;
        }
        if (camera.readFrame(frame)) {
            ++okCount;
            if (okCount % 10 == 0) {
                double fps = okCount / std::max(elapsed, 0.001);
                std::printf("[camera_test] 已读 %d 帧，当前成功帧率=%.1f fps\n", okCount, fps);
            }

            if (saveMode) {
                // 保存模式：前 30 帧保存到 /tmp/frames/，证明数据链路流畅且画面在变化
                if (okCount <= 30) {
                    char path[128];
                    std::snprintf(path, sizeof(path), "/tmp/frames/frame_%02d.jpg", okCount);
                    cv::imwrite(path, frame); // 直接写盘，绕开 imshow 渲染
                    std::printf("[camera_test] 已保存 %s\n", path);
                }
            } else {
                cv::imshow("hand-ctrl camera test (ESC/q to exit)", frame);
                int key = cv::waitKey(30);
                if (key == 27 || key == 'q') {
                    std::printf("[camera_test] 用户按退出键，正常退出。\n");
                    break;
                }
            }
        } else {
            ++failCount;
        }
    }

    camera.release();
    cv::destroyAllWindows();
    std::printf("[camera_test] 测试结束。成功=%d 失败=%d\n", okCount, failCount);
    return 0;
}
