#pragma once
// ============================================================================
// camera_capture/camera.h
// 作用：摄像头采集封装类，对外提供三个核心接口：
//       - open()      打开摄像头设备
//       - readFrame() 读取单帧图像（cv::Mat）
//       - release()   释放摄像头资源
// 设计要点：
//   - 内部采用 pimpl 模式，封装 Linux V4L2 直接抓 MJPG 字节 + OpenCV imdecode 解码
//   - 选用此方案的原因：本项目在 VMware 直通 UVC 摄像头场景下，MJPG 流存在
//     EOI 缺失等不标准情况；OpenCV 自带的 V4L2 MJPG 解码器处理失败产生花屏，
//     而 cv::imdecode（基于 libjpeg）宽容性更高、且能稳定解码。
//   - 对外接口保持极简（open / readFrame / release / isOpened），
//     业务代码无需关心底层是 V4L2 还是 VideoCapture。
// ============================================================================

#include <memory>
#include <opencv2/opencv.hpp>
#include "utils/common.h"

namespace hand_ctrl {

class Camera {
public:
    Camera();
    ~Camera();

    // 打开摄像头设备
    // @param deviceIndex 摄像头设备索引（默认 0，对应 /dev/video0）
    // @param width       采集宽度（像素）
    // @param height      采集高度（像素）
    // @param fps         采集帧率
    // @return true 打开成功，false 打开失败（内部已输出中文错误日志）
    bool open(int deviceIndex = 0,
              int width = kFrameWidth,
              int height = kFrameHeight,
              int fps = kTargetFps);

    // 读取单帧图像
    // @param frame 输出参数，成功时写入一帧 BGR 图像
    // @return true 读取成功，false 读取失败（如设备被拔出、解码失败）
    bool readFrame(cv::Mat& frame);

    // 释放摄像头资源（析构函数中也会自动调用，保证 RAII）
    void release();

    // 查询摄像头当前是否处于打开状态
    bool isOpened() const;

    // [诊断专用] 暴露底层 VideoCapture 引用。
    // 兼容旧测试程序：当前 V4L2 直读实现不再使用 OpenCV VideoCapture，
    // 返回静态 dummy 引用避免调用方崩溃。
    cv::VideoCapture& debugCapture();

private:
    // pimpl 前置声明：将 Linux V4L2 系统调用细节完全隐藏在 .cpp 中
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace hand_ctrl