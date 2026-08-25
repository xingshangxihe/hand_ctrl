// ============================================================================
// infer_base/infer_test.cpp
// 作用：推理模块【独立验收测试程序】（ONNX Runtime 版）。
// 行为：
//   - 加载 ONNX 双模型（palm_detection.onnx + hand_landmark.onnx）
//   - 从摄像头读取实时帧，执行推理，打印 21 个关键点坐标与置信度
// 用法：
//   ./infer_test [模型目录] [设备索引]
//   ./infer_test                          # 使用默认路径 models/ncnn_model 和摄像头0
//   ./infer_test [模型目录] -img 测试图片.jpg   # 用静态图片测试（无需摄像头）
// 说明：若模型文件尚未放入 models/ 下，程序会提示缺失并退出。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>
#include "infer_base/infer_base.h"
#include "infer_base/onnx_infer.h"
#include "camera_capture/camera.h"

using namespace hand_ctrl;

int main(int argc, char* argv[]) {
    // 参数解析：./infer_test [模型目录] [设备索引]  或  ./infer_test [模型目录] -img <图片>
    // 约定：第一个非 - 开头的参数是模型目录，第二个非 - 开头的参数是设备索引
    std::string modelDir = "models";
    std::string imagePath;
    int deviceIndex = 0;
    int positionalIdx = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-img") == 0 && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (argv[i][0] != '-') {
            if (positionalIdx == 0) {
                modelDir = argv[i];
            } else if (positionalIdx == 1) {
                deviceIndex = std::atoi(argv[i]);
            }
            ++positionalIdx;
        }
    }

    // 1. 构造推理器并加载模型
    ONNXInfer infer;
    ONNXModelPaths paths;
    paths.palmModel     = modelDir + "/palm_detection.onnx";
    paths.landmarkModel = modelDir + "/hand_landmark.onnx";
    if (!infer.loadModels(paths)) {
        std::printf("[infer_test] 模型加载失败，请确认模型文件位于: %s\n", modelDir.c_str());
        return 1;
    }

    // 2. 图像来源：命令行图片 或 摄像头
    cv::Mat frame;
    if (!imagePath.empty()) {
        frame = cv::imread(imagePath);
        if (frame.empty()) {
            std::printf("[infer_test] 测试图片打开失败: %s\n", imagePath.c_str());
            return 1;
        }
    } else {
        Camera camera;
        if (!camera.open(deviceIndex)) {
            std::printf("[infer_test] 摄像头打开失败。\n");
            return 1;
        }
        if (!camera.readFrame(frame)) {
            std::printf("[infer_test] 摄像头读帧失败。\n");
            return 1;
        }
    }

    // 3. 执行推理
    HandKeypoints kp;
    if (!infer.infer(frame, kp)) {
        std::printf("[infer_test] 推理失败。\n");
        return 1;
    }

    // 4. 打印结果
    if (!kp.valid) {
        std::printf("[infer_test] 未检测到手（置信度=%.3f）。\n", kp.confidence);
        return 0;
    }
    std::printf("[infer_test] 检测到手，置信度=%.3f，关键点=%zu\n",
                kp.confidence, kp.points.size());
    for (size_t i = 0; i < kp.points.size(); ++i) {
        std::printf("  [%2zu] (%.1f, %.1f)\n", i, kp.points[i].x, kp.points[i].y);
    }
    return 0;
}
