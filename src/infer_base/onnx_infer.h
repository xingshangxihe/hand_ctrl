#pragma once
// ============================================================================
// infer_base/onnx_infer.h
// 作用：基于 ONNX Runtime 的推理实现子类，继承自抽象基类 InferBase。
// 职责：模型加载、图像预处理、推理执行、关键点后处理解析。
//
// 实现说明（MediaPipe Hands 双模型推理链路，与 NCNNInfer 逻辑一致）：
//   step1 手掌检测：palm_detection.onnx（输入 192x192）检测手掌包围盒
//   step2 手部关键点：裁剪手掌区域缩放 224x224，hand_landmark.onnx 输出 21 关键点
//
// 选型背景（2026-08 决策）：NCNN 新版移除了 onnx2ncnn 转换工具且 GitHub 不可达，
//   模型获取困难；改用 ONNX Runtime：
//   - 安装简单（apt 安装 libonnxruntime-dev / pip 安装 onnxruntime）
//   - MediaPipe Hands 的 ONNX 版模型源更丰富
//   - ONNX 是后续 RV1106 移植到 RKNN 的官方最优输入格式（RKNN-Toolkit2 原生支持 ONNX）
//
// 面向接口约束：上层业务仅依赖 InferBase，本类 ONNX Runtime 细节完全隔离在 .cpp 中。
// ============================================================================

#include <memory>
#include <string>
#include "infer_base/infer_base.h"

// 前置声明 ONNX Runtime 对象，将底层 API 完全隔离在 .cpp 中（面向接口约束）
namespace Ort {
class Env;
class Session;
class SessionOptions;
}

namespace hand_ctrl {

// ONNX 推理模型文件路径（双模型）
struct ONNXModelPaths {
    std::string palmModel;     // 手掌检测模型 *.onnx
    std::string landmarkModel; // 手部关键点模型 *.onnx
};

class ONNXInfer : public InferBase {
public:
    ONNXInfer();
    ~ONNXInfer() override;

    // 实现基类接口：加载 ONNX 模型
    // 说明：与 NCNNInfer 一致，model.paramPath 承载手掌检测 onnx，
    //       model.landmarkParamPath 承载关键点 onnx
    bool loadModel(const ModelPaths& model) override;

    // 便捷接口：直接传入双模型路径（推荐主程序调用）
    bool loadModels(const ONNXModelPaths& paths);

    // 实现基类接口：执行推理并解析 21 个手部关键点
    bool infer(const cv::Mat& image, HandKeypoints& out) override;

private:
    // ONNX Runtime 会话对象（pimpl 隔离）
    std::unique_ptr<Ort::Env>     m_env;             // 运行环境
    std::unique_ptr<Ort::Session> m_palmSession;     // 手掌检测会话
    std::unique_ptr<Ort::Session> m_landmarkSession; // 关键点会话

    // 双模型是否均已成功加载
    bool m_modelsLoaded = false;

    // 模型输入尺寸常量（与 MediaPipe Hands 官方一致）
    static constexpr int kPalmInputSize     = 192;   // 手掌检测输入边长
    static constexpr int kLandmarkInputSize = 224;   // 关键点模型输入边长

    // 创建会话选项的辅助函数（双线程、全优化）
    Ort::SessionOptions sessionOptionsInit();
};

} // namespace hand_ctrl