#pragma once
// ============================================================================
// infer_base/ncnn_infer.h
// 作用：基于 NCNN 的推理实现子类，继承自抽象基类 InferBase。
// 职责：模型加载、图像预处理、推理执行、关键点后处理解析。
//
// 实现说明（MediaPipe Hands 双模型推理链路）：
//   step1 手掌检测：palm_detection 模型（输入 192x192）检测画面中的手掌，
//                   输出手部包围盒（矩形框）；
//   step2 手部关键点：根据检测框裁剪手掌区域，缩放到 224x224 后输入
//                     hand_landmark 模型，输出 21 个手部关键点坐标。
//   说明：本实现按"手掌检测框已在模型输出内解码"的常见转换版本编写，
//         若实际模型输出为原始 anchor 回归值，需在 ncnn_infer.cpp 的
//         后处理中补充 anchor 解码（代码中已标注 TODO 位置）。
//
// 面向接口约束：上层业务仅依赖 InferBase，本类 NCNN 细节完全隔离在 .cpp 中。
// ============================================================================

#include <memory>
#include <string>
#include "infer_base/infer_base.h"

// 前置声明 NCNN 网络对象，将 NCNN 细节完全隔离在 .cpp 中（面向接口约束）
namespace ncnn {
class Net;
}

namespace hand_ctrl {

// NCNN 推理模型文件路径（封装双模型四文件，避免调用方暴露两个 ModelPaths）
struct NCNNModelPaths {
    std::string palmParam;      // 手掌检测模型结构文件 *.param
    std::string palmBin;        // 手掌检测模型权重文件 *.bin
    std::string landmarkParam;  // 手部关键点模型结构文件 *.param
    std::string landmarkBin;    // 手部关键点模型权重文件 *.bin
};

class NCNNInfer : public InferBase {
public:
    NCNNInfer();
    ~NCNNInfer() override;

    // 实现基类接口：加载 NCNN 模型
    // 说明：当模型为双模型（palm + landmark）时，两个模型路径分别放在
    //       model.paramPath/binPath（手掌检测）与 landmarkParamPath/landmarkBinPath（关键点）
    bool loadModel(const ModelPaths& model) override;

    // 便捷接口：直接传入双模型路径（推荐主程序调用）
    bool loadModels(const NCNNModelPaths& paths);

    // 实现基类接口：执行 NCNN 推理并解析 21 个手部关键点
    bool infer(const cv::Mat& image, HandKeypoints& out) override;

private:
    // pimpl 方式持有两个 NCNN 网络对象，隐藏底层 API
    // 说明：palm 网络负责手掌检测，landmark 网络负责关键点回归
    std::unique_ptr<ncnn::Net> m_palmNet;
    std::unique_ptr<ncnn::Net> m_landmarkNet;

    // 双模型是否均已成功加载（infer 前校验）
    bool m_modelsLoaded = false;

    // 模型输入尺寸常量（与 MediaPipe Hands 官方一致）
    static constexpr int kPalmInputSize     = 192;   // 手掌检测输入边长（像素）
    static constexpr int kLandmarkInputSize = 224;   // 关键点模型输入边长（像素）
};

} // namespace hand_ctrl