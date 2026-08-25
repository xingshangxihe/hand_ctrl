#pragma once
// ============================================================================
// infer_base/rknn_infer.h
// 作用：RKNN 推理预留空壳子类，继承自抽象基类 InferBase。
// 定位：仅为后续移植 RV1106 瑞芯微开发板预留框架。
//       当前只实现基类虚函数声明，不填充任何逻辑（阶段3 要求编译无告警）。
// 移植约定：将来接入时仅需填充本类实现，上层业务代码零修改。
// ============================================================================

#include "infer_base/infer_base.h"

namespace hand_ctrl {

class RKNNInfer : public InferBase {
public:
    RKNNInfer();
    ~RKNNInfer() override;

    // 预留：加载 RKNN 模型（后续移植时实现）
    bool loadModel(const ModelPaths& model) override;

    // 预留：执行 RKNN 推理（后续移植时实现）
    bool infer(const cv::Mat& image, HandKeypoints& out) override;
};

} // namespace hand_ctrl
