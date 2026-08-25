// ============================================================================
// infer_base/rknn_infer.cpp
// 作用：RKNNInfer 类实现（预留空壳）。
// 注意：本文件不允许输出任何编译告警；所有参数通过 (void) 显式忽略。
// 后续移植：仅需填充本文件的 loadModel / infer，上层业务代码零修改。
// ============================================================================

#include "infer_base/rknn_infer.h"

#include <cstdio>

namespace hand_ctrl {

RKNNInfer::RKNNInfer() {
    // 空壳阶段：无初始化逻辑
}

RKNNInfer::~RKNNInfer() {
    // 空壳阶段：无资源需要释放
}

bool RKNNInfer::loadModel(const ModelPaths& model) {
    // TODO(RV1106移植): 调用 RKNN-Toolkit2 的 rknn_init 加载 .rknn 模型
    (void)model;
    std::printf("[RKNNInfer] 预留空壳：loadModel 待移植实现\n");
    return false;
}

bool RKNNInfer::infer(const cv::Mat& image, HandKeypoints& out) {
    // TODO(RV1106移植): 调用 rknn_run / rknn_outputs_get 执行推理
    (void)image;
    out = HandKeypoints{};
    std::printf("[RKNNInfer] 预留空壳：infer 待移植实现\n");
    return false;
}

} // namespace hand_ctrl
