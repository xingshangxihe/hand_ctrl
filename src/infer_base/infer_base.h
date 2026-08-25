#pragma once
// ============================================================================
// infer_base/infer_base.h
// 作用：推理抽象基类。
// 核心约束（需求文档强制规则6）：
//   「推理层严格遵循面向接口编程思想：上层手势业务逻辑仅依赖抽象基类
//     InferBase，严禁直接耦合 NCNN/RKNN 底层 API」
// 因此：
//   - 上层（gesture_fsm / main）只 include 本头文件并持有 InferBase 指针；
//   - NCNN / RKNN 的实现细节全部隔离在各自子类中。
// 骨架阶段：定义纯虚接口与统一出参结构，子类实现在阶段3完成。
// ============================================================================

#include <string>
#include <opencv2/opencv.hpp>
#include "utils/common.h"

namespace hand_ctrl {

// 模型文件路径集合：统一约定推理子类通过该结构加载模型
// 说明：NCNN 需要 param + bin 两个文件；RKNN 为单一 rknn 文件（后续扩展）。
//       NCNN 双模型（MediaPipe Hands）时使用 landmarkParamPath/landmarkBinPath
//       承载第二个模型（手部关键点）；仅单模型框架可留空这两项。
struct ModelPaths {
    std::string paramPath;          // 网络结构文件（NCNN 的 .param / 推理框架对应结构文件）
    std::string binPath;            // 权重文件（NCNN 的 .bin，可留空表示单文件模型）

    // NCNN 双模型扩展：手部关键点模型的 param/bin（手掌检测模型的路径在 paramPath/binPath）
    std::string landmarkParamPath;  // 手部关键点模型结构文件（可留空表示单模型）
    std::string landmarkBinPath;    // 手部关键点模型权重文件（可留空表示单模型）
};

// InferBase：推理层统一抽象基类（纯虚接口）
class InferBase {
public:
    virtual ~InferBase() = default;

    // 加载模型文件
    // @param model 模型文件路径
    // @return true 加载成功，false 失败（子类内部输出中文错误日志）
    virtual bool loadModel(const ModelPaths& model) = 0;

    // 执行一次推理
    // @param image 输入图像（BGR 格式，cv::Mat）
    // @param out   输出参数，携带 21 个手部关键点坐标 + 置信度
    // @return true 推理成功，false 失败
    virtual bool infer(const cv::Mat& image, HandKeypoints& out) = 0;
};

} // namespace hand_ctrl
