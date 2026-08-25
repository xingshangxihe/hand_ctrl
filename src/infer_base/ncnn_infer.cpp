// ============================================================================
// infer_base/ncnn_infer.cpp
// 作用：NCNNInfer 类实现（阶段3 完整版）。
// 职责：MediaPipe Hands 双模型推理链路：
//   palm_detection（192x192）→ 检测手掌框 → 裁剪放大 → hand_landmark（224x224）
//   → 输出 21 个手部关键点（映射回原图坐标）
// 要点：
//   - 模型输入为 RGB 三通道图像，像素归一化到 [0,1]
//   - 后处理包含：手掌框解析、坐标尺度还原
//   - 所有失败场景输出清晰中文错误日志（强制约束第8条）
//   - 本文件是唯一允许出现 NCNN 底层 API 的地方（面向接口约束）
// ============================================================================

#include "infer_base/ncnn_infer.h"

#include <cstdio>
#include <fstream>

// 引入 NCNN 头文件（仅在 .cpp 中可见，对上层完全隐藏）
#include <ncnn/net.h>

namespace hand_ctrl {

// --------------------------- 构造 / 析构 ---------------------------
NCNNInfer::NCNNInfer() {
    // 骨架阶段：m_net 保持空指针，阶段3 在 loadModel 中创建
}

NCNNInfer::~NCNNInfer() {
    // pimpl 智能指针自动释放 NCNN 网络资源（RAII）
}

// --------------------------- 模型加载 ---------------------------
// 检查文件是否存在（避免 NCNN 内部报错不明确）
static bool fileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

bool NCNNInfer::loadModels(const NCNNModelPaths& paths) {
    // 1. 校验四个模型文件全部存在
    if (!fileExists(paths.palmParam) || !fileExists(paths.palmBin)) {
        std::fprintf(stderr, "[NCNNInfer] 手掌检测模型文件缺失！请检查:\n  %s\n  %s\n",
                     paths.palmParam.c_str(), paths.palmBin.c_str());
        return false;
    }
    if (!fileExists(paths.landmarkParam) || !fileExists(paths.landmarkBin)) {
        std::fprintf(stderr, "[NCNNInfer] 关键点模型文件缺失！请检查:\n  %s\n  %s\n",
                     paths.landmarkParam.c_str(), paths.landmarkBin.c_str());
        return false;
    }

    // 2. 创建手掌检测网络并加载
    m_palmNet = std::make_unique<ncnn::Net>();
    m_palmNet->opt.use_vulkan_compute = false; // 不启用 GPU，保持纯 CPU（VMware/板卡兼容）
    m_palmNet->opt.num_threads = 2;            // 双线程推理（与主程序采集线程解耦）
    if (m_palmNet->load_param(paths.palmParam.c_str()) != 0 ||
        m_palmNet->load_model(paths.palmBin.c_str()) != 0) {
        std::fprintf(stderr, "[NCNNInfer] 手掌检测模型加载失败！\n");
        return false;
    }

    // 3. 创建关键点网络并加载
    m_landmarkNet = std::make_unique<ncnn::Net>();
    m_landmarkNet->opt.use_vulkan_compute = false;
    m_landmarkNet->opt.num_threads = 2;
    if (m_landmarkNet->load_param(paths.landmarkParam.c_str()) != 0 ||
        m_landmarkNet->load_model(paths.landmarkBin.c_str()) != 0) {
        std::fprintf(stderr, "[NCNNInfer] 关键点模型加载失败！\n");
        return false;
    }

    m_modelsLoaded = true;
    std::printf("[NCNNInfer] 双模型加载成功（palm + landmark）。\n");
    return true;
}

bool NCNNInfer::loadModel(const ModelPaths& model) {
    // 基类接口实现：把 ModelPaths 映射到 NCNNModelPaths 后加载
    NCNNModelPaths paths;
    paths.palmParam     = model.paramPath;
    paths.palmBin       = model.binPath;
    paths.landmarkParam = model.landmarkParamPath;
    paths.landmarkBin   = model.landmarkBinPath;
    return loadModels(paths);
}

// --------------------------- 推理 ---------------------------
bool NCNNInfer::infer(const cv::Mat& image, HandKeypoints& out) {
    // 前置检查：模型必须已加载
    if (!m_modelsLoaded || !m_palmNet || !m_landmarkNet) {
        std::fprintf(stderr, "[NCNNInfer] 模型未加载，无法推理。请先调用 loadModels()。\n");
        out = HandKeypoints{};
        return false;
    }
    // 前置检查：输入图像非空
    if (image.empty()) {
        std::fprintf(stderr, "[NCNNInfer] 输入图像为空，无法推理。\n");
        out = HandKeypoints{};
        return false;
    }

    // ==================== step1: 手掌检测 ====================
    // 1.1 预处理：BGR→RGB，缩放至 192x192，归一化到 [0,1]
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    ncnn::Mat palmIn = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,
                                                     image.cols, image.rows,
                                                     kPalmInputSize, kPalmInputSize);
    // 归一化：均值0 方差1/255（即像素值 /255 到 [0,1]）
    const float palmMean[3]   = {0.f, 0.f, 0.f};
    const float palmNorm[3]   = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    palmIn.substract_mean_normalize(palmMean, palmNorm);

    // 1.2 推理：创建提取器，输入手掌图
    ncnn::Extractor palmEx = m_palmNet->create_extractor();
    palmEx.input("input", palmIn);
    // 手掌检测输出：score 分数 + box 检测框（不同转换版本输出 blob 名可能不同，
    // 若模型来自不同转换渠道，此处输出名可能需要调整）
    ncnn::Mat palmScore;   // 各 anchor 的置信度
    ncnn::Mat palmBox;     // 检测框（x,y,w,h 或 中心点+尺寸，依转换版本而定）
    if (palmEx.extract("score", palmScore) != 0 ||
        palmEx.extract("box", palmBox) != 0) {
        std::fprintf(stderr, "[NCNNInfer] 手掌检测推理失败（输出提取失败）。\n");
        out = HandKeypoints{};
        return false;
    }

    // 1.3 后处理：从输出中取置信度最高的检测框
    // 说明：模型输出通常含多个 anchor 候选，取 score 最大的那个框。
    //       TODO(模型版本适配)：若输出为原始回归需 anchor 解码，此处补解码逻辑。
    int bestIndex = -1;
    float bestScore = 0.f;
    const int palmN = palmScore.total(); // 候选框数量
    const float* scorePtr = palmScore;
    for (int i = 0; i < palmN; ++i) {
        float s = scorePtr[i];
        if (s > bestScore) {
            bestScore = s;
            bestIndex = i;
        }
    }
    if (bestIndex < 0 || bestScore < 0.5f) {
        // 未检测到手：返回无效结果（valid=false），上层据此处理
        out = HandKeypoints{};
        out.valid = false;
        return true;
    }

    // 1.4 解析检测框：按常见 NCNN 版输出格式 [x, y, w, h]（相对 192x192 图，归一化或像素值）
    //     TODO(模型版本适配)：实际布局需与转换出的 param 输出一致。
    const float* boxPtr = palmBox.row(bestIndex);
    const float bx = boxPtr[0]; // 框左上角 x
    const float by = boxPtr[1]; // 框左上角 y
    const float bw = boxPtr[2]; // 框宽
    const float bh = boxPtr[3]; // 框高

    // ==================== step2: 关键点推理 ====================
    // 2.1 从原图中裁剪手掌区域（按检测框，外扩 30% 保证包含整只手）
    //     坐标从 192x192 归一化空间映射回原图尺寸
    const float scaleX = static_cast<float>(image.cols) / kPalmInputSize;
    const float scaleY = static_cast<float>(image.rows) / kPalmInputSize;
    // 中心点 + 半宽高（外扩）
    const float cx = (bx + bw * 0.5f) * scaleX;
    const float cy = (by + bh * 0.5f) * scaleY;
    const float halfW = bw * 0.5f * scaleX * 1.3f; // 外扩30%
    const float halfH = bh * 0.5f * scaleY * 1.3f;

    // 裁剪边界裁剪到图像内
    int cropX = std::max(0, static_cast<int>(cx - halfW));
    int cropY = std::max(0, static_cast<int>(cy - halfH));
    int cropW = std::min(image.cols, static_cast<int>(cx + halfW)) - cropX;
    int cropH = std::min(image.rows, static_cast<int>(cy + halfH)) - cropY;
    if (cropW <= 0 || cropH <= 0) {
        std::fprintf(stderr, "[NCNNInfer] 手掌裁剪区域无效。\n");
        out = HandKeypoints{};
        out.valid = false;
        return false;
    }
    cv::Mat palmCrop = image(cv::Rect(cropX, cropY, cropW, cropH));

    // 2.2 预处理：裁剪图→RGB→224x224→归一化
    cv::Mat cropRgb;
    cv::cvtColor(palmCrop, cropRgb, cv::COLOR_BGR2RGB);
    ncnn::Mat lmIn = ncnn::Mat::from_pixels_resize(cropRgb.data, ncnn::Mat::PIXEL_RGB,
                                                   cropW, cropH,
                                                   kLandmarkInputSize, kLandmarkInputSize);
    const float lmMean[3] = {0.f, 0.f, 0.f};
    const float lmNorm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    lmIn.substract_mean_normalize(lmMean, lmNorm);

    // 2.3 关键点模型推理
    ncnn::Extractor lmEx = m_landmarkNet->create_extractor();
    lmEx.input("input", lmIn);
    // 输出：21 个关键点。常见格式为 [21, 3]（x,y,z，归一化到 224x224）
    ncnn::Mat lmOut;
    if (lmEx.extract("output", lmOut) != 0) {
        std::fprintf(stderr, "[NCNNInfer] 关键点推理失败（输出提取失败）。\n");
        out = HandKeypoints{};
        return false;
    }

    // ==================== step3: 关键点后处理 ====================
    // 3.1 解析 21 个点，映射回原图坐标
    //     lmOut 布局：[21, 3] 或 [63]，按每点 (x,y,z) 顺序
    const int pointCount = lmOut.total() / 3; // 期望 21
    if (pointCount < kHandKeypointCount) {
        std::fprintf(stderr, "[NCNNInfer] 关键点输出数量异常（%d < %d）。\n",
                     pointCount, kHandKeypointCount);
        out = HandKeypoints{};
        return false;
    }

    HandKeypoints result;
    result.valid = true;
    result.confidence = bestScore;
    result.points.resize(kHandKeypointCount);

    const float* lmPtr = lmOut;
    for (int i = 0; i < kHandKeypointCount; ++i) {
        // 归一化坐标 [0,1]（除以输入边长）→ 映射回裁剪图 → 加偏移回原图
        float nx = lmPtr[i * 3 + 0];
        float ny = lmPtr[i * 3 + 1];
        // 注意：若模型输出为像素值（0~224），此处不应除 224；默认按归一化处理，
        //       TODO(模型版本适配)：根据实际模型输出范围调整
        result.points[i].x = nx / kLandmarkInputSize * cropW + cropX;
        result.points[i].y = ny / kLandmarkInputSize * cropH + cropY;
        result.points[i].score = bestScore; // 关键点置信度暂用检测框分数
    }

    out = std::move(result);
    return true;
}

} // namespace hand_ctrl