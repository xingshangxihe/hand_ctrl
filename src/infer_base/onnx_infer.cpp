// ============================================================================
// infer_base/onnx_infer.cpp
// 作用：ONNXInfer 类实现（阶段3）。
// 职责：MediaPipe Hands 双模型推理链路：
//   palm_detection（192x192）→ 检测手掌框 → 裁剪放大 → hand_landmark（224x224）
//   → 输出 21 个手部关键点（映射回原图坐标）
//
// 实现要点（ONNX Runtime C++ API）：
//   - 使用 Ort::Session 加载 .onnx 模型
//   - 输入为 NCHW 格式 float 张量，RGB 通道，像素归一化到 [0,1]
//   - 通过 Run() 一次性传入输入张量、取回输出张量
//   - 本文件是唯一允许出现 ONNX Runtime 底层 API 的地方（面向接口约束）
// ============================================================================

#include "infer_base/onnx_infer.h"

#include <cstdio>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// 引入 ONNX Runtime C++ API（仅在 .cpp 中可见，对上层完全隐藏）
#include <onnxruntime_cxx_api.h>

namespace hand_ctrl {

// --------------------------- 构造 / 析构 ---------------------------
ONNXInfer::ONNXInfer() {
    // 创建 ONNX Runtime 推理环境（CPU 版本，禁用日志刷屏）
    // ORT_LOGGING_LEVEL_WARNING: 仅输出警告及以上级别
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "hand_ctrl_onnx");
}

ONNXInfer::~ONNXInfer() {
    // unique_ptr 自动释放会话与环境资源（RAII）
}

// --------------------------- 模型加载 ---------------------------
// 检查文件是否存在（避免 ORT 内部报错不明确）
static bool fileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

bool ONNXInfer::loadModels(const ONNXModelPaths& paths) {
    // 1. 校验关键点模型（必须存在）
    if (!fileExists(paths.landmarkModel)) {
        std::fprintf(stderr, "[ONNXInfer] 关键点模型缺失！请检查: %s\n", paths.landmarkModel.c_str());
        return false;
    }

    // 2. 加载关键点模型（必须成功）
    m_landmarkSession = std::make_unique<Ort::Session>(*m_env, paths.landmarkModel.c_str(), sessionOptionsInit());
    std::printf("[ONNXInfer] 关键点模型加载成功: %s\n", paths.landmarkModel.c_str());

    // 3. 手掌检测模型（可选，缺失时启用单模型模式）
    //    说明：tflite2onnx 对 palm_detection 模型存在 layout 转换 bug，
    //          若转换失败可只加载 landmark 模型，推理时用图像中央区域代替 palm 检测框。
    if (fileExists(paths.palmModel)) {
        try {
            m_palmSession = std::make_unique<Ort::Session>(*m_env, paths.palmModel.c_str(), sessionOptionsInit());
            std::printf("[ONNXInfer] 手掌检测模型加载成功: %s\n", paths.palmModel.c_str());
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[ONNXInfer] 手掌检测模型加载失败（降级为单模型模式）：%s\n", e.what());
            m_palmSession.reset();
        }
    } else {
        std::printf("[ONNXInfer] 手掌检测模型缺失，启用单模型模式（推理时用图像中央区域）。\n");
    }

    m_modelsLoaded = true;
    return true;
}

// 创建会话选项的辅助函数（双线程、全优化）
Ort::SessionOptions ONNXInfer::sessionOptionsInit() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

bool ONNXInfer::loadModel(const ModelPaths& model) {
    // 基类接口实现：映射到 ONNXModelPaths 后加载
    ONNXModelPaths paths;
    paths.palmModel     = model.paramPath;
    paths.landmarkModel = model.landmarkParamPath;
    return loadModels(paths);
}

// --------------------------- 推理 ---------------------------
bool ONNXInfer::infer(const cv::Mat& image, HandKeypoints& out) {
    // 前置检查：关键点模型必须已加载（手掌检测模型可选，缺失时启用单模型模式）
    if (!m_modelsLoaded || !m_landmarkSession) {
        std::fprintf(stderr, "[ONNXInfer] 关键点模型未加载，无法推理。请先调用 loadModels()。\n");
        out = HandKeypoints{};
        return false;
    }
    if (image.empty()) {
        std::fprintf(stderr, "[ONNXInfer] 输入图像为空，无法推理。\n");
        out = HandKeypoints{};
        return false;
    }

    try {
        // 公共资源：内存信息与分配器（双模型/单模型模式共用）
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;

        // ==================== step1: 确定手掌裁剪区域 ====================
        // 两种模式：
        //   A) 双模型模式：palm 检测框 + 外扩 30%（精度高，需 palm 模型）
        //   B) 单模型模式：palm 缺失时用图像中央 60% 区域（精度略降，开发验收够用）
        int cropX = 0, cropY = 0, cropW = 0, cropH = 0;
        float bestScore = 1.0f;  // 单模型模式下置信度记为 1.0

        if (m_palmSession) {
            // ---------- 双模型模式：执行手掌检测 ----------
            // 模型：PINTO palm_detection_full_inf_post_192x192.onnx
            // 输入：[1,3,192,192]（保持纵横比 resize + pad 成正方形，归一化 [0,1]，RGB）
            // 输出：[N,8]（单输出张量）pdscore, box_x, box_y, box_size, kp0x,kp0y,kp2x,kp2y
            // 坐标含义：box_x/box_y/box_size 为"正方形填充图空间"的归一化坐标 [0,1]

            // 1.1 预处理（对照 PINTO keep_aspect_resize_and_pad）：
            //     长边为基准等比缩放，短边 pad 到正方形，再整体 resize 到 192x192
            const int ih = image.rows, iw = image.cols;
            const float scale = static_cast<float>(kPalmInputSize) / std::max(ih, iw);
            const int rh = static_cast<int>(ih * scale);
            const int rw = static_cast<int>(iw * scale);
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(rw, rh));
            // 创建 192x192 黑底图，resize 图居中放置
            cv::Mat padded = cv::Mat::zeros(kPalmInputSize, kPalmInputSize, CV_8UC3);
            const int startH = (kPalmInputSize - rh) / 2;
            const int startW = (kPalmInputSize - rw) / 2;
            resized.copyTo(padded(cv::Rect(startW, startH, rw, rh)));
            // BGR→RGB + 归一化 [0,1] + NCHW
            cv::Mat rgb;
            cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
            std::vector<float> palmData(3 * kPalmInputSize * kPalmInputSize);
            const unsigned char* src = rgb.data;
            for (int c = 0; c < 3; ++c) {
                float* dst = palmData.data() + c * kPalmInputSize * kPalmInputSize;
                for (int i = 0; i < kPalmInputSize * kPalmInputSize; ++i) {
                    dst[i] = src[i * 3 + c] / 255.0f;
                }
            }
            // 记录 pad 半尺寸（用于坐标映射回原图）
            const int padHalfH = startH;  // 上下 pad
            const int padHalfW = startW;  // 左右 pad

            // 1.2 构建输入张量并推理
            std::vector<int64_t> palmShape = {1, 3, kPalmInputSize, kPalmInputSize};
            Ort::Value palmTensor = Ort::Value::CreateTensor<float>(memInfo, palmData.data(),
                                                                    palmData.size(),
                                                                    palmShape.data(), palmShape.size());
            std::vector<Ort::AllocatedStringPtr> palmInNamesAlloc, palmOutNamesAlloc;
            std::vector<const char*> palmInNames, palmOutNames;
            for (size_t i = 0; i < m_palmSession->GetInputCount(); ++i) {
                palmInNamesAlloc.push_back(m_palmSession->GetInputNameAllocated(i, allocator));
                palmInNames.push_back(palmInNamesAlloc.back().get());
            }
            for (size_t i = 0; i < m_palmSession->GetOutputCount(); ++i) {
                palmOutNamesAlloc.push_back(m_palmSession->GetOutputNameAllocated(i, allocator));
                palmOutNames.push_back(palmOutNamesAlloc.back().get());
            }
            std::vector<Ort::Value> palmOutputs = m_palmSession->Run(
                Ort::RunOptions{nullptr}, palmInNames.data(), &palmTensor, 1,
                palmOutNames.data(), palmOutNames.size());

            // 1.3 解析单输出 [N,8]
            if (palmOutputs.size() < 1) {
                std::fprintf(stderr, "[ONNXInfer] 手掌检测无输出。\n");
                out = HandKeypoints{};
                return false;
            }
            const float* palmOut = palmOutputs[0].GetTensorData<float>();
            auto palmInfo = palmOutputs[0].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> palmShapeOut = palmInfo.GetShape();
            // 计算行数 N 与每行元素数（应为 8）
            int64_t total = 1;
            for (int64_t d : palmShapeOut) total *= d;
            int64_t stride = palmShapeOut.empty() ? 8 : palmShapeOut.back();
            int64_t numBoxes = total / stride;

            // 1.4 找分数最高的手掌（分数过滤：真手通常 >0.9，脸部/耳朵偶发 0.6~0.7）
            //     阈值 0.7：从源头减少脸部/耳朵误检，真手不受影响
            int bestIndex = -1;
            bestScore = 0.0f;
            for (int64_t i = 0; i < numBoxes; ++i) {
                float s = palmOut[i * stride + 0];
                if (s > bestScore) {
                    bestScore = s;
                    bestIndex = static_cast<int>(i);
                }
            }
            if (bestIndex < 0 || bestScore < 0.7f) {
                // 未检测到手掌：视为无手，脸部/耳朵低分误检被此过滤
                out = HandKeypoints{};
                out.valid = false;
                return true;
            }

            // 1.5 解析手掌框（对照 PINTO __postprocess）
            const float* box = palmOut + bestIndex * stride;
            const float pd_score = box[0];   // 分数
            const float box_x = box[1];      // 中心 x（正方形空间归一化）
            const float box_y = box[2];      // 中心 y
            const float box_size = box[3];   // 框尺寸
            const float kp0_x = box[4];      // 关键点0
            const float kp0_y = box[5];
            const float kp2_x = box[6];      // 关键点2
            const float kp2_y = box[7];

            // 旋转角：rotation = 0.5*pi - atan2(-kp02_y, kp02_x)
            const float kp02x = kp2_x - kp0_x;
            const float kp02y = kp2_y - kp0_y;
            float rotation = 0.5f * 3.14159265f - std::atan2(-kp02y, kp02x);
            // 归一化到 [-pi, pi]
            while (rotation > 3.14159265f) rotation -= 2 * 3.14159265f;
            while (rotation < -3.14159265f) rotation += 2 * 3.14159265f;

            // 手掌中心（正方形空间归一化坐标）
            const float sqn_rr_size = 2.9f * box_size;
            const float center_x = box_x + 0.5f * box_size * std::sin(rotation);
            const float center_y = box_y - 0.5f * box_size * std::cos(rotation);

            // 映射回原图像素坐标（对照 PINTO app.py 精确实现）
            //   PINTO 中 sqn_rr_center_x 基准 = 正方形边长(=640)，sqn_rr_center_y 基准 = 原图高度(=480)
            //   pad 在 192 输入空间为 padHalfH/padHalfW 像素，映射回原图需乘 (squareSize/192)
            const int squareSize = std::max(iw, ih);
            const float scaleToOrig = static_cast<float>(squareSize) / kPalmInputSize;
            float cxImg, cyImg;
            if (iw >= ih) {
                // 宽>=高：水平无pad，垂直有pad（pad 量映射到原图像素）
                cxImg = center_x * squareSize;
                cyImg = center_y * squareSize - padHalfH * scaleToOrig;
            } else {
                // 高>宽：水平有pad，垂直无pad
                cxImg = center_x * squareSize - padHalfW * scaleToOrig;
                cyImg = center_y * squareSize;
            }
            // 裁剪手掌区域（PINTO: 半宽=size/2*640，半高=size*wh_ratio/2*480，
            //   两者数值相等因为 wh_ratio*480=640）
            const float halfSize = sqn_rr_size * squareSize * 0.5f;
            cropX = std::max(0, static_cast<int>(cxImg - halfSize));
            cropY = std::max(0, static_cast<int>(cyImg - halfSize));
            cropW = std::min(iw, static_cast<int>(cxImg + halfSize)) - cropX;
            cropH = std::min(ih, static_cast<int>(cyImg + halfSize)) - cropY;
        } else {
            // ---------- 单模型模式：用图像中央 60% 区域代替 palm 检测框 ----------
            // 适用场景：palm_detection.onnx 转换失败时（tflite2onnx layout bug），
            //           仍可验证关键点推理链路。要求手部大致位于画面中央。
            cropW = static_cast<int>(image.cols * 0.6f);
            cropH = static_cast<int>(image.rows * 0.6f);
            cropX = (image.cols - cropW) / 2;
            cropY = (image.rows - cropH) / 2;
        }

        if (cropW <= 0 || cropH <= 0) {
            std::fprintf(stderr, "[ONNXInfer] 手掌裁剪区域无效。\n");
            out = HandKeypoints{};
            out.valid = false;
            return false;
        }
        cv::Mat palmCrop = image(cv::Rect(cropX, cropY, cropW, cropH));

        // 2.2 关键点模型预处理
        cv::Mat cropRgb, cropResized;
        cv::cvtColor(palmCrop, cropRgb, cv::COLOR_BGR2RGB);
        cv::resize(cropRgb, cropResized, cv::Size(kLandmarkInputSize, kLandmarkInputSize));
        std::vector<float> lmData(3 * kLandmarkInputSize * kLandmarkInputSize);
        const unsigned char* lsrc = cropResized.data;
        for (int c = 0; c < 3; ++c) {
            float* dst = lmData.data() + c * kLandmarkInputSize * kLandmarkInputSize;
            for (int i = 0; i < kLandmarkInputSize * kLandmarkInputSize; ++i) {
                dst[i] = lsrc[i * 3 + c] / 255.0f;
            }
        }

        // 2.3 关键点模型推理
        std::vector<int64_t> lmShape = {1, 3, kLandmarkInputSize, kLandmarkInputSize};
        Ort::Value lmTensor = Ort::Value::CreateTensor<float>(memInfo, lmData.data(),
                                                              lmData.size(),
                                                              lmShape.data(), lmShape.size());
        std::vector<Ort::AllocatedStringPtr> lmInNamesAlloc, lmOutNamesAlloc;
        std::vector<const char*> lmInNames, lmOutNames;
        for (size_t i = 0; i < m_landmarkSession->GetInputCount(); ++i) {
            lmInNamesAlloc.push_back(m_landmarkSession->GetInputNameAllocated(i, allocator));
            lmInNames.push_back(lmInNamesAlloc.back().get());
        }
        for (size_t i = 0; i < m_landmarkSession->GetOutputCount(); ++i) {
            lmOutNamesAlloc.push_back(m_landmarkSession->GetOutputNameAllocated(i, allocator));
            lmOutNames.push_back(lmOutNamesAlloc.back().get());
        }
        std::vector<Ort::Value> lmOutputs = m_landmarkSession->Run(
            Ort::RunOptions{nullptr}, lmInNames.data(), &lmTensor, 1,
            lmOutNames.data(), lmOutNames.size());

        if (lmOutputs.empty()) {
            std::fprintf(stderr, "[ONNXInfer] 关键点推理无输出。\n");
            out = HandKeypoints{};
            return false;
        }

        // 2.4 解析 21 个关键点
        const float* lmDataOut = lmOutputs[0].GetTensorData<float>();
        auto lmInfo = lmOutputs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> lmShapeOut = lmInfo.GetShape();
        size_t lmTotal = 1;
        for (int64_t d : lmShapeOut) lmTotal *= static_cast<size_t>(d);
        if (lmTotal < static_cast<size_t>(kHandKeypointCount * 3)) {
            std::fprintf(stderr, "[ONNXInfer] 关键点输出数量异常（%zu）。\n", lmTotal);
            out = HandKeypoints{};
            return false;
        }

        // 2.5 读取手部置信度（输出1，Identity_1 = hand_scores）用于过滤非手误判
        //     模型输出结构（对照 PINTO hand_landmark.py 确认）：
        //       out0: Identity   [1,63]   21 关键点(x,y,z)  = xyz_x21s
        //       out1: Identity_1 [1,1]    手部置信度        = hand_scores  ← 过滤用
        //       out2: Identity_2 [1,1]    左右手分类        = left_hand_0_or_right_hand_1s
        //       out3: Identity_3 [1,63]   3D 世界坐标
        //     注意：此前误用 out2（左右手分类）导致脸部过滤失效，现改为 out1。
        float lmScore = bestScore;  // 默认用 palm 检测分数
        if (lmOutputs.size() >= 2) {
            const float* scoreData = lmOutputs[1].GetTensorData<float>();
            auto scoreInfo = lmOutputs[1].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> sShape = scoreInfo.GetShape();
            size_t sCount = 1;
            for (int64_t d : sShape) sCount *= static_cast<size_t>(d);
            if (sCount >= 1) {
                lmScore = scoreData[0];
                // 诊断日志（每 100 次打印一次，观察脸部/手部分数差异）
                static int scoreLogCnt = 0;
                if (++scoreLogCnt % 100 == 1) {
                    std::printf("[ONNXInfer] landmark 置信度=%.3f\n", lmScore);
                }
            }
        }

        // ==================== step3: 手部置信度过滤 ====================
        // palm 偶发把脸部/耳朵判为手掌（bestScore>0.7）时，landmark 输出置信度低
        // （真手 >0.9，脸部/耳朵通常 <0.4）。阈值 0.5 与 FSM 一致：低于则视为无手，
        // 返回 valid=false，显示层不再画出误检骨架。
        if (lmScore < 0.5f) {
            out = HandKeypoints{};
            out.valid = false;
            out.confidence = lmScore;
            return true;
        }

        // ==================== step4: 坐标映射回原图 ====================
        HandKeypoints result;
        result.valid = true;
        result.confidence = lmScore;   // 用手部置信度（输出2）
        result.points.resize(kHandKeypointCount);

        // 按 (x,y,z) 逐点读取（z 为深度，本阶段忽略），映射回原图坐标
        for (int i = 0; i < kHandKeypointCount; ++i) {
            float nx = lmDataOut[i * 3 + 0];
            float ny = lmDataOut[i * 3 + 1];
            result.points[i].x = nx / kLandmarkInputSize * cropW + cropX;
            result.points[i].y = ny / kLandmarkInputSize * cropH + cropY;
            result.points[i].score = lmScore;
        }

        out = std::move(result);
        return true;
    } catch (const Ort::Exception& e) {
        // 捕获 ONNX Runtime 异常（输入尺寸不符、内存等），输出中文日志
        std::fprintf(stderr, "[ONNXInfer] 推理异常：%s\n", e.what());
        out = HandKeypoints{};
        return false;
    }
}

} // namespace hand_ctrl