#pragma once
// ============================================================================
// utils/common.h
// 作用：本项目【通用常量 / 枚举 / 结构体】的唯一收拢头文件。
// 注意：遵循强制约束「代码杜绝魔法数字」——
//       - 状态枚举、坐标索引、错误码、固定常量必须统一在此定义；
//       - 业务代码中禁止出现散落的裸数字（0/1/640/480 等）。
// 演进：骨架阶段只放"全项目确定不变"的基础定义，手势相关枚举在阶段4补充。
// ============================================================================

#include <vector>
#include <cstdint>
#include <string>

namespace hand_ctrl {

// --------------------------- 全局固定常量 ---------------------------
// 摄像头固定采集分辨率与帧率（与需求文档一致：640x480 @ 30fps）
static constexpr int kFrameWidth   = 640;   // 图像宽度（像素）
static constexpr int kFrameHeight  = 480;   // 图像高度（像素）
static constexpr int kTargetFps    = 30;    // 目标帧率（帧/秒）

// MediaPipe Hands 输出 21 个手部关键点
static constexpr int kHandKeypointCount = 21;

// 程序版本号（v2.4：握拳点击 + 相对位移方案）
static constexpr const char* kProjectVersion = "2.4.0";

// --------------------------- 错误码枚举 ---------------------------
// 各模块异常情况的统一错误码，配合阶段约束「输出清晰中文日志」使用
enum class ErrorCode {
    kSuccess            = 0,   // 成功
    kCameraOpenFailed   = 1,   // 摄像头打开失败
    kCameraReadFailed   = 2,   // 摄像头读帧失败
    kModelLoadFailed    = 3,   // 模型文件加载失败
    kInferFailed        = 4,   // 推理执行失败
    kUInputCreateFailed = 5,   // uinput 虚拟设备创建失败
    kMemoryAllocFailed  = 6,   // 内存分配异常
    kConfigParseFailed  = 7,   // 配置文件解析失败
};

// --------------------------- 手部关键点结构 ---------------------------
// 单个关键点：归一化或像素坐标 + 置信度
// 说明：骨架阶段先定义数据结构，具体坐标归一化规则在阶段3确定
struct HandPoint {
    float x = 0.0f;      // X 坐标（阶段3 明确归一化/像素约定）
    float y = 0.0f;      // Y 坐标
    float score = 0.0f;  // 该点置信度 [0,1]
};

// 一次推理输出的全部手部关键点集合
struct HandKeypoints {
    std::vector<HandPoint> points; // 按 MediaPipe 固定索引顺序存放 21 个点
    float confidence = 0.0f;       // 整体检测置信度 [0,1]
    bool valid = false;            // 是否检测到手（未检测到为 false）
};

// --------------------------- 通用工具函数声明 ---------------------------
// 将错误码转换为可读的中文错误描述，供日志打印复用
std::string errorCodeToString(ErrorCode code);

} // namespace hand_ctrl
