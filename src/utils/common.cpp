// ============================================================================
// utils/common.cpp
// 作用：通用工具函数实现。
// 说明：当前仅实现错误码 -> 中文描述 的映射函数，
//       供各模块异常捕获后输出统一清晰的中文日志（强制约束第8条）。
// ============================================================================

#include "utils/common.h"

namespace hand_ctrl {

std::string errorCodeToString(ErrorCode code) {
    // 错误码枚举 -> 可读中文描述，与 common.h 中的枚举一一对应
    switch (code) {
        case ErrorCode::kSuccess:            return "成功";
        case ErrorCode::kCameraOpenFailed:   return "摄像头打开失败";
        case ErrorCode::kCameraReadFailed:   return "摄像头读帧失败";
        case ErrorCode::kModelLoadFailed:    return "模型文件加载失败";
        case ErrorCode::kInferFailed:        return "推理执行失败";
        case ErrorCode::kUInputCreateFailed: return "uinput 虚拟设备创建失败";
        case ErrorCode::kMemoryAllocFailed:  return "内存分配异常";
        case ErrorCode::kConfigParseFailed:  return "配置文件解析失败";
    }
    // 兜底：防止未来新增错误码后遗漏映射
    return "未知错误";
}

} // namespace hand_ctrl
