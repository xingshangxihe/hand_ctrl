#pragma once
// ============================================================================
// gesture_fsm/kalman_filter.h
// 作用：卡尔曼滤波工具类封装。
// 目标：对 21 个关键点的二维坐标分别做时序平滑，
//       抑制画面抖动、手部微颤造成的跳变误识别。
// 实现：4 维状态向量 [x, y, vx, vy]，常数速度模型，使用 Eigen 矩阵库运算。
// ============================================================================

#include <Eigen/Dense>

namespace hand_ctrl {

class KalmanFilter {
public:
    KalmanFilter();
    ~KalmanFilter();

    // 初始化滤波器
    // @param processNoise 过程噪声（越大越"信任"测量值，响应越快但更抖）
    // @param measureNoise 测量噪声（越大越"信任"预测值，越平滑但延迟更高）
    // 说明：阶段4 提供默认经验值，并支持外部按需调整
    void init(float processNoise, float measureNoise);

    // 输入一个新测量点，输出平滑后的坐标
    // @param mx  测量值 X
    // @param my  测量值 Y
    // @param ox  输出参数，平滑后 X
    // @param oy  输出参数，平滑后 Y
    void update(float mx, float my, float& ox, float& oy);

    // 重置滤波器状态（例如：切换到新手势或手丢失重检时调用）
    void reset();

private:
    // 状态向量 [x, y, vx, vy]：位置 + 速度
    Eigen::Vector4f m_x;

    // 协方差矩阵（4x4）：状态的不确定性
    Eigen::Matrix4f m_P;

    // 过程噪声矩阵 Q（4x4 对角）
    Eigen::Matrix4f m_Q;

    // 测量噪声矩阵 R（2x2 对角）
    Eigen::Matrix2f m_R;

    // 噪声参数缓存
    float m_processNoise = 1.0f;
    float m_measureNoise = 4.0f;

    // 初始化标志（未 init 时用默认参数）
    bool m_initialized = false;

    // 首帧标志（首帧直接以测量值作为初始状态）
    bool m_firstFrame = true;
};

} // namespace hand_ctrl