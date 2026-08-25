// ============================================================================
// gesture_fsm/kalman_filter.cpp
// 作用：KalmanFilter 类实现（4 维卡尔曼滤波）。
// 状态向量：[x, y, vx, vy]（位置 + 速度），用于对单个关键点坐标做时序平滑。
// 适用场景：MediaPipe Hands 输出的 21 个关键点存在抖动/微颤，每个点持有一个
//           KalmanFilter 实例分别平滑，可显著抑制误识别。
// 数学模型（常数速度模型）：
//   状态预测：x' = F·x，其中 F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
//   协方差预测：P' = F·P·F^T + Q
//   卡尔曼增益：K = P·H^T·(H·P·H^T + R)^-1，其中 H = [[1,0,0,0],[0,1,0,0]]
//   状态修正：x = x' + K·(z - H·x')
//   协方差修正：P = (I - K·H)·P
// ============================================================================

#include "gesture_fsm/kalman_filter.h"

#include <cmath>

namespace hand_ctrl {

KalmanFilter::KalmanFilter() {
    // 默认初始化：4x4 状态向量清零，4x4 协方差矩阵设为单位矩阵
    m_x.setZero();        // 状态向量 [x,y,vx,vy]
    m_P.setIdentity();    // 协方差矩阵（先验不确定性较大）
}

KalmanFilter::~KalmanFilter() = default;

void KalmanFilter::init(float processNoise, float measureNoise) {
    // 记录噪声参数
    m_processNoise = processNoise;
    m_measureNoise = measureNoise;
    m_initialized = true;

    // 重置状态向量与协方差矩阵
    m_x.setZero();
    m_P.setIdentity();

    // 初始化过程噪声矩阵 Q（4x4 对角阵，对角值为 processNoise）
    // 说明：Q 反映状态转移的不确定性，越大越信任测量值
    m_Q = Eigen::Matrix4f::Identity() * processNoise;

    // 初始化测量噪声矩阵 R（2x2 对角阵，对角值为 measureNoise）
    // 说明：R 反映测量值的不确定性，越大越信任预测值
    m_R = Eigen::Matrix2f::Identity() * measureNoise;
}

void KalmanFilter::update(float mx, float my, float& ox, float& oy) {
    // 首次调用：直接以测量值作为初始状态（速度设为 0）
    if (!m_initialized) {
        init(1.0f, 4.0f);  // 默认参数（若用户未显式 init）
    }
    if (m_firstFrame) {
        m_x(0) = mx;
        m_x(1) = my;
        m_x(2) = 0.f;  // 初始速度未知，设为 0
        m_x(3) = 0.f;
        m_firstFrame = false;
        ox = mx;
        oy = my;
        return;
    }

    // 1. 状态预测：x' = F·x（dt=1，常数速度模型）
    //    说明：本实现按帧间 dt=1 处理（30fps 下足够稳定）；
    //         若需精确 dt，可扩展 update 接口接收时间间隔
    m_x(0) += m_x(2);  // x' = x + vx*dt
    m_x(1) += m_x(3);  // y' = y + vy*dt

    // 2. 协方差预测：P' = F·P·F^T + Q
    //    F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]，F·P 等价于把 P 的第 3/4 行加到第 1/2 行
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F(0, 2) = 1.f;
    F(1, 3) = 1.f;
    m_P = F * m_P * F.transpose() + m_Q;

    // 3. 计算卡尔曼增益：K = P·H^T·(H·P·H^T + R)^-1
    //    H = [[1,0,0,0],[0,1,0,0]]（只观测位置）
    Eigen::Matrix<float, 2, 4> H = Eigen::Matrix<float, 2, 4>::Zero();
    H(0, 0) = 1.f;
    H(1, 1) = 1.f;
    Eigen::Matrix2f S = H * m_P * H.transpose() + m_R;
    Eigen::Matrix<float, 4, 2> K = m_P * H.transpose() * S.inverse();

    // 4. 状态修正：x = x' + K·(z - H·x')
    Eigen::Vector2f z(mx, my);
    Eigen::Vector2f innovation = z - H * m_x;
    m_x = m_x + K * innovation;

    // 5. 协方差修正：P = (I - K·H)·P
    m_P = (Eigen::Matrix4f::Identity() - K * H) * m_P;

    // 6. 输出平滑后的坐标
    ox = m_x(0);
    oy = m_x(1);
}

void KalmanFilter::reset() {
    // 重置状态：用于手丢失后重新检测时清空历史
    m_firstFrame = true;
    m_x.setZero();
    m_P.setIdentity();
}

} // namespace hand_ctrl
