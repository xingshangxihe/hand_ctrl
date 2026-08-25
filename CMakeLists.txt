# ============================================================================
# hand-ctrl 顶层 CMakeLists.txt
# 功能：启用 C++17 编译标准，自动查找并链接 OpenCV / ONNX Runtime 依赖，统一构建多模块
# 用法：
#   mkdir build && cd build
#   cmake ..              # 若 ONNX Runtime 无法自动找到：cmake .. -DONNXRUNTIME_DIR=<路径>
#   make -j$(nproc)
# 说明：本项目按"串行模块化"原则开发，每新增一个模块的源文件，仅需在下方
#       PROJECT_SOURCES / PROJECT_HEADERS 列表追加一行，无需改动其他配置。
#       推理框架已切换为 ONNX Runtime（原因见 README「推理框架选型」章节），
#       NCNN 实现保留为对比/移植参考，不再参与默认构建。
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(hand_ctrl CXX)

# --------------------------- 全局编译选项 ---------------------------
set(CMAKE_CXX_STANDARD 17)          # 启用 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 编译器必须支持 C++17，否则报错
set(CMAKE_CXX_EXTENSIONS OFF)       # 禁止 GNU 扩展语法，保证可移植性（利于后续 RV1106 交叉编译）

# 默认 Release 编译模式，开启编译优化；可通过 -DCMAKE_BUILD_TYPE=Debug 覆盖
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 开启常见编译警告，便于尽早发现代码问题
add_compile_options(-Wall -Wextra)

# --------------------------- 依赖查找：OpenCV ---------------------------
# Ubuntu/WSL 安装方式：sudo apt install libopencv-dev
# 源码编译高性能安装：见 README「环境部署」章节
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --------------------------- 依赖查找：ONNX Runtime（主推理框架） ---------------------------
# 安装方式（Ubuntu）：sudo apt install libonnxruntime-dev
# 备选（pip 安装 python 包后，库在 ~/.local/.../onnxruntime/capi，头文件单独下载）：
#   - 头文件：-DONNXRUNTIME_INCLUDE_HINT=<下载的头文件目录>（含 onnxruntime_cxx_api.h）
#   - 库文件：-DONNXRUNTIME_LIB_HINT=<capi目录>           （含 libonnxruntime.so 符号链接）
# 查找顺序：
#   1) find_path / find_library 标准路径查找
#   2) -DONNXRUNTIME_DIR=<路径> 手动指定安装目录（同时推导 include/lib）
#   3) -DONNXRUNTIME_INCLUDE_HINT 与 -DONNXRUNTIME_LIB_HINT 分别指定（pip + 手动头文件场景）
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_DIR}/include ${ONNXRUNTIME_INCLUDE_HINT}
        /usr/include/onnxruntime /usr/local/include/onnxruntime)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  HINTS ${ONNXRUNTIME_DIR}/lib ${ONNXRUNTIME_LIB_HINT}
        /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "未找到 ONNX Runtime 库！请先安装（sudo apt install libonnxruntime-dev），\n"
    "或通过 -DONNXRUNTIME_DIR=<onnxruntime安装路径> 指定其所在目录。")
endif()
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library:     ${ONNXRUNTIME_LIBRARY}")

# --------------------------- 依赖查找：线程库 ---------------------------
# 阶段6 多线程联调需要（std::thread），提前链接避免后续遗漏
find_package(Threads REQUIRED)

# --------------------------- 依赖查找：Eigen（卡尔曼滤波矩阵运算） ---------------------------
# Ubuntu 安装：sudo apt install libeigen3-dev
# Eigen 是 header-only，无需链接库
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core HINTS /usr/include/eigen3)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "未找到 Eigen3 库！请安装：sudo apt install libeigen3-dev")
  endif()
  message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIR}")
else()
  set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
  message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
endif()

# --------------------------- 源文件清单 ---------------------------
# 注意：每新增一个模块源文件，必须在此处登记，否则不会被编译进可执行文件
set(PROJECT_SOURCES
  src/main.cpp                     # 程序入口（阶段6 改为多线程调度器）
  src/utils/common.cpp             # 通用工具函数实现（错误码->中文描述）
  src/camera_capture/camera.cpp    # 摄像头采集模块
  src/infer_base/onnx_infer.cpp    # ONNX Runtime 推理实现子类（主推理框架）
  src/infer_base/rknn_infer.cpp    # RKNN 推理预留空壳子类
  src/gesture_fsm/kalman_filter.cpp# 卡尔曼滤波工具类
  src/gesture_fsm/gesture_fsm.cpp  # 手势有限状态机（核心模块）
  src/linux_uinput/uinput_manager.cpp # uinput 虚拟键鼠封装
)
set(PROJECT_HEADERS
  src/utils/common.h               # 通用常量 / 枚举 / 结构体收拢
  src/camera_capture/camera.h
  src/infer_base/infer_base.h
  src/infer_base/onnx_infer.h
  src/infer_base/ncnn_infer.h      # NCNN 实现保留为移植参考（不参与默认构建）
  src/infer_base/rknn_infer.h
  src/gesture_fsm/kalman_filter.h
  src/gesture_fsm/gesture_fsm.h
  src/linux_uinput/uinput_manager.h
)

# --------------------------- 可执行目标 ---------------------------
add_executable(hand_ctrl ${PROJECT_SOURCES} ${PROJECT_HEADERS})

# 头文件搜索路径：允许以模块前缀方式 include，例如 #include "camera_capture/camera.h"
target_include_directories(hand_ctrl PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

# 链接依赖库
target_link_libraries(hand_ctrl PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} Threads::Threads)

# --------------------------- 阶段2：摄像头模块独立测试目标 ---------------------------
# 作用：单独验收摄像头采集功能（打开设备 / 实时显示 / 断开异常退出），
#       不参与主程序 hand_ctrl 的业务流程。
# 用法：make camera_test && ./camera_test [设备索引]
add_executable(camera_test
  src/camera_capture/camera_test.cpp
  src/camera_capture/camera.cpp
  src/utils/common.cpp
)
target_include_directories(camera_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(camera_test PRIVATE ${OpenCV_LIBS})

# --------------------------- 阶段3：推理模块独立测试目标 ---------------------------
# 作用：验收 ONNX Runtime 双模型推理链路（手掌检测 + 关键点），
#       可用测试图片或摄像头帧验证 21 个关键点输出。
# 用法：make infer_test && ./infer_test [模型目录] [设备索引]
add_executable(infer_test
  src/infer_base/infer_test.cpp
  src/infer_base/onnx_infer.cpp
  src/camera_capture/camera.cpp
  src/utils/common.cpp
)
target_include_directories(infer_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(infer_test PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS})

# --------------------------- 阶段4：FSM 模块独立测试目标 ---------------------------
# 作用：验收卡尔曼滤波 + 手势有限状态机（6 种手势判定）
# 用法：make fsm_test && ./fsm_test [配置文件] [模型目录] [设备索引]
add_executable(fsm_test
  src/gesture_fsm/fsm_test.cpp
  src/gesture_fsm/kalman_filter.cpp
  src/gesture_fsm/gesture_fsm.cpp
  src/infer_base/onnx_infer.cpp
  src/camera_capture/camera.cpp
  src/utils/common.cpp
)
target_include_directories(fsm_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(fsm_test PRIVATE ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS})

# --------------------------- 阶段5：uinput 模块独立测试目标 ---------------------------
# 作用：验收虚拟键鼠双设备（创建/按键/移动/RAII 销毁）
# 用法：sudo make uinput_test && sudo ./uinput_test
add_executable(uinput_test
  src/linux_uinput/uinput_test.cpp
  src/linux_uinput/uinput_manager.cpp
)
target_include_directories(uinput_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(uinput_test PRIVATE Threads::Threads)
