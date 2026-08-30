// ============================================================================
// camera_capture/camera.cpp
// 作用：Camera 类实现（Linux V4L2 直读 + OpenCV imdecode 解码 MJPG）。
//
// 实现背景与选型说明：
//   - 项目实机环境（VMware Ubuntu 22.04/24.04 + UVC 摄像头）下，
//     OpenCV 自带的 CAP_V4L2 后端 MJPG 解码器对摄像头的 MJPG 流处理不稳定，
//     表现为花屏且多帧内容冻结（libjpeg 严格模式遇到 EOI 缺失即解码失败）。
//   - cv::imdecode 基于 libjpeg-turbo，宽容性更高，可稳定解码；
//   - 因此本模块改为：直接 V4L2 抓 MJPG 原始字节，再交给 imdecode 解码。
//   - 对外接口保持极简，业务代码不感知底层细节。
//
// 注意：Camera 析构函数会自动 release，遵循 RAII 资源自动释放原则。
// ============================================================================

#include "camera_capture/camera.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <vector>

namespace hand_ctrl {

// --------------------------- Impl 类：V4L2 抓 MJPG 字节流 ---------------------------
struct Camera::Impl {
    int                       fd = -1;          // V4L2 设备文件描述符
    std::vector<void*>        bufStarts;        // mmap 起始地址列表
    std::vector<size_t>       bufLengths;       // 各 mmap 区域长度
    int                       m_width  = 0;     // 实际生效宽度
    int                       m_height = 0;     // 实际生效高度
    int                       m_fps    = 0;     // 实际生效帧率
    int                       m_deviceIndex = 0;// 当前打开的设备索引（仅日志使用）
    bool                      m_diagPrinted = false; // 缓冲容量诊断是否已打印（只打一次）
    int                       m_errLogCount = 0;     // 连续失败日志计数（限流用）

    // 初始化：open + 配置 MJPG 640x480 + 申请 mmap 缓冲 + 启动流
    bool init(int deviceIndex, int width, int height, int fps) {
        m_deviceIndex = deviceIndex;

        // 1. 打开设备节点 /dev/videoN
        //    注意：使用 O_NONBLOCK 非阻塞模式。配合 capture() 中的 poll 等待，
        //          DQBUF 永远不会永久阻塞——即使 poll 就绪后与 DQBUF 之间出现
        //          竞态（VMware 虚拟 USB 设备不稳定场景），也会立即返回 EAGAIN，
        //          由上层重试，从根本上避免"卡死"。
        char devPath[64];
        std::snprintf(devPath, sizeof(devPath), "/dev/video%d", deviceIndex);
        fd = ::open(devPath, O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            std::fprintf(stderr, "[Camera] 打开 %s 失败：%s\n", devPath, std::strerror(errno));
            return false;
        }

        // 2. 设置采集格式为 MJPG（强制优先，VMware 直通场景下最稳）
        v4l2_format fmt{};
        fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width       = static_cast<__u32>(width);
        fmt.fmt.pix.height      = static_cast<__u32>(height);
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;
        if (::ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            std::fprintf(stderr, "[Camera] 设置 MJPG 格式失败：%s\n", std::strerror(errno));
            cleanup();
            return false;
        }
        // 回读实际生效参数（部分摄像头会忽略请求值）
        m_width  = static_cast<int>(fmt.fmt.pix.width);
        m_height = static_cast<int>(fmt.fmt.pix.height);

        // 3. 设置帧率（允许失败：部分驱动不支持精确 fps 设置）
        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator   = 1;
        parm.parm.capture.timeperframe.denominator = static_cast<__u32>(fps);
        if (::ioctl(fd, VIDIOC_S_PARM, &parm) == 0) {
            m_fps = static_cast<int>(parm.parm.capture.timeperframe.denominator /
                                     (parm.parm.capture.timeperframe.numerator ? parm.parm.capture.timeperframe.numerator : 1));
        } else {
            m_fps = fps; // 设备不支持设置时，按请求值记录
        }

        // 4. 请求 mmap 缓冲区（8 个缓冲轮转；VMware 直通场景下部分缓冲可能空置，
        //    增大缓冲数可提高成功出帧的概率）
        v4l2_requestbuffers req{};
        req.count  = 8;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (::ioctl(fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
            std::fprintf(stderr, "[Camera] 请求缓冲区失败：%s\n", std::strerror(errno));
            cleanup();
            return false;
        }

        // 5. 对每个缓冲区执行 mmap
        bufStarts.resize(req.count, nullptr);
        bufLengths.resize(req.count, 0);
        for (unsigned i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (::ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
                std::fprintf(stderr, "[Camera] 查询缓冲区 #%u 失败：%s\n", i, std::strerror(errno));
                cleanup();
                return false;
            }
            bufLengths[i] = buf.length;
            bufStarts[i]  = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, buf.m.offset);
            if (bufStarts[i] == MAP_FAILED) {
                std::fprintf(stderr, "[Camera] mmap 缓冲区 #%u 失败：%s\n", i, std::strerror(errno));
                cleanup();
                return false;
            }
        }

        // 6. 将所有缓冲区入队（驱动将依次填充帧数据）
        for (unsigned i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (::ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
                std::fprintf(stderr, "[Camera] 入队缓冲区 #%u 失败：%s\n", i, std::strerror(errno));
                cleanup();
                return false;
            }
        }

        // 7. 启动视频流
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (::ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            std::fprintf(stderr, "[Camera] 启动视频流失败：%s\n", std::strerror(errno));
            cleanup();
            return false;
        }

        // 重开成功后重置错误日志计数，避免上一次损坏期间的失败日志
        // 把本次重新打开后的真实失败掩盖掉（否则限流后看不到诊断信息）
        m_errLogCount = 0;

        std::printf("[Camera] 摄像头打开成功：设备索引=%d，实际分辨率=%dx%d，帧率=%d\n",
                    m_deviceIndex, m_width, m_height, m_fps);
        return true;
    }

    // 抓取一帧：poll 等待就绪 → DQBUF 取出 MJPG 字节 → imdecode 解码 → 入队归还
    bool capture(cv::Mat& frame) {
        if (fd < 0) {
            std::fprintf(stderr, "[Camera] 摄像头未打开，无法读取帧。\n");
            return false;
        }

        // 1. poll 等待帧就绪（最多 1000ms；超时通常意味着设备断开/停流）
        //    说明：VMware USB 重定向下传输帧率极低，缩短超时让上层重试更频繁，
        //         提高画面刷新连贯性；设备断开时也能更快感知。
        struct pollfd pfd;
        pfd.fd     = fd;
        pfd.events = POLLIN;
        int pr = ::poll(&pfd, 1, 1000);
        if (pr <= 0) {
            if (pr == 0) {
                // 超时不算致命错误，静默返回由上层继续重试（避免刷屏）
                return false;
            }
            std::fprintf(stderr, "[Camera] poll 等待失败：%s\n", std::strerror(errno));
            return false;
        }

        // 2. 取出已填充的缓冲（非阻塞模式：poll 已确认就绪，但 poll 与 DQBUF 之间
        //    可能存在竞态返回 EAGAIN，属正常情况，静默返回由上层重试，不打印日志）
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (::ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno != EAGAIN) {
                std::fprintf(stderr, "[Camera] 读帧失败：%s（设备可能已拔出）\n", std::strerror(errno));
            }
            return false;
        }

        // 3. 空缓冲（bytesused==0）或索引异常：归还后返回失败，由上层重试。
        //    原因说明：VMware USB 直通场景下 UVC 传输不稳定，部分缓冲可能未填充，
        //    属正常现象，跳过即可，不打印日志避免刷屏。
        if (buf.index >= bufStarts.size() || buf.bytesused == 0) {
            ::ioctl(fd, VIDIOC_QBUF, &buf);
            return false;
        }

        // 4. 提取 MJPG 有效区间并解码
        //    背景：VMware USB 直通下 UVC 传输不稳定，帧可能包含损坏/多余字节。
        //    策略：定位最后一个 SOI(0xFFD8) 到 EOI(0xFFD9)，截取有效区间后解码；
        //          若找不到 EOI，则以 SOI 至缓冲末尾为区间（libjpeg 容忍 EOI 缺失）。
        const unsigned char* data   = static_cast<const unsigned char*>(bufStarts[buf.index]);
        const __u32         used    = buf.bytesused;
        // 保存 bytesused 副本：QBUF 归还后驱动可能清零 buf.bytesused，导致日志误导
        // 诊断辅助：打印首次帧的缓冲实际容量（若远小于正常 MJPG 帧大小，说明缓冲过小需扩容）
        if (!m_diagPrinted) {
            std::printf("[Camera] 诊断：缓冲容量=%zu 字节，首帧字节数=%u\n",
                        bufLengths[buf.index], used);
            m_diagPrinted = true;
        }

        // 4a. 从尾部向前找最后一个 EOI（0xFFD9），取其后位置作为区间结尾
        int endPos = static_cast<int>(used);
        for (int i = static_cast<int>(used) - 1; i >= 1; --i) {
            if (data[i] == 0xD9 && data[i - 1] == 0xFF) {
                endPos = i + 1; // 包含 EOI 两个字节
                break;
            }
        }
        // 4b. 从头部向后找最后一个 SOI（0xFFD8），取其位置作为区间起点
        int startPos = 0;
        for (int i = 0; i + 1 < endPos; ++i) {
            if (data[i] == 0xFF && data[i + 1] == 0xD8) {
                startPos = i;
            }
        }

        // 4c. 构造有效区间 Mat 并解码
        int validLen = endPos - startPos;
        if (validLen > 0) {
            cv::Mat raw(1, validLen, CV_8UC1, const_cast<unsigned char*>(data + startPos));
            cv::Mat decoded = cv::imdecode(raw, cv::IMREAD_COLOR);

            // 5. 立即将缓冲归还驱动（不等解码完成，保证缓冲轮转速度）
            ::ioctl(fd, VIDIOC_QBUF, &buf);

            if (!decoded.empty()) {
                decoded.copyTo(frame); // 隔离 mmap 缓冲引用
                return true;
            }
            // 解码失败：日志限流，只打印前 5 次，避免 VMware 丢帧场景刷屏
            if (m_errLogCount < 5) {
                std::fprintf(stderr, "[Camera] 读帧失败：MJPG 解码失败（区间=%d/%u 字节）。\n", validLen, used);
                ++m_errLogCount;
            }
        } else {
            // 区间无效（连 SOI 都找不到），归还缓冲
            ::ioctl(fd, VIDIOC_QBUF, &buf);
            if (m_errLogCount < 5) {
                std::fprintf(stderr, "[Camera] 读帧失败：MJPG 区间无效（SOI/EOI 定位失败，缓冲=%u 字节）。\n", used);
                ++m_errLogCount;
            }
        }
        return false;
    }

    // 释放资源：STREAMOFF → munmap → REQBUFS(0) → close
    void cleanup() {
        if (fd < 0) {
            return; // 重复调用或未打开
        }
        // 停止流
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ::ioctl(fd, VIDIOC_STREAMOFF, &type);
        // 解除 mmap
        for (size_t i = 0; i < bufStarts.size(); ++i) {
            if (bufStarts[i] && bufStarts[i] != MAP_FAILED) {
                ::munmap(bufStarts[i], bufLengths[i]);
            }
        }
        bufStarts.clear();
        bufLengths.clear();
        // 释放驱动分配的缓冲区
        v4l2_requestbuffers req{};
        req.count  = 0;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        ::ioctl(fd, VIDIOC_REQBUFS, &req);
        // 关闭 fd
        ::close(fd);
        fd = -1;
        std::printf("[Camera] 摄像头资源已释放。\n");
    }
};

// --------------------------- Camera 接口实现 ---------------------------
Camera::Camera() : m_impl(std::make_unique<Impl>()) {
    // 构造时创建 Impl；默认 fd=-1 表示未打开
}

Camera::~Camera() {
    // RAII：析构自动释放摄像头资源
    release();
}

bool Camera::open(int deviceIndex, int width, int height, int fps) {
    return m_impl->init(deviceIndex, width, height, fps);
}

bool Camera::readFrame(cv::Mat& frame) {
    if (!isOpened()) {
        std::fprintf(stderr, "[Camera] 摄像头未打开，无法读取帧。\n");
        return false;
    }
    return m_impl->capture(frame);
}

void Camera::release() {
    if (m_impl) {
        m_impl->cleanup();
    }
}

bool Camera::isOpened() const {
    return m_impl && m_impl->fd >= 0;
}

cv::VideoCapture& Camera::debugCapture() {
    // 当前 V4L2 直读实现不再使用 OpenCV VideoCapture，
    // 返回静态 dummy 引用以兼容测试程序调用。
    static cv::VideoCapture dummy;
    return dummy;
}

} // namespace hand_ctrl