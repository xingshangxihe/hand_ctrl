// ============================================================================
// linux_uinput/uinput_manager.cpp
// 作用：UInputManager 类实现。
// 职责：
//   1. 创建虚拟键盘 + 虚拟鼠标两个 HID 设备（通过 /dev/uinput）
//   2. 封装按键发送、鼠标移动、左键单击/拖拽、右键单击底层接口
//   3. 利用析构函数 RAII 机制，自动销毁 uinput 设备，避免 /dev/uinput 卡死残留
//
// Linux uinput 工作流程：
//   1. open("/dev/uinput", O_WRONLY | O_NONBLOCK) 打开 uinput 设备节点
//   2. ioctl(UI_SET_EVBIT, EV_KEY/EV_REL/EV_SYN) 设置支持的事件类型
//   3. ioctl(UI_SET_KEYBIT, KEY_*) 注册具体按键 / ioctl(UI_SET_RELBIT, REL_X/REL_Y) 注册相对位移
//   4. write(uinput_user_dev) 写入设备描述（名称、bus、vendor、product、版本）
//   5. ioctl(UI_DEV_CREATE) 创建虚拟设备
//   6. write(input_event) 上报事件（按键按下/释放、鼠标位移、同步事件）
//   7. ioctl(UI_DEV_DESTROY) 销毁设备（RAII 析构时调用）
//
// 权限说明：访问 /dev/uinput 需要 root 权限或加入 input 用户组
//   - sudo 运行程序，或
//   - sudo usermod -aG input $USER 后重新登录
// ============================================================================

#include "linux_uinput/uinput_manager.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/input.h>     // KEY_* / BTN_* / REL_* / EV_* 宏定义
#include <linux/uinput.h>   // uinput 设备 ioctl 接口与结构体

namespace hand_ctrl {

// --------------------------- 构造 / 析构 ---------------------------
UInputManager::UInputManager() = default;

UInputManager::~UInputManager() {
    // RAII：对象销毁时自动释放虚拟键鼠设备，避免 /dev/uinput 卡死残留
    destroyDevices();
}

// --------------------------- 设备创建 ---------------------------
bool UInputManager::createDevices() {
    // 1. 打开 /dev/uinput 主设备节点（O_WRONLY | O_NONBLOCK）
    //    说明：鼠标和键盘都从同一节点创建，需分别 open 两次
    m_fdKeyboard = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (m_fdKeyboard < 0) {
        std::fprintf(stderr, "[UInput] 打开 /dev/uinput 失败（键盘）：%s\n", std::strerror(errno));
        std::fprintf(stderr, "[UInput] 请用 sudo 运行，或执行：sudo usermod -aG input $USER 后重新登录\n");
        return false;
    }
    m_fdMouse = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (m_fdMouse < 0) {
        std::fprintf(stderr, "[UInput] 打开 /dev/uinput 失败（鼠标）：%s\n", std::strerror(errno));
        ::close(m_fdKeyboard);
        m_fdKeyboard = -1;
        return false;
    }

    // 2. 配置虚拟键盘
    if (!setupKeyboard()) {
        std::fprintf(stderr, "[UInput] 虚拟键盘创建失败\n");
        destroyDevices();
        return false;
    }
    // 3. 配置虚拟鼠标
    if (!setupMouse()) {
        std::fprintf(stderr, "[UInput] 虚拟鼠标创建失败\n");
        destroyDevices();
        return false;
    }

    std::printf("[UInput] 虚拟键鼠双设备创建成功\n");
    return true;
}

// 配置虚拟键盘：注册常用按键 + 创建设备
bool UInputManager::setupKeyboard() {
    // 2a. 设置事件类型：EV_KEY（按键）、EV_SYN（同步）
    if (::ioctl(m_fdKeyboard, UI_SET_EVBIT, EV_KEY) < 0 ||
        ::ioctl(m_fdKeyboard, UI_SET_EVBIT, EV_SYN) < 0) {
        std::fprintf(stderr, "[UInput] 键盘 EV_BIT 设置失败：%s\n", std::strerror(errno));
        return false;
    }
    // 2b. 注册常用按键（覆盖手势映射所需的所有键码）
    //     说明：KEY_* 宏见 linux/input-event-codes.h
    const int keys[] = {
        KEY_ENTER, KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN,  // 导航类
        KEY_SPACE, KEY_BACKSPACE, KEY_TAB,                // 编辑类
        KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_LEFTALT,         // 修饰键
        KEY_ESC,                                           // ESC
        KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F,          // 字母（部分示例，可按需扩展）
        KEY_1, KEY_2, KEY_3, KEY_4, KEY_5
    };
    for (int k : keys) {
        if (::ioctl(m_fdKeyboard, UI_SET_KEYBIT, k) < 0) {
            std::fprintf(stderr, "[UInput] 键盘 KEYBIT(%d) 注册失败：%s\n", k, std::strerror(errno));
            return false;
        }
    }
    // 2c. 写入设备描述并创建
    struct uinput_user_dev dev{};
    std::snprintf(dev.name, UINPUT_MAX_NAME_SIZE, "hand-ctrl Virtual Keyboard");
    dev.id.bustype = BUS_USB;
    dev.id.vendor  = 0x1234;
    dev.id.product = 0x0001;
    dev.id.version = 1;
    if (::write(m_fdKeyboard, &dev, sizeof(dev)) < 0) {
        std::fprintf(stderr, "[UInput] 键盘设备描述写入失败：%s\n", std::strerror(errno));
        return false;
    }
    if (::ioctl(m_fdKeyboard, UI_DEV_CREATE) < 0) {
        std::fprintf(stderr, "[UInput] 键盘设备创建失败：%s\n", std::strerror(errno));
        return false;
    }
    return true;
}

// 配置虚拟鼠标：注册相对位移 + 左右键 + 创建设备
bool UInputManager::setupMouse() {
    // 3a. 设置事件类型：EV_KEY（按键）、EV_REL（相对位移）、EV_SYN
    if (::ioctl(m_fdMouse, UI_SET_EVBIT, EV_KEY) < 0 ||
        ::ioctl(m_fdMouse, UI_SET_EVBIT, EV_REL) < 0 ||
        ::ioctl(m_fdMouse, UI_SET_EVBIT, EV_SYN) < 0) {
        std::fprintf(stderr, "[UInput] 鼠标 EV_BIT 设置失败：%s\n", std::strerror(errno));
        return false;
    }
    // 3b. 注册鼠标按键（左、右、中键）
    const int buttons[] = {BTN_LEFT, BTN_RIGHT, BTN_MIDDLE};
    for (int b : buttons) {
        if (::ioctl(m_fdMouse, UI_SET_KEYBIT, b) < 0) {
            std::fprintf(stderr, "[UInput] 鼠标 KEYBIT(%d) 注册失败：%s\n", b, std::strerror(errno));
            return false;
        }
    }
    // 3c. 注册相对位移轴（X、Y）
    if (::ioctl(m_fdMouse, UI_SET_RELBIT, REL_X) < 0 ||
        ::ioctl(m_fdMouse, UI_SET_RELBIT, REL_Y) < 0) {
        std::fprintf(stderr, "[UInput] 鼠标 REL_BIT 设置失败：%s\n", std::strerror(errno));
        return false;
    }
    // 3d. 写入设备描述并创建
    struct uinput_user_dev dev{};
    std::snprintf(dev.name, UINPUT_MAX_NAME_SIZE, "hand-ctrl Virtual Mouse");
    dev.id.bustype = BUS_USB;
    dev.id.vendor  = 0x1234;
    dev.id.product = 0x0002;
    dev.id.version = 1;
    if (::write(m_fdMouse, &dev, sizeof(dev)) < 0) {
        std::fprintf(stderr, "[UInput] 鼠标设备描述写入失败：%s\n", std::strerror(errno));
        return false;
    }
    if (::ioctl(m_fdMouse, UI_DEV_CREATE) < 0) {
        std::fprintf(stderr, "[UInput] 鼠标设备创建失败：%s\n", std::strerror(errno));
        return false;
    }
    return true;
}

// --------------------------- 事件上报工具 ---------------------------
// 上报一个 input_event 到指定 fd（按键/位移/同步）
static void emitEvent(int fd, __u16 type, __u16 code, __s32 value) {
    struct input_event ev{};
    ev.type  = type;
    ev.code  = code;
    ev.value = value;
    ::write(fd, &ev, sizeof(ev));
}

// 上报同步事件（每完成一组按键/位移后必须发送，否则事件不生效）
static void emitSync(int fd) {
    emitEvent(fd, EV_SYN, SYN_REPORT, 0);
}

// --------------------------- 键盘接口 ---------------------------
void UInputManager::sendKey(uint16_t keyCode) {
    if (m_fdKeyboard < 0) return;
    // 按下 + 释放 + 同步（一个完整的按键事件）
    emitEvent(m_fdKeyboard, EV_KEY, keyCode, 1);  // 按下
    emitEvent(m_fdKeyboard, EV_KEY, keyCode, 0);  // 释放
    emitSync(m_fdKeyboard);
}

// --------------------------- 鼠标接口 ---------------------------
void UInputManager::moveMouse(int dx, int dy) {
    if (m_fdMouse < 0) return;
    // 相对位移：REL_X/REL_Y + 同步
    emitEvent(m_fdMouse, EV_REL, REL_X, dx);
    emitEvent(m_fdMouse, EV_REL, REL_Y, dy);
    emitSync(m_fdMouse);
}

void UInputManager::clickLeft() {
    if (m_fdMouse < 0) return;
    // 左键单击：按下 → 同步 → 短暂保持 → 释放 → 同步
    // 关键：按下和释放必须各带一次 SYN 分隔，且中间保持 ~20ms。
    // 若按下/释放在同一 SYN 周期内发出，输入栈会认为"按住时间=0"，
    // 点击事件被丢弃（表现为单击无效，但拖拽 pressLeft 单独发送有效）。
    emitEvent(m_fdMouse, EV_KEY, BTN_LEFT, 1);
    emitSync(m_fdMouse);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 保持 20ms，模拟真实点击
    emitEvent(m_fdMouse, EV_KEY, BTN_LEFT, 0);
    emitSync(m_fdMouse);
}

void UInputManager::doubleClick() {
    if (m_fdMouse < 0) return;
    // 双击：两次单击，中间间隔 50ms（模拟真实双击节奏）
    for (int i = 0; i < 2; ++i) {
        emitEvent(m_fdMouse, EV_KEY, BTN_LEFT, 1);
        emitSync(m_fdMouse);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        emitEvent(m_fdMouse, EV_KEY, BTN_LEFT, 0);
        emitSync(m_fdMouse);
        if (i == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 两次之间间隔
        }
    }
}

void UInputManager::pressLeft() {
    if (m_fdMouse < 0) return;
    // 仅按下（供拖拽使用，由调用方在合适时机调用 releaseLeft）
    emitEvent(m_fdMouse, EV_KEY, BTN_LEFT, 1);
    emitSync(m_fdMouse);
}

void UInputManager::releaseLeft() {
    if (m_fdMouse < 0) return;
    // 仅释放（与 pressLeft 配对）
    emitEvent(m_fdMouse, EV_KEY, BTN_LEFT, 0);
    emitSync(m_fdMouse);
}

void UInputManager::clickRight() {
    if (m_fdMouse < 0) return;
    // 右键单击：按下 → 同步 → 短暂保持 → 释放 → 同步
    // 与 clickLeft 同理：按下/释放必须各带 SYN 分隔，否则输入栈认为
    // "按住时间=0"，点击事件被丢弃（表现为右键无效）。
    emitEvent(m_fdMouse, EV_KEY, BTN_RIGHT, 1);
    emitSync(m_fdMouse);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 保持 20ms，模拟真实点击
    emitEvent(m_fdMouse, EV_KEY, BTN_RIGHT, 0);
    emitSync(m_fdMouse);
}

// --------------------------- 设备销毁 ---------------------------
void UInputManager::destroyDevices() {
    // 依次销毁鼠标与键盘（顺序无强约束，但先销毁非活跃设备更稳妥）
    if (m_fdMouse >= 0) {
        ::ioctl(m_fdMouse, UI_DEV_DESTROY);  // 销毁虚拟鼠标
        ::close(m_fdMouse);
        m_fdMouse = -1;
        std::printf("[UInput] 虚拟鼠标已销毁\n");
    }
    if (m_fdKeyboard >= 0) {
        ::ioctl(m_fdKeyboard, UI_DEV_DESTROY);  // 销毁虚拟键盘
        ::close(m_fdKeyboard);
        m_fdKeyboard = -1;
        std::printf("[UInput] 虚拟键盘已销毁\n");
    }
}

// --------------------------- 状态查询 ---------------------------
bool UInputManager::isReady() const {
    // 两个设备均创建成功才算就绪
    return m_fdKeyboard >= 0 && m_fdMouse >= 0;
}

} // namespace hand_ctrl
