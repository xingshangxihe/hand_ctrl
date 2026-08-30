#pragma once
// ============================================================================
// linux_uinput/uinput_manager.h
// 作用：uinput 虚拟键鼠封装类。
// 目标：
//   - 一次性创建虚拟键盘 + 虚拟鼠标两个 HID 设备
//   - 封装按键发送、鼠标移动、左键单击/拖拽、右键单击底层接口
//   - 利用 C++ 析构函数 RAII 机制，正常/异常退出自动销毁 uinput 设备，
//     避免 /dev/uinput 卡死残留
// 骨架阶段：仅完成类声明；完整实现（ioctl / write 系统调用）在阶段5完成。
// ============================================================================

#include <cstdint>

namespace hand_ctrl {

class UInputManager {
public:
    UInputManager();
    ~UInputManager();

    // 创建虚拟键盘 + 虚拟鼠标两个设备
    // @return true 双设备均创建成功，false 失败（内部输出中文错误日志）
    bool createDevices();

    // 按下并松开一个键盘按键
    // @param keyCode Linux 键码（KEY_* 宏，例如 KEY_ENTER / KEY_LEFT）
    void sendKey(uint16_t keyCode);

    // 相对移动鼠标
    // @param dx X 方向位移（可为负）
    // @param dy Y 方向位移（可为负）
    void moveMouse(int dx, int dy);

    // 鼠标左键单击
    void clickLeft();

    // 鼠标左键双击（两次快速单击，间隔 50ms）
    void doubleClick();

    // 鼠标左键拖拽（按下后移动，由调用方在合适时机调用 releaseLeft）
    void pressLeft();
    void releaseLeft();

    // 鼠标右键单击
    void clickRight();

    // 主动销毁设备（析构函数也会自动调用，本接口用于提前清理）
    void destroyDevices();

    // 查询设备是否已成功创建
    bool isReady() const;

private:
    // TODO(阶段5): 私有成员
    //   - 两个文件描述符：m_fdKeyboard / m_fdMouse（指向 /dev/uinput）
    int m_fdKeyboard = -1;
    int m_fdMouse    = -1;

    // 私有辅助：配置虚拟键盘（注册按键 + 创建设备）
    bool setupKeyboard();

    // 私有辅助：配置虚拟鼠标（注册按键+相对位移 + 创建设备）
    bool setupMouse();
};

} // namespace hand_ctrl
