// ============================================================================
// linux_uinput/uinput_test.cpp
// 作用：uinput 模块【独立验收测试程序】。
// 行为：
//   - 创建虚拟键鼠双设备
//   - 自动触发：空格键 → 鼠标移动 → 左键 → 右键，各间隔 1 秒
//   - 验证：全局生效（即使焦点不在终端也能触发）、RAII 析构无残留
// 用法：
//   sudo ./uinput_test           # 需 sudo（/dev/uinput 权限）
// 说明：程序运行后立即开始注入事件，10 秒后自动退出。
//       验收时建议把焦点放在文本编辑器/浏览器，观察按键/鼠标动作是否生效。
// ============================================================================

#include <cstdio>
#include <chrono>
#include <thread>
#include <csignal>
#include <linux/input.h>            // KEY_* / BTN_* / REL_* 键码定义（兼容所有内核版本）
#include "linux_uinput/uinput_manager.h"

using namespace hand_ctrl;

static volatile sig_atomic_t g_exit = 0;
static void onSig(int) { g_exit = 1; }

int main(int argc, char* argv[]) {
    std::signal(SIGINT, onSig);

    std::printf("[uinput_test] 创建虚拟键鼠设备...\n");
    std::printf("[uinput_test] 请把焦点切到文本编辑器或浏览器，5 秒后开始注入事件\n");

    UInputManager uinput;
    if (!uinput.createDevices()) {
        std::printf("[uinput_test] 设备创建失败，请用 sudo 运行\n");
        return 1;
    }

    // 5 秒预热，给用户切换焦点的时间
    for (int i = 5; i > 0; --i) {
        std::printf("[uinput_test] %d 秒后开始...\n", i);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (g_exit) break;
    }

    if (!g_exit) {
        // 1. 发送空格键（在文本框中应产生一个空格）
        std::printf("[uinput_test] 发送空格键\n");
        uinput.sendKey(KEY_SPACE);
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 2. 鼠标移动（相对位移 dx=50, dy=20，鼠标应向右下移动）
        std::printf("[uinput_test] 鼠标移动 (50, 20)\n");
        uinput.moveMouse(50, 20);
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 3. 左键单击
        std::printf("[uinput_test] 左键单击\n");
        uinput.clickLeft();
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 4. 右键单击（应弹出上下文菜单）
        std::printf("[uinput_test] 右键单击\n");
        uinput.clickRight();
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 5. 发送回车键
        std::printf("[uinput_test] 发送回车键\n");
        uinput.sendKey(KEY_ENTER);
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::printf("[uinput_test] 事件注入完成，2 秒后退出\n");
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // RAII 析构自动销毁设备，无残留
    std::printf("[uinput_test] 退出（析构自动销毁设备）\n");
    return 0;
}
