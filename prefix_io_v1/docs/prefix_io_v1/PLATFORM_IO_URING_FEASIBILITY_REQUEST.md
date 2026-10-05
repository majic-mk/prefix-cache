# 平台处理材料：目标实验容器的原生 io_uring 必要能力

这是一份待用户转交的平台说明；本任务未发送消息或提交工单。无密码、token或密钥。

目标：connect.westc.seetacloud.com:24801；容器 autodl-container-67e44a9c3a-cb95ca9e。
已测内核：5.15.0-94-generic，x86_64；检查时间2026-09-26 16:35 UTC。

## 实际故障

未修改的作者py-kvcache需要原生io_uring_setup和io_uring_enter。最小setup(depth=2, flags=0)在标准C及作者LiburingRing均返回EPERM，失败发生于SSD读写和GPU运行之前。

当前UID0；io_uring_disabled=0；memlock不限；PID1和当前进程都有Seccomp=2/一个filter；AppArmor记录为docker-default(enforce)。本容器内没有发现Docker/containerd/Podman管理CLI或所检查的本地管理socket。这些事实支持平台侧策略限制的判断，但没有读取实际宿主机filter规则，尚不能唯一归因。

## 请平台核实的最小事项

- 当前目标容器是否能通过受支持的单容器安全配置，允许普通flags=0的原生io_uring_setup与io_uring_enter，以及作者随后实际需要的READ/WRITE/OPENAT/CLOSE操作？
- 如可支持，请给出具体生效方式、是否需要重启/重建、哪些目录保留、恢复步骤及影响范围。
- 如此实例类型不支持，请明确说明限制，并说明是否存在支持原生io_uring的环境类型；本材料不授权新购或迁移。
- 保留其他安全限制，不要求--privileged、全面seccomp=unconfined、修改驱动/CUDA/内核或宿主机全局参数。

当前作者文件未使用io_uring_register。交付C探针对register(-1)的调用仅为入口诊断，不是请求额外开放它。

## 无GPU最小复现

先使用随包的纯标准库 io_uring_setup_probe.py：

~~~bash
python3 -I -S io_uring_setup_probe.py
~~~

预期最低条件：setup返回有效fd并正常close，exit0。当前实际为exit2/EPERM。该成功仅是必要前提，不证明SSD路径已完整通过。

另有使用Linux系统头文件的独立C探针：

~~~bash
gcc -O2 -Wall -Wextra -Werror io_uring_syscall_probe.c -o io_uring_syscall_probe
./io_uring_syscall_probe
~~~

C程序只对有效setup做立即close；enter/register使用fd=-1，无I/O提交。平台如只允许setup/enter，register无效fd仍被拒绝不影响作者当前能力要求。

平台调整后，项目将继续原作者ring/mmap、实际enter/CQE及项目私有目录读写验证。不要以同步read/write、其他异步后端、文件存在或GPU Prefix命中替代原生SSD证据。
