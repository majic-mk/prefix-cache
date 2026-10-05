# 可交平台的 io_uring 最小诊断

现有历史 JSON 已证明本容器原生作者 LiburingRing 与 raw setup 返回 EPERM；此前未找到可独立携带的纯标准库脚本。新增 io_uring_setup_probe.py 仅用于平台资格诊断，不是替代 io_uring 执行器。

复制单个脚本到待核查的 Linux 64-bit 环境，使用普通现有身份运行一次：

```bash
python3 -I -S io_uring_setup_probe.py
```

仅调用一次 io_uring_setup(entries=2, flags=0)。如果返回有效 fd，立即关闭。没有 mmap、io_uring_enter、SQE/I/O 提交、文件数据写入、网络、Torch、模型、GPU、sysctl 修改、权限切换或 seccomp 修改。-I -S 避免项目/PYTHONPATH/site 包影响；脚本不依赖当前项目、NumPy、liburing Python 包或第三方库。

退出码：
- 0：setup 成功且 fd 已关闭。仅表示最小 syscall 可用。
- 2：setup syscall 返回失败；JSON 保留即时 errno 和 symbolic name。
- 3：不支持的 Linux 架构/64-bit ABI 或 params 结构大小不匹配。
- 4：探针内部或 close 错误。

支持的 64-bit Linux 架构与现成作者封装一致：x86_64/amd64、aarch64/arm64、riscv64。目标环境不在该范围时不要猜测 syscall 编号。当前服务器 /usr/include/x86_64-linux-gnu/asm/unistd_64.h:340 确认 io_uring_setup=425；/usr/include/linux/io_uring.h:264–275 的 params 布局与脚本相符，当前 sizeof(params)=120。作者实现对应 py_kvcache/liburing_file.py:58–65、96–108、144–152。

## 本次实际运行

服务器仅执行一次新探针，命令记录于 commands.jsonl：

```bash
/root/miniconda3/bin/python -I -S artifacts/prefix_io_v1/new-server-03/platform-probe/io_uring_setup_probe.py
```

结果保存为 setup-probe-once.txt 与其逐字 JSON 副本 setup-probe.json：

- Linux 5.15.0-94-generic，x86_64，Python 3.12.3；
- effective_uid=0（原有身份，没有提权）；
- entries=2、flags=0、return=-1、errno=1/EPERM、exit=2；
- kernel.io_uring_disabled=0，kernel.io_uring_group=-1；
- Seccomp=2、Seccomp_filters=1、NoNewPrivs=0；
- RLIMIT_MEMLOCK soft/hard 均 unlimited；
- 无 fd 返回，因此没有可关闭的 ring；未尝试 mmap/enter/I/O。

## 诊断边界

EPERM 是该 syscall 在本进程/环境中的真实失败，不是根据 Python 异常字符串猜测。它发生在任何 ring 映射和实际数据 I/O 之前，支持把当前阻塞定位在最小 setup 的平台资格层。

Seccomp=2 只证明进程存在 filter 模式，Seccomp_filters=1 只给数量；这些字段不公开过滤规则或拒绝原因。不能由此声称唯一根因就是 seccomp。io_uring_disabled=0 表示本次读取未见该全局关闭开关启用，不保证容器、其他安全策略或内核条件允许此 syscall。

RLIMIT_MEMLOCK unlimited 说明本次进程没有观测到有限的 soft/hard memlock 上限；它不证明物理内存、cgroup/平台策略或其他内核资源条件全部满足。不同内核/平台对限制与 errno 的映射可能不同，本脚本没有改变任何限制，也不以另一组 flags 或更高权限尝试绕过。

请平台核查本实例对普通 flags=0 io_uring_setup 的支持与实际运行时约束，提供其支持的可运行环境或说明。可附本脚本、setup-probe.json 和本说明。未向平台发送消息。

探针将来返回 0 后，下一步仍须在获准环境验证原生 LiburingRing 的 mmap/完成队列及真实 SSD store/restore。setup 成功不能替代 O_DIRECT、异步 I/O、缓存生命周期或整个 P1 验收。

## 证据

- io_uring_setup_probe.py：可独立复制脚本；
- source-check.txt：AST/import 表面审查，没有执行 main；
- setup-probe-once.txt / setup-probe.json：本次唯一真实 syscall 运行；
- commands.jsonl：实际命令、时间、环境与退出码；
- probe-manifest.json：脚本/结果/本地 ABI header 来源哈希。
