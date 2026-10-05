# 新服务器 io_uring 平台检查请求

目标容器：autodl-container-67e44a9c3a-cb95ca9e；SSH connect.westc.seetacloud.com:24801。仅暴露的目标 GPU 是设备 6（RTX 5090）。请求是解除该容器内项目所需的 io_uring 阻塞，不涉及驱动升级、其他 GPU 或共享数据清理。

实际证据：

- Linux 5.15.0-94-generic，x86_64。
- kernel.io_uring_disabled = 0，io_uring_group = -1。
- 进程 Seccomp=2、Seccomp_filters=1。
- 独立 raw io_uring_setup(depth=2, flags=0) 返回 -1 / errno 1 / Operation not permitted。
- 作者 LiburingRing 和 3 项真实 io_uring CPU 测试得到同样的 EPERM。
- 项目私有路径的 4096 字节 O_DIRECT 写入/读回通过。

这些事实表明不是“无 GPU”或 O_DIRECT 对齐失败。seccomp 是需要平台核实的原因，当前证据没有读取过滤器规则，不能断言唯一根因。

请平台核实此容器是否允许 x86_64 io_uring_setup（425）和 io_uring_enter（426），及宿主机/容器安全策略是否另有禁用。若需要调整启动配置，请保留项目磁盘内容并使这两个原生系统调用可以正常执行；原合同不允许用同步 I/O 或另一缓存引擎替换作者执行器。

调整后可执行以下 CPU 检查（无需 GPU/模型下载）：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES="" PYTHONPATH=third_party/work/py-kvcache-observer \
.venv-prefix/bin/python -c 'from py_kvcache.liburing_file import LiburingRing; r=LiburingRing(2); r.close(); print("io_uring_setup OK")'
```

随后运行 tests/test_py_kvcache_async_open.py，确认之前 3 项不是继续 skipped。完整证据位于 artifacts/prefix_io_v1/new-server-01/io-uring-diagnostic.json。未从容器内部尝试关闭 seccomp、修改 sysctl 或改驱动。

## 第二轮复测

作者 vLLM 已源码构建成功，指定 UUID 的基础 CUDA、原生复制及平台检查均通过，但原生 LiburingRing(2) 仍返回 EPERM。最新证据为 artifacts/prefix_io_v1/new-server-02/io-uring-current.json。这进一步将 SSD 阻塞与模型/复制扩展构建问题区分开，仍不能据此断言 seccomp 是唯一原因。
