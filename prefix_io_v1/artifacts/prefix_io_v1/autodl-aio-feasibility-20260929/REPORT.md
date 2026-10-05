# AutoDL Linux AIO 兼容路线评估（2026-09-29）
结论：现有 AutoDL 实例的 Linux AIO + O_DIRECT 必要能力实测通过，值得实施兼容适配；完整缓存集成与 GPU 路径尚未验证。原 io_uring 路线仍为 NO_GO。兼容后端尚未实现，等待用户明确调整原范围。

## 本轮实际执行
目标 connect.westd.seetacloud.com:40168；autodl-container-mz3u56hqpn-f855e672；Linux 5.15.0-78-generic。
- gcc -O2 -Wall -Wextra -Werror linux_aio_probe.c：退出 0。
- 独立 CPU 探针：io_setup(depth=2) 成功；O_DIRECT 排他创建专用文件成功。
- 两个 4096 字节写请求一次 io_submit 提交，io_getevents 返回两个正确 ID/字节数；fsync 成功。
- 两个 4096 字节读请求一次提交，两个完成记录正确；8192 字节逐字节相同。
- io_destroy、close 正常，退出 0。文件保留为证据。
- io_uring_setup 同机复测仍 errno=1 EPERM，退出 2。
- 此探针只证明小规模真实文件 I/O 能力，不证明物理并行、性能收益、大队列稳定性、生产 KV 正确性或端到端系统可行。
- GPU 工作负载 0；模型下载 0；未修改驱动、系统安全策略或现有缓存代码。未重新运行历史回归套件。

完整命令、stdout、stderr、返回码在 build.json、roundtrip.json、io_uring_recheck.json。
权限文件、GPU 账本及三个被审计源码的前后 SHA256 保持相同。

## 为什么需要用户调整范围
原 docs/prefix_io_v1/04_CODEX_EXECUTION.md 明确要求“不得重新设计缓存引擎、io_uring执行器、Prefix身份规则、attention、decode或模型逐层执行”。
当前没有现成 io_backend 配置，reactor 直接构造作者 LiburingRing。
传统 Linux AIO 是另一种后端，不能把它描述为原 io_uring 复现，也不能把替换后端伪装为小配置修改。
本轮只做能力探针和方案，不实施这个范围变更。

## 建议的有限调整（待授权）
继续使用锁定 py-kvcache 与作者 vLLM，允许增加可显式选择的 Linux AIO 兼容后端，作为各实验臂一致的共同基础；保留原 io_uring 实现与默认选择。
不改变 Prefix 身份、LoadPlanner 准入算法、共享 staging、预加载、复制合并、GPU copy 流水线、资源所有权、模型执行器和研究策略范围。
成本曲线必须针对新后端重新标定并冻结，不能沿用旧 io_uring 延迟参数。

拟改位置：
1. third_party/work/py-kvcache/py_kvcache/fs_config.py：新增明确的后端选择与严格校验（目前不存在）。
2. reactor.py：仅将 ring 构造接入后端工厂；不增加第二套缓存任务调度。
3. 新增 Linux AIO 适配模块：对接 queue_rw、queue_openat、queue_close、submit_pending、poll_all、close 及 pending_submit 语义。
4. 配置/环境锁/补丁清单/实验报告/测试：记录后端、队列/工作线程上限、实际内存和成本曲线身份。

Linux AIO 只承担支持的文件读写。open/close 无同等 AIO 操作，需要有界工作线程完成并回送结果；不能把阻塞元数据操作塞进 reactor pump。
io_submit 的提交延迟也要测量；若可能阻塞 pump，适配必须隔离该阻塞并保持有界提交。不能凭异步 API 名称保证非阻塞。
适配不分配第二套 staging、不复制生产 KV 到新的隐藏缓冲区；保持 user_data、短 I/O 和负 errno 传播。
入队/内核提交并不代表完成。只有真实完成后原 reactor 才能推进依赖；CUDA event 完成前不得释放 copy 持有的 slot。
关机必须停止接收、排空已接收操作与完成通知，再关闭 AIO context 和 FD，防止提前释放指针与双重关闭。

## 实验公平性与 off 语义
后端选择与研究策略是两个独立维度。
- io_uring + policy off：完整原路线，在当前容器仍不可运行，必须如实标记。
- linux_aio + policy off：共同兼容基础上的无新策略基线，不可称未修改作者基线。
- linux_aio + 各策略：与上项使用相同异步后端、预算、线程数、预加载和复制合并。
绝不能用同步阻塞基线对比异步策略来制造收益。
AIO 实验只能支持该后端上的结论；io_uring 性能推广需要以后另行验证。

## 授权后阶段顺序
1. AIO 后端有界队列、短 I/O/失败/部分提交、乱序完成、关闭排空和引用生命周期 CPU 测试。
2. 作者真实 staging 与 GPU 拷贝接口、生产 KV 往返、mandatory drain 和 policy off 验证。
3. 重新标定成本并恢复 P1 集成；满足原阶段门槛后继续策略实验。
当前 GPU 授权仍绑定旧 UUID，后续实际 GPU 工作前需核对现设备和明确范围，累计预算不可重置。

参考：
- Docker 官方默认策略允许 io_setup/io_submit/io_getevents/io_destroy：https://raw.githubusercontent.com/moby/profiles/main/seccomp/default.json
- 锁定作者 ring：https://raw.githubusercontent.com/atlarge-research/py-kvcache/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/liburing_file.py
