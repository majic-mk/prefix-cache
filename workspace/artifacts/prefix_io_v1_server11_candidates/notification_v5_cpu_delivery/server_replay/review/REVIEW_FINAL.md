# C5 通知等待协议：独立 CPU 审查

结论：最终冻结源码通过本报告范围内的 **32/32 项 CPU 反例**。可以进入预先定义的完整 reactor 双线程 CPU 比较；**不能据此认定 GPU-ready、性能提升、成本迁移或生产资格**。本审查未连接服务器、未调用 RPC、未使用 GPU，也未修改 C4、C5 或既有结果。

## 最终来源与实际命令

| 实际读取对象 | 字节 | SHA-256 |
|---|---:|---|
| C5 reactor.py | 185744 | a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47 |
| C5 native_full_step_collector.py | 24553 | bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf |
| C4 reactor.py | 178979 | 96bfd88dcee7f9c7996518ffe762b87be5ad81ef11d598329d41145e2b01fd87 |
| 冻结原 WorkerObservationConnection 源 | 13090 | 096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b |
| 冻结原 RuntimeScalarConnection 源 | 14795 | 347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb |

在工作区根目录实际执行：

```powershell
& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S artifacts/prefix_io_v1_server11_candidates/notification_v5_cpu_review/run_review_cpu.py --output artifacts/prefix_io_v1_server11_candidates/notification_v5_cpu_review/local_frozen_run01
```

冻结复跑：2026-10-03T15:25:11Z，Python 3.12.14，32 项全部通过，unittest 0.370 秒、子进程及来源复核墙钟 0.547 秒。全部输入前后 SHA 相同。原始命令、来源清单、stdout/stderr 在 `local_frozen_run01/`；此耗时是测试耗时，不是策略开销指标。

服务器复跑可使用同一 `run_review_cpu.py --candidate <C5目录> --previous-candidate <C4目录> --output <新的结果目录>`。保留现有测试所需的 `SERVER11_AUTHOR_SOURCE_ROOT` / `SERVER11_BASE_CONTROL_ROOT` 等项目环境设置。冻结的两份纯 CPU observer/scalar 源已逐字节复制到本审查目录 `frozen_observer_sources/`，测试先校验 SHA，再直接执行其中的方法，无需本机历史绝对路径。

## 已覆盖的协议边界

- 用真实 Python Queue、锁及线程执行 registration 之前、registration 锁内交错、最终检查到 get 之间、get 之后的 end；验证旧 run/nonce/step、重复消息、旧 pending wake 不清除新注册，合并通知最多保留一份未消费的 wake。
- end 的原 `raw.record()` 和 `record_after_ns` 写入先于通知入队。原 record 抛异常时仍尝试唤醒；通知异常不覆盖原异常或跳过记录。disarm 异常不吞原消息，也不覆盖原 `_intake` 异常。
- 直接加载冻结 observer/scalar 类，测试 worker invalidate、disable、observe 异常、original_failed、detach，以及 scalar fail、observe 异常、original_failed、detach 九个入口。它们无 end record 也能解除等待并拒绝再次停车；解除接线不覆盖较晚安装的外部 wrapper。
- scheduled-load 成功发布和同序冲突清 frame 均覆盖检查→get 竞态；不能假定其原共享标量接口已有 Queue 通知。未 arm 的 end/fail 不制造空控制消息。Queue.put 异常清 pending 并禁用后续停车。
- 使用实际 C5 源抽取的 submit_job、request_mandatory、intake 和 STOP：验证 FIFO、原 Future 未提前完成、shared/plain 保留槽只释放一次；STOP 从真实 get 唤醒且随后拒绝新等待。后端 pool 和 pump 工作是显式 CPU fixture，不是实际 I/O。
- 原 intake 去掉新增 Wake 分支后，AST 与 C4 完全一致。实际 deadline 谓词逐项拒绝 active、inflight、copy、其它 ready/pending、mandatory waiter、未知 drain、故障、容量漂移；所有计费 STAGES 的 inflight_ops/inflight_bytes 非零均拒绝停车。
- 等待期限取原 ready.open_start + max_wait、原 snapshot/native/state freshness 的最早值。反复检查不刷新四个时间戳；100 ms 原策略期限不延长。真实 Queue 的短超时返回原处理路径。OS 调度可能让线程晚于逻辑期限得到运行机会，测试不承诺硬实时墙钟上限。

## 资源、off 与范围判断

`released_slot=True` 只出现在等待资格的键比较调用，原决策调用仍默认 False。它仅将上次真实 reserve 后已 release 的 free/reclaimable 两项各减 1，重建旧键用于等值比较；测试确认该函数没有 reserve、release 或资源信用副作用，真实容量变化立即拒绝停车。

已读取 reactor 的容量采样、共享缓存 pin/unpin、普通/共享释放、evict、shutdown 和提交路径。现有实际槽/引用变更由 owner 的 pump/intake 推进；shutdown 通过原 Queue 发 STOP，join 后才 close pool/ring。停车资格要求无可推进的实际 I/O/copy。没有把任意外线程直接赋值内部 free_count 当作受支持接口，也没有漏掉本次检查到的合法跨线程容量入口。

off/uninstalled 路径没有新注册、锁、clock、Queue.get 或通知消息；原 pump 和决策保留。不过每次 `_run` 循环确实多了一次 helper 调用及 getattr 快速返回，**不能描述成字节相同或零开销**，这部分必须纳入完整 CPU 对照。

协议保证的是同一时刻一个 registration、至多一份待消费 wake；其它合法原消息可导致回到 pump 后重新核验并再次注册，不应把它表述为“每步绝对只注册一次”。这澄清了最初 `REVIEW_CONTRACT.md` 第 9 项的过强措辞。

修改遵守已读取 `04_CODEX_EXECUTION.md` 的原 Queue、原 owner、原 engine/executor 约束。end 通知仅表示原 host 记录边界已经经过，不表示 CUDA 完成、原始 GPU 槽释放或成本资格。保留 100 ms 原最大等待，不提高阈值，不给予额外释放信用。

## 审查中发现与修复的事项

实现代理修复了：optional catch 吞原 intake 异常、通知异常跳过原 end.record、scheduled-load 变化未唤醒、原 worker/scalar fault/detach 未唤醒，以及通知先于原 end.record 的额外阻塞。最初独立 18 项中 1 项通知失败反例报错；修复后 18 项通过，扩为 28 项通过。最后新增原始 STOP/FIFO 集成时，本审查 fixture 重复执行 after_prepare 导致 3 个子项错误；改为新建并只准备一次的 fixture 后，冻结 32 项全部通过。上述中间失败不冒充成功，不涉及改写既有冻结结果。

没有发现需要继续修改当前冻结 C5 的剩余源级必修问题。下一步只允许完成同源服务器 CPU 重放与完整 `_run/_pump_once` 双线程成本/延迟比较。GPU 校准、真实 I/O/模型输出、跨臂一致来源锁与有限授权仍是独立阶段；这份报告不授权或替代它们。
