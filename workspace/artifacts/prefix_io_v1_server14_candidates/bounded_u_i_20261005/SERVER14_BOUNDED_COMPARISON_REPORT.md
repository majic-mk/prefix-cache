本轮已完成一个冻结开发负载上的真实模型 U/I 对照，结论为 **未触发 I 可作用的目标竞争；没有证明方法提升**。整批请求观察到 I 比 U 慢 3.455%，但 I 实际延期为 0，不能把差异归因于主动延期策略，也不能据此宣布 I 算法在其他场景无效。

| 含自身开销的实际指标 | U | I | I 相对变化 |
|---|---:|---:|---:|
| 整批外部到达至最后输出（秒） | 25.088952 | 25.955887 | +3.455% |
| 请求流调用至原排空（秒） | 25.107749 | 25.973843 | +3.450% |
| 平均请求延迟，含排队（秒） | 13.129325 | 13.686835 | +4.246% |
| 平均首 token 延迟，含排队（秒） | 11.170458 | 11.658842 | +4.372% |
| 平均逐 token 间隔（毫秒） | 15.424147 | 15.968448 | +3.529% |
| 完整作业，含启动、方法、收尾（秒） | 118.119813 | 120.201273 | +1.762% |

两臂均 12 请求、每请求完整生成 128 tokens；全部 1,536 输出 token 逐条相同。原模型捕获各 1,536 帧及对应 CUDA Event 对齐；两臂 guard/child exit 均 0，原 shutdown 返回，native/AIO 和 OS session 均排空。后处理命令返回 0，`valid_comparison=true`。这证明对照记录完整，不等于证明方法收益。

两臂各自然产生 48 次 SSD 预读（44,040,192 字节）、605 次 SSD 写和 605 次 D2H。H2D 为 0，所有请求的前端 num_cached_tokens 为 0；未观察到这些 SSD 读取与 decode 的主机时段重叠。不能把这 48 次预读写成已经完成 SSD→GPU 恢复、帮助了某个请求或消除了 GPU 干扰。目标恢复及 decode 竞争的完整场景未成立。

I 已在真实原生 owner 安装，普通原生预读候选进入现有 I 接口；实际 eligible windows/calls 为 0，实际 native deferrals 为 0。I 条件记录：187 次未知/过时，956 次原生 mandatory/年龄优先，115 次限定成本签名不覆盖。未覆盖时确实保留原路径；没有追加 D/J、批量策略、资源释放 credit 或替代队列。唯一成本 cell 是 eager、batch1、decode1、prefill0、context783、原有 I/O 全零、单次917,504字节 SSD_read，不能泛化到其他运行负载。

实验使用现有 Qwen2.5-7B-Instruct，原模型来源/完整历史散列证明和本机叶元数据保持一致，没有重算 15 GB 权重或下载。设备为 RTX5090 / GPU-cb140ee8-a52e-93ba-38f7-e2fddfbb43b3，驱动595.71.05；原作者执行器和 py-kvcache reactor 保持冻结。原日志执行器版本为 v0.1.dev1+g817a7e312。

共同参数为 BF16/eager/Triton/Uni，模型并行序列1，前端最多3，预加载 lookahead2；GPU KV256MiB、共享 staging128MiB、iodepth4、parent上限8。固定的12条开发请求选自已有 P3 token 数据，前11条不同前缀，最后精确重放首条，每250ms外部到达一次，seed0/temperature0/ignore_eos。原 Prefix Cache、LoadPlanner 成本准入、共享 staging、自然预加载、复制合并和异步生命周期保留。该适用范围是小容量、单模型序列的开发探索，不能声称多序列吞吐、真实服务 SLO、生产负载或独立泛化收益。

A/B 成本是两次独立真实 GPU 作业。B 在 offset16 通过原 preload_async 提交一次独立已有 KV 文件的 SSD-only 读取，明确只作成本测量；U/I 没有手动 preload、受控读取注入、人工延迟或缓存清空。实际128帧、原 I/O、原 shutdown、完成 guard 和累计 ledger 已按原校验代码全量回放。

A=12.889184ms，B=15.283136ms，增量=2.393952ms。阈值取本次 A=12.889184ms，仅是开发工程设置，不是生产 SLO 或统计上界。原前端报告缓存768，但真实首帧显示有效计算起点752/复算16，原模型有16-token加载失败回退日志；新 CPU 适配从真实首帧读取752并让原校验器重验128帧，原报告值768和原始数据均保留。

实际改动只在新的私有 server14 候选：p4_cost_table/p4_policy/p4_bridge 对真实 A/B 单个精确 cell 接线；finite_current_binding/finite_startup 给现有 I 补当前运行输入；薄运行器调用原 LLM、请求驱动、drain/shutdown；同用于 U/I 的原生返回计数仅保存数值；CPU 成本回放和结果分析。原作者文件、模型执行器、驱动、系统和已安装包未修改。关闭 I 后使用 U，未重实现缓存引擎。

服务器 CPU 拒绝无效成本证据测试最终7/7通过（Python3.12，0.086秒，GPU0）；私有 I 接线本地9/9 CPU检查通过。早期服务器测试曾因本地路径夹具导致6项错误，修复后6/6及7/7通过，失败日志保留。原始共同源码和实际运行入口按冻结 refs 核验；没有重复整套历史框架和全模型散列。

第一组 U01/I01 原始记录保留。I01 模型流及 shutdown/排空已经完成，但主线程直接读取受 owner 限制的 bridge.snapshot 导致作业退出1，因此该组未纳入最终成功对照。唯一修复是改为复制原 native_after.owner_snapshot.p4 已发布值；随后同一 V4 薄运行器完成 U02/I02。成本不重跑，既有执行器、模型、算法、请求流和共同安全修复不变。

历史限定 I 结果继续保留：2026-10-03 的 context144、LoadPlanner关闭、offset16受控单文件实验有181次真实延期，on 请求1.775296238秒，off1.664287303秒，退化约7.28%。它属于限定历史负例，不能冒充本轮自然负载结果，也没有重新归零为完全未测。

本机保存 ACTUAL_U_I_COMPARISON_02.json、ACTUAL_AB_CLOSURE_01.json、FAILED_FIRST_PAIR_01.json、DELIVERY_MANIFEST_01.json、全部6次原始结果/guard/process.log和成本/效果源锁（project_evidence/）；21份原始运行证据及源锁共38,018,120字节，按服务器冻结 SHA逐项验证备份。原始服务器证据根：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server14-bounded-u-i-20261005；原始执行日志在 experiments/prefix_io_v1/runs/server14-bounded-*/。

实际执行的每条 argv、PID、启动时间保存在 LAUNCH_{A01,B01,U01,I01,U02,I02}.json。最终两臂命令（均先后在服务器执行）：

```bash
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path artifacts/prefix_io_v1/server14-bounded-u-i-20261005/EFFECTIVE_STANDING_GPU_PERMISSION_01.json --label server14-bounded-natural-u02 --seconds 300 -- .venv/bin/python -B artifacts/prefix_io_v1/server14-bounded-u-i-20261005/bounded_original_model_run_v4.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server14-bounded-u-i-20261005/CONFIG_U02.json --execute
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path artifacts/prefix_io_v1/server14-bounded-u-i-20261005/EFFECTIVE_STANDING_GPU_PERMISSION_01.json --label server14-bounded-natural-i02 --seconds 300 -- .venv/bin/python -B artifacts/prefix_io_v1/server14-bounded-u-i-20261005/bounded_original_model_run_v4.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server14-bounded-u-i-20261005/CONFIG_I02.json --execute
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server14-bounded-u-i-20261005/test_development_cost_pair_cpu_v3.py
```

完整分析 argv 位于 ACTUAL_PAIR_ANALYSIS_02_COMMAND.json，执行结果 ACTUAL_PAIR_ANALYSIS_02_RESULT.json 为 exit0、0.223秒、GPU0。

本轮真实 GPU 作业6次，5次 exit0，I01 exit1；全部会话已排空。原8小时 ledger 本轮增加 681.429511秒，累计 27635.628578秒，剩余 1164.371422秒。此数是原 guard 作业时间，不是云平台账单时长。GPU作业已停止，未安排下一轮。数据盘剩余 45.77GiB；本轮没有删除数据，无需为此次交付扩大盘。

EFFECT_SOURCE_LOCK_02.json SHA-256：24e34e3921247eda58b7bbf7871c86c0da5dfec48e3eed9f9b958b1c7d7090a8

FROZEN_REQUEST_INPUTS_02.json SHA-256：a5cab33912bdd9d0fb58e3a927123bfa9f971b92649b4f3900636af5bfe6e754

bounded_original_model_run_v4.py SHA-256：7403456b6020bc588dcb1d77593bb400cd79d3fa2b987189a3da5f773175a1eb

下一允许阶段是交付后对现有记录进行 CPU/只读解释：为何自然 SSD 预读未转为 H2D、未形成 decode 竞争。**本轮停止于“未触发、没有收益证据”**，不自动追加成本表、资格框架、策略种类或 GPU 实验。现有结果达不到论文中“我们的方法带来性能提升”的证据要求。
