# I-only 首轮对照的 CPU 接线准备

本轮新增可以直接在服务器执行的原函数集成测试入口及冻结源码 bundle；没有发现必须修复的生产接线问题，因此没有为了产生改动而修改原 policy、bridge 或 reactor。初次测试中的三个失败均来自新增 fixture 对原接口的错误假设：结束 live step 后窄 cost-preview 计数不会增加；原 foreground reserve 会尝试回收空缓存；ring 接口名是 submit_pending。修正仅限新增测试，初次失败结果 LOCAL_I_BRIDGE_CPU_01.json 完整保留。

实际最终结果：20/20 PASS，0 failures，0 errors，0 skips。38 个源文件来自历史 GPU 实验实际接收并核验过的 payload，复制前同时比对本地完整候选和归档原字节；测试前后再次验证同一 38 行 SHA。来源 manifest 固定 SHA 为 4a2e4a2ffa7a16eefbbf7eda462b0856403a037adc7d7664c7a0d7b17f99efc1，测试入口不能靠替换 manifest 接受别的源码。

测试执行完整的纯 CPU P4Policy、NativeP4Bridge 和 collector 模块，并从原 reactor AST 提取、编译未改动的 28 个原函数及所需值类。原 AST SHA 和行号逐项写入结果。原 reactor 模块没有导入，Torch/vLLM/py-kvcache/SDK/ctypes/SSH 导入会硬拒绝。本轮没有 GPU 查询或执行，没有原生 I/O 提交、系统修改、下载、删除或账本修改。

## 已完成的接线验收

- off 在原 decision/wait/collect 的返回分支不取时钟、样本或额外元数据，继续原 ready 队列路径；make_p4_policy(off) 返回 None。
- shadow 对冻结的普通 defer 只记录预测，原 queue 仍进入 submit 边界，blocked_attempts=0。
- on 只连接现有 narrow single-file interference；dispatch_controller=None，不添加另一 allowance owner。
- 原 collect→reserve→P4Policy→NativeP4Bridge→defer→release 返回同一 ready FD，open_start 不变，原 preload_inflight 不退役。
- 128 次重试复用同一个不可变观察值和工作身份，释放全部临时 reserve，不刷新原样本时间或原到达年龄。
- 原 mandatory_support、max_wait 和 STOP 提供性能兜底；mandatory 仍不能越过原设备深度和 slot 取得失败。
- 原 incoming Queue 的 STOP 消息通过原 wait、_intake、_run、_pump_once 和正常 retained-cache cleanup 分支完成有限 CPU 运行；原 pump 次序仍为 completion→CQE→schedule→fusion→submit_pending→finish。
- 结束 live step、过期、上下文变化、bridge/capture fault、detach、native 新工作及未知 drain/accounting 均退出窄等待；intake 异常仍传播。
- 保留 metadata 值不可变、有限和弱引用；本次 fixture 的 slot release 不能当作原生物理资源复用证据。

## 实际本机命令

```text
C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe -B -I -S artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/i_bridge/prepare_i_bridge_bundle.py --project-root C:/Users/mamengkui/OneDrive/文档/论文2

C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe -B -I -S artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/i_bridge/test_i_bridge_cpu.py --output artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/i_bridge/LOCAL_I_BRIDGE_CPU_FINAL.json
```

来源复制命令 exit=0；最终测试命令 exit=0。初始 _01 exit=1、_02/_03 exit=0 是新增 fixture 修正历史，最终代码增加了 manifest 固定 SHA 和完整 _run/STOP 集成测试。每次输出为独立新文件，没有覆盖旧证据。

## 服务器复验入口

根代理可将 test_i_bridge_cpu.py、FROZEN_CPU_SOURCE_MANIFEST.json 和 frozen 目录原字节上传到新 CPU 专用目录，在 CUDA_VISIBLE_DEVICES='' 下运行：

```text
.venv/bin/python -B -I -S <新服务器CPU目录>/i_bridge/test_i_bridge_cpu.py --output <新服务器CPU目录>/i_bridge/SERVER_I_BRIDGE_CPU_FINAL.json
```

这一段是可执行的服务器验收命令，不表示本子代理已连接服务器或已获得服务器结果。bundle 只有原源码，不含历史模型/SDK、真实缓存数据、GPU job 或账本。服务器执行前后可复核 38 行原字节和 test/manifest 新源锁。

## 保留的真实限制

本 suite 内的 receipt、resource pool、live state 和事件均是明确的 TEST ONLY memory fixtures，submit 方法只是边界记录器。typed receipt 通过显式 object.__new__ 构造，以执行原 policy 的类型分支；没有调用生产 verifier factory、没有序列化或安装到真实 owner。origin='native_gpu_recording' 仅用于触发原 scalar accessor 的分支，不构成 CUDA 证据。成本为 90/100 的例子仅是与历史失败 cell 分开的 CPU 分支 fixture，绝不能替代真实历史 receipt。

主测试保留 U=16,238,752ns 与原 A-only=13,171,328ns。ordinary_frozen_candidate_admissible=false、cost_gate_changed=false；原普通成本门仍应 defer。本轮不能把 fixture 通关当作 real on、安全物理 drain 或收益，也不能把 CPU slot release 当作 GPU owner generation/refcount/protector 释放见证。

下一 GPU 阶段仍需要前瞻冻结协议、独立开发预算、真实成本覆盖和真实 on 完整输出/原 shutdown/OS session drain，然后才有同执行器 off/on 的请求流收益对照。原三次 off 的成本覆盖反例继续有效。依赖排序的真实释放见证属于另一独立接口工作；本轮不会借 I-only fixture 为 dependency/joint 策略授予能力。
