本轮 CPU 实施与服务器核验已完成，GPU 启动仍受当前驱动挂载阻塞。严格完整预检没有通过，不能将本包称为可直接执行的 GPU 环境。

新增 g3_reference_plan.py、g3_reference_runtime.py、g3_reference_result.py、run_g3_reference_pilot.py 及四份测试。它们冻结两个有限数值诊断作业，复用原 acquire_native_aio_costs.py、原异步缓存路径、原 SDK/原累计预算 guard 和原数值比较器。runtime 只在私有模块内临时接线 purpose、固定参数和 guard，结束后恢复；新旧原始回执同时保留。没有重写缓存引擎或模型执行器，没有启用研究策略，也没有修改作者文件、系统、驱动或已安装包。

服务器 Python 3.12.3 的最终四套 CPU 测试全部通过：plan 24、runtime 11、result v2 36、入口 14，共 85 项。CPU 夹具模拟参数、输出记录和生命周期，不能当作真实 GPU 模型输出。初次计划测试因命令中的日期目录写错未运行；初次结果夹具在 Linux 3.12 出现 relative_to/is_relative_to 互递归。两个失败日志均保留，后者只新增 v2 测试修正，结果模块未改。原比较器在数值差 1e-6 的夹具中真实返回 FAILED，入口保留失败并退出 1，未增加容差或升级资格。

实际服务器命令均显式使用 CUDA_VISIBLE_DEVICES=''，以下路径以项目根 /root/autodl-tmp/prefix-io-v1-handoff/project 为工作目录。完整执行记录和 stdout/stderr 见本目录 REFERENCE_SERVER_*.json 与 REFERENCE_CPU_REMOTE_COMMAND_TRANSCRIPT.json。

```sh
D=artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/test_g3_reference_plan.py" --g2-baseline-path artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/test_g3_reference_runtime.py" --source-root /root/autodl-tmp/prefix-io-v1-handoff/project -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/test_g3_reference_result_v2.py" --source-root /root/autodl-tmp/prefix-io-v1-handoff/project -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/test_run_g3_reference_pilot.py" --source-root /root/autodl-tmp/prefix-io-v1-handoff/project -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/run_g3_reference_pilot.py" --project /root/autodl-tmp/prefix-io-v1-handoff/project --preflight
```

四套测试退出 0；最后一条严格预检退出 78，原因是 existing file byte drift。另以同一服务器、同一冻结源执行独立只读源/输入/环境审计：4,088 个源引用全部逐文件读取并核验字节与 SHA，4,075 个祖先引用保持不变；24 个现有 SSD 缓存共 22,020,096 字节与冻结清单完全相同。原数值比较器及真实 YAML/config 依赖加载、恢复均通过；未导入 torch/vLLM/py-kvcache。四份既有交付清单的 97/24/90/101 个文件在服务器和本地均保持字节不变。

源锁 gpu-source-lock-reference-v1.json：910,072 字节，SHA-256 8443f45dd4dd6c1c64595199051f5e88449268af347bcbd836c927b754eabe30。CPU 计划、原比较器计划、模型上下文和输入清单全部纳入该锁；授权模板的 GPU 开关均为 false，human 记录为 null。没有生成新 GPU 授权。

当前只读环境证据：/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05 为 0 字节，SHA e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855；原 SDK 冻结清单要求 91,501,576 字节、SHA 76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6。没有证据证明其变化原因，因此只记录驱动资产挂载不满足资格；未改驱动、复制替代库或放宽检查。详情见 REFERENCE_ACTUAL_FULL_CPU_SOURCE_INPUT_AND_ENVIRONMENT_AUDIT.json 与 REFERENCE_SERVER_FULL_PREFLIGHT_01.json。

本轮实际 GPU 运行、GPU 元数据查询、模型后端导入均为 0。原 8 小时预算不变，GPU 账本前后 SHA 均为 be14eb7e34ccde501481777340ec3baf62e1f8511436614324ff976585ebf0f3，已用 17,809.140367632266 秒、剩余 10,990.859632367734 秒，活动预留为 null。最后只读快照 PRIMARY 可用 12,599,627,776 字节。

上一包 cold/populate/paired 02 三次真实 GPU 运行已完成，只证明该单点原缓存路径及关闭/OS 排空通过；没有验证研究策略的性能提升。本包也没有数值一致性结果、成本曲线、物理释放证明、完整 KV 字节证明或 P4/P5 效果资格。带 top-5 logprobs 的诊断计时禁止用于成本拟合。

下一步候选已经具体化：先 server09-g3-reference-native-01，正常关闭并由原 guard 证明 OS 会话排空后，才允许 server09-g3-reference-paired-01；每个作业 6 请求，各生成 1 token 并记录 top-5 logprobs，原比较器包括 warmup 在内比较 6 组外部缓存与 GPU 热缓存参照。两次各执行最多 300 秒、收尾预留 20 秒，总预留 640 秒；沿用原 8 小时预算，PRIMARY 总预留 2 GiB、保留 8 GiB，使用现有 Qwen2.5-7B、已有 24 缓存和冻结代码。失败即停止、不重试、不下载、不删除已有数据、不修改系统/驱动/安装包/作者文件。

当前允许继续做只读 CPU 交付与环境核验。进入上述两次 GPU 作业还必须同时满足：驱动挂载恢复且字节/SHA 与原清单一致、新人类授权绑定本源锁/CPU 计划/输入/原分析器计划、原 guard 核验 GPU UUID/实际会话/预算/存储通过。上一份授权限定的三次 GPU 作业已用完，不能用于这两个新作业。若恢复后环境身份不匹配，仍停止并保留证据。
