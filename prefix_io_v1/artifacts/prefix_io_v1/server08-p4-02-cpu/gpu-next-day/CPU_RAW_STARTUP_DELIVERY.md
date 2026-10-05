# P4-02 明日 GPU 准备与 raw CPU 胶水交付

本轮只在服务器 08 无卡授权范围内执行。新增 GPU 运行 0、GPU 初始化 0、GPU 可用性探测 0、模型下载/加载 0、预算预留 0、原缓存删除/合并 0。GPU ledger 前后 SHA-256 均为 31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1。

## 已实施

- prepare_p4_gpu_next_day.py：读取现有 permissions/CPU 授权/账本/真实 CLI/实际空间，预览现有 run_gpu_stage 预算入口与新 P4 off/shadow qualification；检查真实工作区全部 Python 源、新增 load/ETA/verifier/startup 模块和精确旧 .so 的锁定覆盖。
- qualify_p4_native_gpu.py：新增薄 qualification runner，沿原 P316 tensor/FileMapper/TransferCoordinator/handler store/restore/shared 流程，选择新 P4 native/control/author Python；原 .so 仅精确锁定 fallback。每次 author import 校验源 SHA 并直接 compile 源字节，拒绝旧 Python/namespace/.pyc fallback。执行支路需要新 GPU scope receipt 和原 run_gpu_stage 唯一预算消费；今天默认与 --check-launch 拒绝退出 78。
- prepare_p4_calibration_startup.py：从原真实 ENGINE/SAMPLING 常量、缓存模型锁，准备原 BF16/eager/unquantized LLM 与单 stage 1/2/4/8 单位动作。沿原 add_request/step，未重写模型执行器。
- p4_raw_pair_recorder.py 与 prepare_p4_raw_pair.py：只接受完整显式 exact9 active-decode、冻结 context、四阶段 existing/new IO、完整配对 step windows/输出工作/accepted IO drain；append-new 四角色 raw 元数据，通过唯一现有 build_paired_cell_candidate 估计器生成 analysis，最终 existing loader 再做五角色/成本/字节绑定 roundtrip。
- native_gpu_recording 标签也不会将 CPU prepared table 提升为 GPU 或 production。scheduled-work-v1、P3 host execute timings、12-pulse aggregate、缺失 per-window counters、增长的 context 和多 stage 窗口均不自动转换。

## 实际验证与证据

94 项 CPU 测试通过，0 失败、0 skip：37 项 nextday/native-source guard、18 项真实 startup/config、39 项 raw preparation/CLI/final loader。CPU import guard 拒绝 torch/vllm/py_kvcache/cupy 等 backend，实测 GPU_modules_imported=[]。

- cpu-tests-06.xml、cpu-tests-06.log、cpu-test-guard-06.json：最终 94 项与账本不变。
- owned-source-lock-v4.json：本 agent 10 个源码/测试文件的实测 SHA；不是完整 GPU source lock，不可直接用于 GPU launch。
- RAW_STARTUP_CLI_RESULT.json 与 cli-execution-1..7-v1.json：真实服务器 CLI argv/stdout/stderr/退出码。
- calibration-startup-v1.json / calibration-launch-denied-v1.json：原执行器配置、未知 timing scope 与 actual active GPU load 采集要求，0 / 78。
- raw-cli-fixture-v1/：明确 origin=cpu_fixture 的合成 raw 与冻结 plan/ref。
- raw-cli-roundtrip-v1/：真实 CLI 产出四角色 raw、唯一 builder analysis、candidate、qualification candidate、最终 CPU receipt。没有真实 GPU 时间。
- plan-v3.json：本轮新工作区真实命令预览。
- native-shadow-dry-run-v3.json / native-off-launch-denied-v3.json：0 / 78，没有调用 GPU。

测试命令为 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -c <保存的 cpu-test-command-06.py 内容>，由该代码安装 CPU import guard 后运行 pytest.main(["-q","tests/prefix_io_v1_p4_next_day","tests/prefix_io_v1_p4_raw_preparation","--junitxml=.../cpu-tests-06.xml"])。CLI 执行使用 RAW_STARTUP_CLI_RESULT.json 中的原始 argv，与上述空 CVD/禁 pyc 环境。早期一次相对 log 重定向在测试启动前找不到目录，改成绝对路径后通过；没有将其计作测试或 GPU 实验。

## 尚未取得的真实资格

新 native 与新 author Python + 旧二进制 ABI、实际 CUDA/AIO/exact KV/full parent/shared/age/drain 仍没有新 GPU 验证。G1 runner 可在明确 scope 恢复、源锁与预算/空间 gate 满足后运行。

原 LLM startup 配置和离线 raw assembly 已可执行；可信实际 GPU step event + active-decode/context + 每窗口 native existing/new IO collector 尚未实现/验证。新 scheduler signature11 只代表真实 scheduled work，不能替代成本表 exact9 active GPU domain。先验证真实 worker/native 支持的采集接口、选合法单 stage 匹配窗口，然后才能采集成本；不能把 P3 aggregate 改标签作为新测量。

plan-v3 实测 PRIMARY free 10,896,822,272 bytes，8 GiB floor 之上只有 2,306,887,680 bytes。128 MiB primitive 可预检通过；原模型 case 固定 3 GiB 预留不够。未降低模型预留，也未删除/合并缓存。

下一 GPU 顺序：人工明确恢复限定 scope/同 UUID、最终全源锁/版本锁/账本/空间 → 原 native off/shadow CUDA+AIO+KV 合格 → 真实 observer/collector 与独立配对成本合格 → I/D/J 模式与模型 pilot。P5 SLO 仍 null、P5 不允许；方法效果与完整 P4 不得宣称通过。
