# Server09：无卡阶段完成，下一轮 GPU 原路径验证已准备
日期：2026-10-02（北京时间）

当前结论：CPU 软件入口、有限作业合同、原采集器依赖、源码与非驱动工具链均已完成实际服务器核验。状态是 `CPU_ENTRY_READY_REQUIRES_GPU_POWER_AND_NEW_SCOPE`。当前没有可见 NVIDIA 设备节点，驱动库是 0 字节占位；GPU UUID、显存、驱动加载与原生 I/O 的真机资格仍未验证。本轮 GPU 操作、模型加载、GPU 预算新增消耗均为 0。

## 实际新增与边界
新增薄运行接线 `g3_calibration_runtime.py`、作业合同 `g3_calibration_plan.py`、结果与 SSD 发布校验 `g3_calibration_result.py`、具体入口 `run_g3_calibration_pilot.py`，以及 CPU 检查、实际依赖回放和无卡就绪核验脚本。没有修改作者执行器、缓存算法、已安装包、系统或驱动。共享 staging、预加载、复制合并及作者异步 I/O 保持原路径；模型调度固定同步。新研究策略关闭。

复用原 `acquire_native_aio_costs.py`、已通过 G2 的作者 Finder/CUDA13/Ninja/site 接线，以及原 `run_gpu_stage.py`。修复限于当前 PRIMARY 授权接线、紧凑原请求观察与正常 shutdown 后只读标量观察。未额外消费 get_finished/poll、释放资源或增加请求。原返回与异常保持一次调用；所有临时 helper 最终恢复。

本轮是原生路径与原 TTFT 采集资格 pilot。raw calibration 按原采集器使用 planner off、curves None；这不授予生产成本准入资格。关闭研究策略仍走原路径。TTFT 是原 metrics.first_token_latency 的秒值；不是 GPU 事件耗时、纯 I/O 带宽或研究策略收益。完整 KV 字节、资源释放、GPU 时间、P4 效果、成本曲线及性能提升资格均保持 false/UNKNOWN。

## 实际服务器执行与结果
项目：`/root/autodl-tmp/prefix-io-v1-handoff/project`。所有 CPU 命令使用 `CUDA_VISIBLE_DEVICES=''`。

- 计划合同：31/31。
- 薄接线：18/18。
- 原结果与发布校验：29/29。
- 具体入口：8/8。
- 合计 86 个独立 CPU 测试通过。没有把回放次数或重复检查累加为新测试。

原始命令、输出和失败也保留：
1. `run_g3_cpu_checks.py` → `cpu-evidence-01/ACTUAL_SERVER_CPU_RESULT.json`。计划与接线通过；结果测试一项因遗漏 Linux --source-root、误取 Windows 镜像路径失败。
2. `run_g3_cpu_remaining_checks.py`（正确 --source-root）→ `cpu-evidence-02/ACTUAL_SERVER_REMAINING_CPU_RESULT.json`。29 个结果测试、8 个入口测试、真实正常 site 支持导入通过。完整源锁检查后，严格 SDK 门禁因无卡驱动 0 字节返回 78；未标为 GPU 成功。
3. `verify_g3_no_gpu_readiness.py` → `ACTUAL_NO_GPU_READINESS_RESULT.json`。实际核验全部 4,063 源/资产引用、1,934 工具链文件（217,111,529 字节）、现有 CUDA runtime、原 CPU 编译输出、Ninja 和可选依赖缺失事实。仅驱动列为 DEFERRED。没有返回伪 SDK pin、修改任何 GPU 门禁或加载框架。

可复现最后一条实际 CPU 命令：
```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/verify_g3_no_gpu_readiness.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --source-lock artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/gpu-source-lock-final.json
```

独立审查记录在 `INDEPENDENT_CALIBRATION_CLI_CPU_BOUNDARY_REVIEW.json`。原真实测试日志与两个实际服务器回合均保留，结果没有被覆盖成 PASS。

## 版本、预算与当前能力
GPU 源锁：904,019 字节，4,063 引用，SHA-256：
`263b3eea9560de916860fb3340e9421ad76a49d3a15d4008857773e342fc152e`。
完整保留 G2 的 4,050 引用及其基线锁，仅追加新文件。

原账本 SHA-256 保持 `7dd465ef868fb6cc4fbacddd8cc58045cc47b11aa0768765325613b454c6b744`；229 事件，active_reservation 为 null。累计使用 17,206.34363487875 秒，原 8 小时预算剩余 11,593.65636512125 秒（约 3.22 小时），没有重置预算。
最后实际 PRIMARY 空间 12,784,250,880 字节；拟预留 3 GiB 后仍满足 8 GiB 下限，启动前重新实读。

| 条目 | 当前证据 | 下一步 |
|---|---|---|
| 原 CPU 合同及真实依赖导入 | 86 测试、正常 site 回放通过 | 已完成 |
| 原代码和非驱动工具链 | 全 4,063 引用与 1,934 文件实际核验 | 启动前保留完整核验 |
| 当前 GPU 设备/UUID/显存 | 无设备节点；未取得真机事实 | 有卡后核验 |
| 原驱动库 | 当前 0 字节；预计身份按原库存冻结 | 有卡后必须通过原严格字节门禁 |
| G2 完整模型输出 | 前轮真实 GPU off05/shadow05 共 512 tokens 已通过 | 此轮不重复计为新增结果 |
| SSD 与 staging 当前原路径 | 本轮没有 GPU/原生数据 | 下一轮有限 pilot |
| 成本曲线/P4 策略效果 | 当前无合格新曲线或提升证据 | pilot 之后另行按计划推进 |

## 下一允许阶段：3 个有限 GPU 原路径作业
需新的人类限定授权，并恢复有卡模式。此前 G2 的两次授权已用完；模板仍 `NOT_AUTHORIZED`，没有生成新授权。

固定顺序：cold-01 → populate-01 → paired-01。每个最多执行 300 秒 + 20 秒收尾，总预约 960 秒（16 分钟），沿用原 8 小时累计预算。domain 1024、prefix 128、reps 3、max_num_seqs 1、iodepth 4；各 3/9/6 个原请求，共 18 个请求、每请求输出 1 token。rep0 为 warmup，f/g 各仅两个测量样本，不拟合/导出曲线，不宣称性能提升。

populate 只有在原 shutdown 成功且原 guard 的 OS session 完全排空后，由父进程发布 24×917,504 字节 SSD 清单；paired 在独立原 guard 进程中复验清单。任何作业失败、超时、源码漂移、关闭不全或会话未排空均停止；不重试、不运行后继、不覆盖旧数据。

恢复有卡后仍必须通过原驱动字节、登记 GPU UUID 与显存检查，当前无卡报告不能替代这些事实。若恢复后平台身份不同则停止重新确认；不修改系统驱动、不下载、不删除现有数据、不扩大研究范围。完成此 pilot 只证明当前原路径能否采集，尚不是 P4 策略收益验证。
