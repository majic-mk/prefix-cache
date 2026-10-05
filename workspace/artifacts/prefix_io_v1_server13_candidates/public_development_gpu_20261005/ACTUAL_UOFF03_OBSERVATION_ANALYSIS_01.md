真实 U 功能与完整流观测：server13-public-development-uoff03

原 GPU guard 正常结束并排空会话；5 个请求完整生成 640 tokens，原引擎共 485 步，485 份真实 CUDA 事件见证。逐帧输出与原前台 token 序列一致，原 shutdown 和 native tail 已闭合。

包含 153 个不适合单一精确成本单元的帧（31.5%）；所有帧均保留。全步 CUDA elapsed 的 p50 为 13.243 ms，p95 为 18.457 ms。这是此次运行的描述性观测，未作为策略提升或成本表资格结论。

| 原生阶段 | 完成次数 | 实际完成字节 |
|---|---:|---:|
| ssd_read | 0 | 0 |
| ssd_write | 48 | 44040192 |
| h2d | 0 | 0 |
| d2h | 48 | 44040192 |

原生 journal 的阶段完成及字节可验证；capture 的每帧 existing_io/new_io 未通过 journal 补造。时间戳落在原 execute/sample 的 host 区间只描述时间包含关系，不能证明 I/O 导致 CUDA 或 token 延迟。

本小规模 pilot 未观察到 SSD 读/H2D，因此没有覆盖 SSD 冷前缀恢复机会。这不能证明方法无效；实际收益仍需满足成本资格、合法机会和独立对照条件后验证。

与 Uoff02 的五份实际输入 token 完全一致；全部输出 token 一致性结果为 True。Uoff02 的 CUDA capture 失败，未用其时延作为性能对照。

当前未签发成本表、普通 I 的成本与运行资格尚未成立、未验证策略性能提升，未推断服务 SLO。

证据：

- /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server13-public-development-uoff03/details/actual-request-outputs.json；SHA-256 8fddd5ea5af861a4e37297208cad8571c3b9fda648b9c090d30dc441d166a4c5
- /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server13-public-development-uoff03/details/actual-original-full-step-capture.json；SHA-256 7113b154192cf9019342f7234edd3dea683d21da2f3237ce01da8abd7458e4d4
- /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server13-public-development-uoff03/details/actual-native-stage-journal.json；SHA-256 0fadf95fc228dd7e9b83bb9bd0093641066ab9484b208cefd1efce48f82fba31
- /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server13-public-development-uoff03/details/strong-native-workload-result.json；SHA-256 ff9855c6c8a06219e1af3f099ea85b364b0e02ed0e344ab9a9debdcf69b89926
- /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server13-public-development-uoff03/result.json；SHA-256 1d264e45d05f9d88061a6183288f9bc6025bcd960298d7026bbe45d573dc3bc0
- /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server13-public-development-gpu-20261005/ACTUAL_UOFF03_COLLECTION_CLOSED_02.json；SHA-256 a35b739dc7b3d8319716cba635e35655d4d3fb9e92878ded2b741e0e5de7072c
- /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server13-public-development-uoff02/details/actual-request-outputs.json；SHA-256 8cc1731bf64719465f526b4583cb275924802e1969c857d37a9b46041a3c967c
