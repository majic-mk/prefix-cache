# Server07 P3 复现说明

根目录 /root/autodl-tmp/prefix-io-v1-handoff/project。
本次完整 GPU argv 位于 artifacts/prefix_io_v1/server07-p3-01/commands.txt，各 launch JSON 另有完整输出和退出状态。

运行环境沿用 artifacts/prefix_io_v1/new-server-03/native-prefix-04-launch.json 的 environment_overrides；
设置 PYTHONPATH 为绝对路径 third_party/work/py-kvcache-p2-aio、src、experiments/prefix_io_v1/scripts 的冒号连接。
CPU 测试另设 CUDA_VISIBLE_DEVICES=""；GPU 命令必须由 run_gpu_stage.py 包装，继续使用现有 permissions.yaml 和累计 ledger，不能重置账本。

重跑时：
1. 为 --label 和 --output/--output-dir 指定新的唯一标签。旧证据不可覆盖。
2. 使用已冻结的 manifests/p3-development-01 输入，不重新随机生成后混称为同一trace。
3. 成本采集顺序为cold、populate、paired，同组populate与paired共享一个新的私有storage；cold不启用外部connector。存储足够才开始，保留8GiB余量。
4. --domain 4096/8192/16384 都只用于采集。当前样本不允许发布为runtime曲线；原exporter对4K差异、样本不足均会拒绝，不应绕过。
5. --native-hot-diagnostic 仅可与cold组合；虽然CLI仍要求--storage参数，冻结的kv_transfer_config为null，该对照不经过外部缓存。
6. 调用方request_id与engine内部UUID不可混用。逐token证据来自每个新增单token对应的原生metrics.last_token_ts；多token块不得拆分成虚构时间。
7. 测量期间不同时运行CPU microbenchmark。所有预定请求、排队迟到、失败、JIT尖峰、末尾drain均保留。
8. 结束后检查GPU进程、显存、AIO回收、ledger预留及磁盘余量。

本轮实验脚本源版本既有最终source-lock，也有每次运行details/driver-source.py及早期失败版本快照。父类缓存/模型执行没有重写。详见 PILOT_PROBLEM_REPORT.md 的结果及限制。
