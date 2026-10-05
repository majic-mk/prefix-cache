# 新服务器基础系统验证

2026-10-03，实际连接 `connect.westd.seetacloud.com:12351`。服务器项目为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。这台 RTX 5090 已完成六次真实 GPU 基础验证；无需继续更换服务器或镜像。本报告不宣称 P3/P4 整体完成或研究策略有效。

| 实际作业 | 结果 | 原 guard 计费秒数 |
| --- | --- | ---: |
| `server10-g3-reference-native-01` | 六个原生冷/热请求完成，正常关闭 | 126.536 |
| `server10-g3-reference-paired-01` | 六个 SSD/staging 恢复请求完成，正常关闭 | 126.828 |
| `server10-p4-native-off-01` | 原生存储、age store、恢复、共享预加载四阶段通过 | 48.550 |
| `server10-p4-native-shadow-01` | 相同四阶段及只观测路径通过 | 48.930 |
| `server10-g2-normal-off-01` | 冷/重复请求各生成 128 tokens | 142.536 |
| `server10-g2-normal-shadow-01` | 冷/重复请求各生成 128 tokens，实际采集 256 帧 | 141.863 |

六次原 guard/child exit 均为 0，无超时，原 shutdown/关闭返回，OS 会话自然排空。六次累计使用 635.243 秒（约 10.59 分钟）；原 8 小时预算累计使用 18,444.383 秒，剩余 10,355.617 秒（约 2.88 小时）。这是原守护器进程时间口径，不是 CUDA kernel 时间，也不是云平台总账单。最终 PRIMARY 可用 10,847,002,624 B，保持 8 GiB 下限。

原始模型参考中的 6 个 cached 对照全部通过原检查器：返回 token、top-5 集合及其 logprob 完全一致，最大差 0。SSD 路径每次实际读取 7,340,032 B；随后 staging 命中没有新增 SSD 读取。冷重算与 GPU-hot 的 top-5 差异另行保留，不能混写为冷/热全部逐 bit 相同。

G1 使用合成 canonical KV，逐内容检查通过；四阶段实际 staging 16,650,239 B，小于 16,777,216 B 限额，AIO accepted/completed/reaped 平衡并 closed/drained，accepted parents 最终为 0。该结果不能替代同一份真实模型生产 KV 的完整字节证明。

G2 off/shadow 共 4 请求、512 生成 tokens，重复前缀命中 112 tokens，off 保持无观测原路径。shadow 采集到真实执行帧，但原报告 `GPU_collector_verified=false`、`production_qualified=false`，不得据此宣称生产 ITL/时钟映射或 SSD 资格。

本轮实际增量为新机 UUID/SDK 配置、私有模块薄接线与 CPU 后处理修复。Python 3.12 整模块和孤立函数编译字节码不同，使 paired 父进程曾以 78 退出；其 GPU 子作业已成功。新增 CPU 后处理改用整模块编译结果核对函数来源，复用原数字比较器，未改变容差，未重跑 GPU，原失败日志保留。

服务器新增适配/边界 CPU 测试 48/48 通过：迁移合同 17、迁移 runtime 9、SDK 4、原生资格配置 4、后处理 4、G2 10。真实 CUDA13 小探针编译、链接通过；探针未作为 GPU 执行证据。新参考源锁 4104 项完整核验通过，原生资格源锁 4107 项，G2 源锁保留原祖先再追加。作者工作树原有改动完整保留；本轮未修改系统、驱动或安装包，未下载模型，未删除数据。

完整源锁、权限、限定作业配置、人类请求依据、实际启动 argv/stdout/stderr、原 GPU 结果及失败现场均保存在对应 `artifacts/prefix_io_v1/server10-*` 与 `experiments/prefix_io_v1/runs/server10-*` 目录。`CAPABILITY_MATRIX.md`、`PATCH_MAP.md` 与 `SERVER10_VERSION_AUDIT.json` 列出能力归属、修改位置和版本。

下一允许阶段是同一份真实模型 KV 的有限字节往返诊断，仍使用原执行器和原累计预算。之后还需当前机器有效成本标定、真实资源释放/mandatory 进展证据、主动策略及持续请求流对照，才可判断 P3/P4 完整资格与性能收益。不得将本次单 token 参考、合成 KV 或迁移通过替代这些门槛。

实际启动命令的完整 JSON argv 见每个交付目录的 `*_COMMAND.json`；原始 GPU guard 命令见每作业根目录 `result.json`。主要入口为 `run_server10_reference.py --mode cold|paired --launch`、原 `qualify_p4_native_gpu.py --mode off|shadow --launch` 和 `run_server10_g2.py --mode off|shadow --launch`。这些作业名已经消费，禁止直接重放；新的 GPU 作业必须使用新的有限作用域并计入原预算。
