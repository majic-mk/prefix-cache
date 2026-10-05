# C5 真实 GPU 入口准备交付（2026-10-04）

已在指定 AutoDL 服务器 connect.westc.seetacloud.com:26909、项目 /root/autodl-tmp/prefix-io-v1-handoff/project 完成本轮真实 GPU 入口代码修订与服务器 CPU 验证。原生分支 23/23、绑定与控制器 35/35、独立审查 30/30，共 88/88 通过，失败、错误和跳过均为 0。现有已通过的 65 项 CPU 检查与 132 次性能基准没有重跑。

新增 server11-c5-gpu-entry-revision-20261004、server11-c5-gpu-entry-review-20261004、server11-c5-gpu-entry-delivery-20261004 三个目录。已恢复真实 native 六窗口执行、原始证据序列化、最终验证和 canonical receipt 的有效分支。它们只在现场资源、独立预注册计划、完整来源、明确本轮 GPU 执行授权和原预算守卫均满足时可达。此前 Stage C/D CPU-only 冻结版本保持原状。本次没有简单删掉旧 block 或复用旧 UUID/授权。

C5 公共包 64 个 Python 文件中 63 个字节不变，只适配新 canonical receipt；reactor、collector、policy、bridge 和实际 runtime identity helper 未改。32 个原数值、capture、I/O、排空及模型辅助函数 AST 保持一致。六窗 AB/BA/AB、两标定对及一独立 holdout、129 prompt / 128 cached / 128 完整输出、offset 16、单次 917504 字节读取、8 accepted parents 和 bridge=None 均保留，原公式和 holdout 不变。没有重写缓存引擎或模型执行器。

新 gpu_entry_binding 与控制器校验同现场 UUID、模型与生产 KV 来源、源码闭包、权限、固定单作业和原 guard 的实际会话。新 binding/config/context/grant/effective permission 全部纳入完整 site source lock，路径引用避免哈希循环。launch intent 独立钉住预注册 plan 和 PLAN_REFERENCE 容器；raw/window/final 回放拒绝后验自引用、旧 job、CPU synthetic 计划和类型/来源替换。直接执行 parent/window 也执行相同守卫。

原 run_gpu_stage.py（SHA-256 3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a）独占持久预算、时限、会话清理及实际消耗记账。完整资产字节哈希核验位于作业准备、启动前和结束后，模型权重不在六个子进程或测量窗口内反复全量重哈希。入口核验实际代码、site 文件和完整核验凭据；最终成本资格仍要求真实六窗记录和真实完成证据。原 guard 发布 session 的短暂竞态仅允许最多一秒元数据等待，不是 GPU 重试。

实际服务器调用均设置 CUDA_VISIBLE_DEVICES=''，使用项目 .venv/bin/python -B -I -S：

```text
freeze_gpu_entry.py --root <project>
run_cpu_native_entry.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_GPU_ENTRY_CPU.json --output-dir <revision>/SERVER_NATIVE_CPU_01 --location server_cpu
run_cpu_entry_binding.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_GPU_ENTRY_CPU.json --output-dir <revision>/SERVER_BINDING_CPU_01 --location server_cpu
run_review_cpu.py --project-root <project> --source-lock <delivery>/SOURCE_LOCK_GPU_ENTRY_CPU.json --candidate-root <revision> --output-dir <review>/SERVER_REVIEW_CPU_01 --location server_cpu
control_native_cost_job.py prepare
control_native_cost_job.py context
seal_gpu_entry.py --root <project>
```

完整全路径 argv、stdout/stderr、退出码保存于各目录 SERVER_SOURCE_FREEZE_*、SERVER_CPU_NATIVE_*、SERVER_CPU_BINDING_*、SERVER_CPU_REVIEW_*、SERVER_UNBOUND_PREPARE_*、SERVER_LIVE_CONTEXT_GATE_* 和 SERVER_FINAL_SEAL_*。prepare 实际退出 0，仅创建 UNBOUND_REVIEW_TEMPLATE.json；context 实际退出 2、stderr 为空，返回 GPU_ENTRY_BLOCKED，原因是现场资源不可用。九个有效现场/配置/授权/作业文件均不存在。模板中的 context/bind/plan/scope/launch/after/verify 命令只是可审查命令，不是执行授权。

CPU source lock 共 318 个引用，SHA-256 e91df96e4203ae4b4925e90613a277632ad2ae3a3a337e9e947ad29c7f7431f5；原生与绑定运行前后逐项核验，独立审查连同锁本身共 319 个引用。原生 CPU 导入审计是在全部测试结束后实际枚举 sys.modules 和记录被禁止的导入尝试，结果均为空。1,153 项旧源码与证据在冻结前真实核验；封存时再次核验，未改旧源码、旧权限、旧结论或用户的其他工作。

本轮服务器只读核验了现有模型目录和 24 个生产 KV 文件（总 22020096 字节）及 3 个 producer 来源文件，全部 SHA 一致；没有下载或重新生成模型/缓存。旧 v6 的 4655 项清单这里只核对清单原始 SHA 作为后续真实资产 inventory 来源，不声称本轮已经重新核验全部模型权重/旧清单行，也不继承旧 GPU 资格。真实现场 context 可用后会执行完整资产核验并冻结新 revision/site 闭包。

本机开发中独立审查首轮为 23 通过、7 个 Windows 临时路径过长错误，发生在业务验证前；仅缩短新 review fixture 路径后 30/30 通过。LOCAL_DEVELOPMENT_RECORD.json 保留失败原因，旧结果与源锁在本机保留。其他本机 20/21/23 原生开发测试记录也保留，它们不能替代服务器最终 23 项或真实 GPU 证据。新的接口审查还修正 Path/string 参数、真实 canonical 路径、严格类型、guard 身份以及实际 CLI 参数和计划引用的接线；原算法和验收门槛未调整。

现场仍无 NVIDIA 设备节点，CPU 配额 50000 100000（0.5 核）、内存 2 GiB，不满足模型验证条件。真实 GPU 运行 0 次、预算新增消耗 0 秒、真实模型进程 0、性能比较 0，没有有效 native receipt 或成本。账本原 SHA-256 bef6d78e43058eaad811ab5ecb08937356c911784f9000e76eea9de0276c0b72 未变，累计 22380.561257688794 秒，原 8 小时剩余 6419.438742311206 秒（约 1.78 小时），无活动预留。数据盘约 56.9 GiB 可用，本轮不需要扩容、租赁、支付、删除、系统或驱动修改。

代码层面的公共六窗口 GPU 启动修订已准备好。下一允许动作是开启 GPU 模式后执行只读 context 核验：单 GPU、至少 28000 MiB 空闲显存、无其他 compute 进程、至少 1 核实际 CPU 配额、32 GiB 可用内存、PRIMARY 至少 10 GiB 可用（含 2 GiB 预留和 8 GiB 底线）。满足时冻结实际 UUID/source/context，准备具体本轮授权请求；用户明确授权后 bind → plan → scope → launch。拟作业只运行一次，最多六子进程，1200 秒执行加 20 秒收尾（1220 秒），在原累计预算内，失败停止，禁止自动重试。

公共六窗得到严格真实 native 凭据后，才允许 off/shadow/on 生命周期与完整入口成本，验收通过后 P4 公平比较。此轮只准备公共标定入口，尚未完成真实 normal/on 集成或其成本。实际没有 defer 时记录 NOT_EXERCISED，不改变负载、额度、成本、SLO 或选样制造提升。CPU 检查不能证明 C5 GPU 收益、系统最终可行性或论文要求。

证据压缩包为 C5_GPU_ENTRY_PREPARATION_EVIDENCE.tar.gz；SERVER_BACKUP_RECEIPT.json 给出实际文件数、字节数、manifest 与 archive SHA。本机完整校验后保存独立 LOCAL_BACKUP_VERIFICATION.json；最终原件、来源快照与实验命令均在三目录中，不包含密码。

