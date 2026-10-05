# 服务器09 Ninja 路径公共修复：CPU 准备交付

截至 2026-10-01，本包的服务器 51/51 CPU 测试、4,040 项冻结清单和预算/空间收口均实际通过。独立审查通过。新 off03/shadow03 未运行、未创建新直接人类授权；本 CPU 准备阶段新增 GPU 作业 0。

## 实际改动

旧 off02 在 CUDA13 原采样器 JIT 构建时找不到 Ninja，失败且无完整请求输出。新的独立入口 `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001` 仅增加固定现有 Ninja 的只读源/ELF/权限校验，和 actual guard 通过后、原 SDK 配置后、框架导入前的私有单链接及 PATH 接线。

固定目标为 `.venv/bin/ninja`：非链接、可执行 ELF，370448B，SHA `08639e194fffa7f08b259fc4abfa4803aff66b64de52549cee42ec527d55cea6`。每作业单独创建 `details/runtime-cache/build-tools/bin/ninja`，PATH 固定为 CUDA13 SDK/bin →该私有 bin→原合法系统路径；不把整段 .venv/bin 放入 PATH。记录普通链接/源引用/PATH，不导出全部继承环境。

源码门禁仅校验文件，不创建链接、不改变环境、不执行编译器或 Ninja。真正 guard 门禁通过后才创建私有链接，目录重复、symlink、非 ELF、非普通/不可执行文件和 ref/哈希漂移均拒绝。失败保留部分新目录，不绕过源异常。

原 SDK V2、框架、系统及驱动未修改。原 worker 和 worker tests 逐字节不变；独审按允许差异归一化后，runner 整个 AST 与旧 CUDA13 runner 完全一致。原模型、bf16、TRITON_ATTN、精确 Prefix、KV64MiB、采样温度 float0.0、完整128-token输出保存、异常、失败后禁止 shadow 和原关闭流程均保持。该公共修复在 off/shadow 两种模式都应用，研究策略关闭。没有 SSD I/O，native drain 仍 not_applicable，事件为诊断，成本/时钟映射/效果资格为 false。

新三项：

- runner：43354B，SHA `bad66233db2a6952e4e8cf2ef3804aadc056e4ef9595ba9b6245c1dc3b9d1535`。
- worker：13090B，SHA `096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b`，原字节不变。
- plan：16514B，SHA `19441c85dd70aa114d3cc58d81e63e8f3c61ef657a196753e7e3c036e434c610`。

## 实际 CPU 执行与验证

在原 ROOT 项目目录，用 `CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S` 实际执行：

1. `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/test_g2_worker_observation.py --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project' --scalar-source artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py --frame-source artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py`：22/22 PASS，0.043s。
2. `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/test_g2_normal_model_lifecycle.py --source-root '/root/autodl-tmp/prefix-io-v1-handoff/project'`：29/29 PASS，0.084s。服务器 Linux 无 skip；本机初先实跑 50 PASS＋1明确 Linux symlink skip 的记录保留。新51为前48回放＋3项 Ninja 合同，未作为51项全新验证重复累计。
3. `artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --preflight --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/gpu-source-lock-candidate.json`：退出0，SOURCE_LOCK_VERIFIED_BUT_NO_NEW_G2_HUMAN_SCOPE，source_count4040、gpu_initializedFalse、actual_gpu_runs0。
4. 标准库分片组装与收口命令实际验证连续 offset、字节、SHA，xb 发布新锁，确认原4,035项原值原序保留、原入口/源锁/权限不变、旧225事件账本不变、新03名称/授权尚不存在、空间与预算足够。

新源锁 896797B / 4040 项，SHA `7b8445363e985a7dc4bff08960b31ee4cc325e3d309322f7f4b15d5b0f8aa423`，只在原4035清单上新增5项：新入口三文件、Ninja ELF、原 f3 源锁本身。全部现有冻结引用的服务器字节已经真正读取并核验；模板仍无 GPU 权限。

独审 `INDEPENDENT_BOUNDARY_REVIEW.json`：17项独立只读回放及 Ninja 正例/六负例门禁通过，AST/8文件 refs/15个rawparts及整锁原始字节重建通过，GPU0/无阻断。

## 真实 CPU 编译与上一 GPU 结果边界

上一 off02 真实 GPU 66.00550774950534s/exit1，全部权重加载后报 ninja 缺失，phases[]，shadow02未运行。LLM 构造未完成，不能声称原 engine shutdown 已返回；原 guard 真正会话排空。

之后 root 用该原 build.ninja，仅改输出 build 目录至新CPU目录，原3 CUDA输入、CUDA13编译器、SM120f/全部编译选项和链接库保持，真实执行 Ninja -j1 -v 和 readelf-d，96.23423069156706s CPU PASS；sampling.so 13184480B/SHA89a017e0…及全部日志/.o/依赖保存。该库从未加载、GPU kernel 未执行、framework未导入，原账本不变。未来 GPU 每作业仍原生 JIT，不复制复用CPU.so。

完整 GPU 失败及 CPU 编译原文件包：本机 `artifacts/prefix_io_v1_server09_g2_cuda13_20261001/SERVER09_G2_CUDA13_FAILURE_AND_CPU_SAMPLER_REPORT.md`，对应服务器 `artifacts/prefix_io_v1/server09-g2-cuda13-20261001`；29文件两侧真实核验通过，manifest SHA `a2d27170a288e408d163fdd78cdced490893c376ce651d790c083c4ed7d9624e`。旧失败和CPU编译文件均保留。

## 证据与下一允许阶段

本目录 `ACTUAL_SERVER_CPU_NORMAL_MODEL_TESTS.json`、`ACTUAL_SERVER_CPU_COMPLETE_SOURCE_PREFLIGHT.json`、`ACTUAL_SERVER_CPU_SOURCE_LOCK_ASSEMBLY.json` 和 `ACTUAL_SERVER_CPU_BUDGET_STORAGE_CLOSURE.json` 包含实际命令、stdout/stderr、退出码和真实资源。完整文件清单为 DELIVERY_MANIFEST；remote/local 双侧 proof 在冻结后单独追加，后续人类授权/真实GPU须另包记录，避免源锁循环变化。

原累计 GPU 16670.387764256913s，剩余 12129.612235743087s（约 3.369h），225事件/active_reservation=null，账本SHA `2d9dbb06280e10aa395f4d9eb1fba7606c401d1ff0c8d4d6a474c454f5c16276` 未改。CPU收口 PRIMARY 空闲 12917706752B（约 12.03GiB），可保持128MiB预留和8GiB下限；128MiB是计划预留，不冒充编译缓存硬上界。没有新增下载、系统/驱动更改或数据删除。

下一阶段只有新的直接人类授权后才可运行 off03，通过完整输出、原 shutdown 和 guard 排空后才 shadow03。最多2次，每次300s＋20s收尾，最多总预留640s，继续原8h预算。旧02 scope 的 shadow只有成功off02后有效，不能借为 retry；原源锁/标签/context已冻结，修复新入口必须新的源锁绑定授权。

资格作业仍每次cold/repeat各完整128 tokens，最多4请求/512输出，现有Qwen2.5-7B模型。任一off03失败即停止shadow03和同scope重试。即使资格通过，也不能声称成本、SSD、四组性能对照、P4或方法效果已通过。
