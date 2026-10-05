# Server09 正常 site / canonical parent 缓存修复最终 CPU 交付

本交付完成 off04 真实失败后的最小共同修复、服务器 CPU 开发和冻结。没有新增 GPU 作业；尚无 off05/shadow05 的 HUMAN_AUTHORIZATION_RECORD_G2.json 或 GPU_STAGE_AUTHORIZATION.json。本文件和提案不构成授权。

## 结果与实际修改

off04 真实运行127.51177137065679秒后失败，phases=[]，ValueError: unknown path hook，shadow04未启动。原权重和GPU KV初始化已到达，但LLM构造没有完成，没有正常原引擎shutdown证据。OS会话排空、进程内包装恢复和账本关闭已核验。完整负结果20文件在 server09-g2-optional-20261002 双侧冻结，manifest SHA3645f3e203861298afaa5702a84df236d6442ad98499d93c121a2341d5623b3e，失败数据未删除或改写。

真实正常虚拟环境启动 .venv/bin/python -B 包含第三个 editable vLLM namespace hook；原 -I -S CPU 测试跳过site，没有覆盖此启动差异。旧 helper 的全局hooks==2假设已在真实正常site CPU上复现同错。

修复只改变 optional_capability_probe.py 的 _validate_search_path，并在 _verify_absence 二次PathFinder前后绑定同一个缓存实例：
- 原 cached capability 函数与 strict BoundAuthorFinder 仍先执行且不修改；只处理原代码三帧精确缺包异常对应的一个可选查询。
- 仅校验原查询已经建立的 sole canonical 作者父目录 cached FileFinder。必须已有、exacttype、路径纯str、标准loader逐项身份、无实例find_spec/find_loader覆盖。missing/None/未知/漂移均拒绝。
- 二次PathFinder只用这个已验证的目录缓存，前后要求同一个对象。不枚举、修改或关闭全局site hooks，不调用它们，不对白名单之外来源放行。
- 其余异常传播、精确来源/物理缺包/唯一parent、外部deep_gemm、required import硬失败与restore语义保持。
- 新 runner 相对冻结V3只换目录、未来05名称和helper pin；worker13090B/SHA096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b整份不变。原模型执行器、py-kvcache路线、共享staging、预加载、复制合并、成本准入及异步I/O均未重写；本次策略仍off。
- 作者文件、已安装包、site hooks、系统、驱动、原permissions/守卫/finder保持原字节；无下载、既有数据删除或CPU sampling.so复用。

## 真实执行与测试

服务器使用 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B，保留真实site初始化，未使用-I/-S作为主要合同环境：
1. test_g2_worker_observation.py --source-root 项目 --scalar-source固定scalar --frame-source固定frame：22/22，0.043秒。
2. test_g2_normal_model_lifecycle.py --source-root 项目：31/31，0.093秒。
3. test_optional_capability_probe.py --source-root 项目：14/14，0.216秒。
合计67/67、0跳过。67为原51回放加helper14和root接线2个合同，不能当成67个全新测试或GPU通过。

实际 replay_normal_site_optional_query.py 以正常-B在服务器执行：保留三个真实site hooks，使用 synthetic top/capability namespace 和两份原AST函数，不导入真实torch/vLLM框架；原strictfinder加载真实空third_party/__init__.py。原缺包异常重现、包装返回False、required import仍拒绝、后续未知meta finder0调用、原函数恢复、缓存对象/全局hooks未改均通过。该证据是CPU启动和查询语义，不是完整模型运行。

最终入口 --mode off --name server09-g2-normal-off-05 --source-lock 当前锁 --preflight 在服务器通过全部4050 refs，gpu_initialized=false、actual_gpu_runs=0、runtime_hook_status=not_installed。原4045 refs序和值完全保留，只追加新runner/worker/plan/helper及旧4045锁本身；仅声明这些引用，不声称整份venv均已锁定。

helper本机14/14及独立14/14、7个必要缓存反例通过；whole AST仅上述缓存函数和same-cache两行改动。早期CPU装配因README名称错误在创建候选之前被拒绝，已按实际README.md修正。首份V4 CPU草稿虽67和replay通过，但独审发现继承plan残留03标签/旧ref和context；原样保留并镜像 CPU_REVIEW_BLOCKED，GPU0。最终plan从当前runtime.context、scope_template及实际文件ref重建；旧计划没有覆盖，不用于GPU授权。

## 当前冻结与预算

最终锁899396B/4050refs：
0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b

runner46746B/SHA82cea015522596a89a89eb1ea0245f9ea111c942b74456cd87288507aec54422。
helper13805B/SHA48138660e3ac43e1ea7b342a2727772e8745ed0f59c355c50f10616584a956d3。
当前plan14452B/SHA6ae76650b8ea50b3dd15669ca780ecffe9421336e95c5e2300c81902645d73d7。
helper独审10578B/SHAc4de3d61cd7bb0152238e105b6f00ce993ecd5147c8288853117bde6104786b1。

原8小时累计账本227事件，344164B/SHAa4e91c95ad640f1a1c1fda7bec3bcb3657e25276fa84522157ea43449e54ce55。累计16929.545041699428秒、剩余11870.454958300572秒（约3.30小时），active_reservation=null；无05事件或作业目录。PRIMARY空闲12852125696字节，满足每次128MiB预留及8GiB底线。预算未重置。

## 证据与下一允许阶段

实际命令、stdout/stderr/exit：ACTUAL_SERVER_CPU_NORMAL_SITE_TESTS.json、ACTUAL_CPU_NORMAL_SITE_QUERY_REPLAY.json、ACTUAL_SERVER_CPU_COMPLETE_SOURCE_PREFLIGHT.json、ACTUAL_SERVER_CPU_SOURCE_LOCK_ASSEMBLY.json、ACTUAL_SERVER_CPU_BUDGET_STORAGE_CLOSURE.json。有限独审：INDEPENDENT_CACHED_PARENT_REVIEW.json、INDEPENDENT_V4_ASSEMBLY_BOUNDARY_REVIEW.json。

本地：C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final
服务器：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002

拟定下一GPU命令（未执行，未授权）：
~~~sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode off --name server09-g2-normal-off-05 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/GPU_STAGE_AUTHORIZATION.json --launch
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode shadow --name server09-g2-normal-shadow-05 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/GPU_STAGE_AUTHORIZATION.json --launch
~~~

新限定授权方可 off05 后 shadow05：最多2次、每次300秒+20秒收尾、总预留640秒；仅off成功、原shutdown返回、会话排空才shadow；off失败停止且不借unused shadow slot重试。仍使用现有Qwen2.5-7B、CUDA13、Ninja和本冻结代码。每次cold/repeat各完整128 tokens，缓存计数0/112及完整数组一致性。该轮不是性能实验，旧04授权已因失败结束，不能复用。

按交接包03，P4为dependency_only/interference/joint最小策略、shadow/fake-backend和授权的小规模真机资格；单模型完整请求流/四臂性能归因属P5。当前G2正常模型/生命周期尚未通过；P4 GPU I/O、运行中真实释放、计时/成本及策略资格均未完成，P5效益未验证。新CPU证据修复了已知启动差异，不保证尚未真实运行的初始化/采样/输出与观察全链都通过。

