# Server09 G2 off05 / shadow05 真实模型闭环交付

G2 的正常模型完整输出与原生命周期验证已真实通过。off05 与 shadow05 均由原 GPU 守卫执行；四个请求各完整输出128 tokens，四份数组相同，cold/repeat 缓存命中均为0/112。shadow 的256个实际步骤帧完整重构其两个请求。当前 AutoDL 可继续此作者模型路径。此结论不等于生产 KV/SSD I/O、GPU 成本计量、新策略效果或P4/P5通过。

## 实际改动及依据

本轮共同修复仅调整 optional_capability_probe.py 对目标作者父目录的标准缓存 FileFinder 校验，并要求二次查询前后实例一致；保留原能力查询先执行、精确原缺包异常匹配、来源校验、强制导入失败及恢复。没有修改作者文件、已安装包、site hooks、系统或驱动。两模式使用同一修复，策略关闭；原 worker 整份13090字节不变。具体代码、正常 site CPU测试和独立源码审查已在前一份40文件CPU交付冻结：
/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002

最终源锁899396字节、4050引用：
0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b
CPU交付 manifest SHA：ef6b2ad10e99d9ca7bec476437fb73ac9ce2b7d4d3195284587f6fd571ee0483

此前 off04 的 unknown path hook 失败已完整封存，不覆盖旧记录。旧 -I/-S 测试跳过真实site第三个 editable hook，属于启动预检查覆盖不足。本修复在真实正常site的CPU回放通过后，由用户新的 off05/shadow05 限定授权执行；没有借旧授权重试。

## 真实执行命令与 GPU 情况

实际运行顺序是 off05，核验 off成功/原shutdown/OS排空并运行 shadow CPU前置门禁后，才执行 shadow05：

~~~sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode off --name server09-g2-normal-off-05 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/GPU_STAGE_AUTHORIZATION.json --launch
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode shadow --name server09-g2-normal-shadow-05 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/GPU_STAGE_AUTHORIZATION.json --launch
~~~

这两个 --launch 入口调用冻结的原 run_gpu_stage.py 和原模型执行器；没有直接执行子进程、绕过架构检查或复用CPU编译的 sampling.so。使用现有Qwen2.5-7B、CUDA13私有布局、Ninja和当前GPU UUID；未下载。当前两次授权已用完，不再运行GPU。

| 模式 | 原守卫/子进程退出 | 真实守卫GPU预算用量 | 完整请求输出 | Prefix缓存命中 | 原shutdown/OS排空 |
|---|---|---:|---|---|---|
| off05 | 0 / 0 | 137.8294115141034秒 | cold128 + repeat128 | 0 / 112 | true / true |
| shadow05 | 0 / 0 | 138.96918166521937秒 | cold128 + repeat128 | 0 / 112 | true / true |

原守卫用量包含初始化、编译、完整请求与收尾，不能当成稳态模型速度或策略开销对照。未超时、未重试。SID分别13706/14582，收尾成员为空，原 helper 包装恢复，运行后全部4050引用核验通过。

## 实际验证与证据

服务器正常site启动 .venv/bin/python -B、CUDA_VISIBLE_DEVICES='' 下的67/67 CPU合同和真实site缺包回放已经通过。授权生成从运行入口 scope_template() 的原Python类型构造19字段；temperature为float0.0，精确绑定用户题目/回答、源锁、GPU UUID、原权限、两次名称及640秒上限。两个GPU阶段之前的CPU门禁均通过。

本地实际闭环命令：
~~~powershell
python "$env:TEMP\PREFIX_SERVER09_VERIFY_G2_05_RAW_CLOSURE.py" "C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server09_g2_site_normal_20261002"
~~~
exit=0。该命令只验证已取回的真实文件，不运行模型、框架或GPU。核验包括10份原始结果/日志/frontend SHA，四份frontend与各normal结果逐字段一致，四个原生request identity独立、完整128整数数组相等；shadow连续ordinal0..255、cold/repeat各一次prefill及127个decode，完整fragment重构原输出，pending0/openfalse。规范输出数组SHA：
6980e281620405918aa14391eb75e06951a2e2fb440ed4e7bf3a06c6f1f4fc84

原始证据：runs/off 与 runs/shadow 各 result.json、process.log、details/cold-frontend.json、repeat-frontend.json、normal-model-lifecycle-result.json。启动命令/原stdout在 ACTUAL_OFF05_GPU_LAUNCH.json、ACTUAL_SHADOW05_GPU_LAUNCH.json；前置门禁、运行后有界读取、源锁门禁各有独立ACTUAL记录。LOCAL_ACTUAL_G2_RAW_CLOSURE.json保存完整闭环结果，INDEPENDENT_G2_OFF_SHADOW_05_REVIEW.json保存独审。

一次运行后CPU只读元数据查询因输出含全部诊断及私有编译缓存清单而超出工具长度，未形成有效JSON；已保留 ACTUAL_OVERSIZED_SHADOW_METADATA_READ.json。随后仅作有界CPU读取，取10个顶层原始证据和计数；GPU没有重跑。原私有编译缓存保留在原作业目录，不复制到本交付，也不删除。

## 原预算与数据闭环

before227事件、off后228事件、shadow后229事件。旧227事件原序原值保留，新两事件逐字段等于各原守卫result，其他账本字段不变，active_reservation=null。原8小时预算未重置；本轮合计276.7985931793228秒，累计17206.34363487875秒，剩11593.65636512125秒（约3.22小时）。最终账本347318字节：
7dd465ef868fb6cc4fbacddd8cc58045cc47b11aa0768765325613b454c6b744

读取时PRIMARY空闲12791758848字节；128MiB预留与8GiB底线均满足。所有旧实验数据、代码和本轮原始结果保留。本地和服务器交付按 DELIVERY_MANIFEST.json逐项核验；证明在 LOCAL_AND_REMOTE_DELIVERY_VERIFICATION.json，原GPU作业目录仍独立保留。

## 结论边界与下一允许阶段

本轮 normal_model_output_qualified=true；native_io=none、native_drain=not_applicable，SSD/production/effect/GPUcollector均为false。原输入是冻结的128个token IDs，本轮没有评估自然语言回答质量。shadow current-stream event只属于诊断；跨时钟映射未资格、全部 gpu_elapsed_ns=null，不能写入成本表或计算策略收益。

按原交接包03，P4是最小策略与小规模机制真机资格，单模型完整请求流/四臂性能归因属于P5。下一允许工作为CPU上准备一个生产KV/原offload handler的原生I/O正确性 shadow pilot：沿 register_kv_caches/get_handlers、原 transfer_async/wait/get_finished、原shutdown和 inspect_snapshot 接线；保留原LoadPlanner、预加载、共享staging、复制合并与异步管线。运行中可释放容量必须有同代真实refs/保护者/fence和原owner实际可复用见证；末尾shutdown不能代替。未知计时/成本/释放保持unknown，成本策略和I/J激活关闭。新GPU pilot需具体冻结代码和新的限定授权；本轮两次scope不能覆盖后续阶段。P4真实策略资格及P5效益尚未验证。
