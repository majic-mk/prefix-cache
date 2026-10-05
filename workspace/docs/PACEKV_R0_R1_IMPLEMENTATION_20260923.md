# 从 ProbeKV 转向精确 KV 缓存资源调度：完整实施与止损合同

日期：2026-09-23。本文件细化并修订同日研究提案的执行范围。旧代码、未提交改动、Source 实验及负结果全部保留；新代码在 `src/pacekv/` 隔离。PaceKV 是临时工作名。

## 1. 定位与硬件决定

新问题：单 GPU 上外部精确前缀 KV 的读取、可选写回与正在 decode 的请求争用资源；延期写回还可能使 GPU 块和 pinned buffer 不能及时归还。研究是否存在一个比充分调优的异步、固定配额、读优先、选择性写回策略更好的有界联合决策。

不再在新主路径中做历史 Source 排名、近似非前缀复用、repair、CFO、QCFuse anchors 或 SparseX。不是将旧系统简单设为 K=1，也不声称旧实验已经证明所有多 Source 思路无效。

硬件决定已更新为优先 RTX 5090 32GB，执行合同见 [5090 租机预检](PACEKV_5090_RENTAL_PREFLIGHT_20260923.md)。以下关于 A800/Torch2.2.1/CUDA12.1 的叙述仅保留为历史 R0 交接背景；新 R1a 的协议、环境审计和 GPU guard 均已单独版本化。旧定制 vLLM0.4.1 不进入 5090 路径。不能以旧 A800 结果代替 5090 证据。

无卡检查时服务器31780可连接，数据盘约22GB可用，未发现运行中的P1任务或GPU。不要清理历史目录来腾空间，也不要全局升级旧环境。现代原生 offload 引擎的独立环境须先估容量、驱动和接口支持，必要时由用户提供额外磁盘。

实际无卡容器上限仅2GiB；宿主机`free`输出不能代替cgroup限额。CPU代码/文件预检可以完成，但模型加载需挂卡后重新观测至少32GiB主存余量及24GiB空闲GPU显存。这里是当前CPU-first加载器的保守执行条件，不是对所有加载器的理论最低要求。资源不足则不启动模型，不通过移除guard制造readiness。

## 2. 不变的正确性与成本边界

- 精确命中要求相同完整祖先前缀、模型权重、tokenizer/template、位置/RoPE、adapter/attention、dtype/layout/backend和namespace。只同文档/目标token或旧`DENSE_EXACT`标签不够。
- 第一版只发布完整块；最后不足一块不padding、不假装可见。不复制旧target-only Source key。
- 字节无损、同执行计划数值一致、调度变化后的回归分别验证。不能用答案F1掩盖缓存错配。
- 传输必须持有源、目的、pinned所有权；已发出的DMA取消后也须等待完成或隔离，不能立刻回收。
- D2H完成可释放不再被请求使用的GPU源块；SSD写完成前CPU缓冲仍不可复用。活动推理引用始终优先。
- 只能放弃尚未发出的可选持久化，不能丢弃当前推理必须的KV。native allocator绑定未实现之前，账本只是CPU合同。
- byte-ns是资源占用积分，不能用传输字节替代；观察完成时间给出的上界不得冒称精确释放时间。
- 新目标是TTFT与ITL双SLO goodput，不继承旧`gamma=0.8`。正式SLO需在独立开发集上冻结，首批微基准不随意设值证明收益。
- 缺失测量为UNSUPPORTED/null，失败、超时、无输出不算成功。不把CUDA分项再次加到host关键路径。

## 3. 阶段依赖与具体工作

### R0：今天无卡交接

实现：完整前缀key、在途所有权/取消/隔离、GPU和pinned分开释放、token时延计算；固定微基准清单、CPU环境/资产核验、显式授权GPU入口、超时监督、原始工件摘要和配对分析。

在原服务器新唯一目录部署；只运行CPU测试和权重/tokenizer/代码摘要检查。不加载完整模型、不启动CUDA、不安装或修改旧推理依赖，不启动旧P1。

交付是“R1a诊断可尝试运行”，不是“新系统已经完成”。现有状态机、模型质量或GPU老结果不替新系统背书。

### R1a：明天首批，输运正确性与干扰存在性微诊断

为了先验证机制，5090 R1a 使用独立环境中的 Transformers4.40.2/Torch2.7.1+cu128 原生Mistral forward/cache接口。该版本组合仍需在真实 5090 上通过导入、小模型与全模型预检。**这不是vLLM连续服务，也不是新系统正式baseline。** 所有arm同模型同执行接口。没有改attention、没有走旧repair路径。

输入为明确标注的构造形状，不是自然QA；长度512/2048，decode batch=1，生成64token。这个范围先验证工具与机制，不能排除长上下文/多请求上的不同现象。

1. 新GPU现场预检、模型/代码/依赖摘要复验；遇其他GPU进程不启动。
2. 真实模型prefill取得完整prefix KV；CPU pinned及SSD staged往返。
3. source before/host/destination/source after字节摘要一致；恢复前后32个连续生成token相同，各位置logits relative-L2≤1e-4。原始logits保存。该相同计划检查不代替并发数值检查。
4. SSD写入flush/fsync；单列读入和pinning费用；页缓存不可控，**不称冷SSD带宽测试**。
5. 每长度四臂：decode only / H2D / D2H / 双向；同等预分配状态。每方向最多一个在途16MiB tile，累计上限256MiB，传输真实捕获KV的独立镜像，不覆盖活动cache。
6. 两轮显式warm-up后10轮配对，循环/逆序平衡arm顺序。记录setup、每token host可见时刻、CUDA copy、drain、传输字节与资源保留观察上界。
7. 原始JSON/PT逐文件摘要；配对差与区间从原始数据重算，温启动不混计；不删除慢样本。

**必须区分：压力注入使用真实KV内容和真实DMA，但不是另一个真实请求的cache restore。** 因此不能从R1a声称production goodput、实际native allocator写回债务或系统优于强基线。若GPU没显示干扰，也不能仅凭512/2048、batch1宣判整个问题不存在。

R1a产物：`cpu_preflight.json`、`runtime.json`、四个roundtrip正确性报告、原始logits、96份decode记录（其中16份warm-up）、`evidence_manifest.json`、配对分析、失败/超时记录。全部记录新代码摘要，不借用旧ProbeKV SHA当新实现SHA。

### R1b：原生服务问题存在性与最强简单基线

只有R1a无损与模型回归通过，再固定一个实际可安装的现代vLLM版本及原生offload接口；独立目录/环境，不沿用旧定制引擎作现代能力证明。不重写attention或模型decoder。

接入真实prefix lookup/import/export、调度迭代、实际allocator引用/完成事件；native GPU cache命中、CPU恢复、SSD staged恢复、超时/取消都须有执行证据。

在一个进程内先让A持续decode，再提交确实需restore的B，同时可选写回C。必须观察真实block保留/归还，不能以R1a影子DMA代替。2K/8K/16K/32K与active decode1/4/8只选容量允许的预注册稀疏点；相同模型、输入、初始缓存预算。

强对照：native GPU prefix、native异步offload、充分调优固定配额、读优先、二次访问写回。不能靠同步/无界弱baseline制造优势。逐请求TTFT、ITL和含drain的全trace成本都记录。

若现代原生接口必须大改decoder才能控制、只有极端压力下有效、或充分调优的简单策略已解决问题：停止开发联合控制器，保留负结果；不自动切换其他研究题目。

### R2：有证据才接入一个联合控制器

输入：active decode/context桶、TTFT等待年龄、实际读写在途状态、GPU与pinned保留量、可用容量、generation。

输出仅两个：下一批restore tile配额；可选写回现在提交/短暂延迟/放弃未提交部分。LRU固定不另做新淘汰算法。经验干扰表缺测量时不得填0，保持保守固定配额或延期可选写回。

每次原生迭代：回收完成事件→读取新快照→让decode推进→选择有依据的有界读批→评估写回保留成本→原子获取所有权后提交。不能把已提交DMA突然改成覆盖同一目的页的重算。

四臂：00基础、10仅干扰控制、01仅写回保留控制、11联合。与最强简单baseline比较，而不只比较00。最多两轮独立开发集修订，不无限调参。

### R3：真实trace与论文可用性

两模型、至少两类自然exact prefix复用工作负载；按会话/文档隔离fit/test。真实文本＋合成到达率必须明确标记。冷启动、稳态、工作集变化、低复用率都报告。

相同GPU/CPU/SSD总预算；包括驻留cache、pinned staging、排队与在途保留。拒绝、失败、超时全部留在分母；不能免费化writeback或丢弃慢请求提高goodput。

如果最终论文声称消费卡能力，在独立支持5090的软件栈上执行，而不以A800结果代替。

## 4. 核心收益和停手标准

R1a只给机制测量，不给系统GO。R1b要证明真实正常负载下可重复存在干扰/资源保留成本及简单策略尚未解决的空间。R2/R3内部投入门槛建议：两类正常负载相对最强可比baseline约10% goodput增益且配对区间不跨0，无竞争额外成本≤3%；它们必须在正式跑结果前另行确认冻结，不是论文录用保证。

新创新仍需对照Tutti、Cascade、CacheFlow等已存在机制；“精确cache”“异步”“写回”“联合”都不是独立新颖性证据。可争取的是可验证的特定干扰/保留成本模型与可部署控制器，而不是组合已有模块后自称首创。SCI/JCR Q2与中科院二区不同，不承诺保底发表。

## 5. 已实现与明确未实现

| 模块 | 当前批次范围 |
| --- | --- |
| `exact_key.py` | 完整prefix链与模型/tenant/layout绑定，CPU测试 |
| `ownership.py` | 在途取消/隔离/两阶段释放/byte-ns，CPU测试；尚非native allocator |
| `evidence.py` | TTFT/ITL/goodput计算，缺失/失败不冒充通过 |
| `audit.py`、`plan.py` | 无卡资产摘要、不可静默修改的R1a清单 |
| `gpu_pilot.py` | 真实模型往返、受控DMA＋decode入口，GPU尚未验证 |
| `analysis.py` | 摘要核对、原始配对分析；不输出production GO |
| 现代vLLM adapter、实际并发restore/writeback、联合controller | **未实现；R1a之后按阶段推进** |
| 正式trace、独立test、论文性能与创新成立 | **未验证** |

## 6. 明天执行与预算

先由用户确认原服务器已挂卡。启动必须显式`--execute-gpu --max-seconds <本批授权时长>`，输出目录必须不存在；超时保留成功前缀与失败标记，脚本不关机、不续租、不付款。今天不运行此命令。

```bash
PY=/root/autodl-tmp/probekv_stage1/envs/cacheblend-cu121/bin/python
cd <本批唯一code目录>
$PY scripts/pacekv/run_pilot.py --preflight <cpu-preflight目录> \
  --output <新GPU结果目录> --execute-gpu --max-seconds <用户授权秒数>
$PY scripts/pacekv/analyze_pilot.py --input <新GPU结果目录> \
  --output <新分析文件.json>
```

约几十分钟至一小时只是R1a待现场核验的估算，不是承诺或默认GPU授权。R1b原生集成预计还需数个工作日；1–2周内作本方向第一轮GO/NO-GO。完整论文级验证仍按数周规划，不能明天就宣布系统收敛。

无卡交接应分别写：CPU测试通过、资产预检通过、可尝试R1a、GPU正确性未验证、原生服务未接入、系统收益未验证。缺资产则保持BLOCKED，不写假SHA。

## 7. 参考

- NVIDIA RTX5090官方规格：https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- vLLM原生offload说明：https://docs.vllm.ai/en/stable/features/kv_offloading_usage/
- 详细创新重叠与研究依据见同日原始提案；本文件不将最新文档接口等同服务器已安装能力。
