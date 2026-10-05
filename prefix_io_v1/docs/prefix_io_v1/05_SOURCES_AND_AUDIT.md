# 05 来源与本轮核验边界

核验日期 2026-09-25。

## 1. 证据口径

本交接包以本次对话已确定的设计边界为需求来源，并使用官方代码／文档核对底座接口。01—04 中新增策略、候选上限、实验门槛与模块名称都是实施建议，不代表已有系统已经提供或本项目已经实现。

本轮进行了静态代码阅读，没有修改用户仓库，没有编译作者分支，没有运行GPU，没有验证两仓库完整兼容，也没有完成全部相关论文和代码的穷尽审计。

本轮不会将前文“已证明系统的优点”扩大为“当前主分支每一个新增功能都已经发表并被独立验证”。仓库当前能力、论文原版本和本地实测分别记录。

## 2. 已核验的主要来源

### [S1] py-kvcache README

- 固定提交 `3abba7a502d553f6e7e2e58b92086487e3395d7e`。
- [官方文件](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/README.md)
- 支持的事实为外部Prefix缓存、pinned staging、异步I/O、共享预加载、CPU cache、break-even和load/defer/recompute规划。
- 同时明确依赖作者vLLM分支与profiler，磁盘文件没有完整GC／淘汰。

### [S2] 作者 vLLM 分支集合

- [官方分支API](https://api.github.com/repos/t348575/vllm/branches?per_page=100)
- 本轮返回的 `preload` SHA 为 `d6eadf416bb5234047760bf55d532f2f038cf697`。
- `main`、`with_profiling`、`extended_profiling` 同时存在，不能把任意分支名当作与py-kvcache天然兼容。
- 该SHA是候选检出，不是通过测试的环境锁。

### [S3] kvcache-experiments

- [作者README](https://github.com/t348575/kvcache-experiments/blob/master/README.md)
- [仓库](https://github.com/t348575/kvcache-experiments)
- 本轮读取README，包含现成bench驱动、prefix replay、数据集replay、pareto_measure与break-even导出路径。
- README中的原始示例包含单输出token，适合相关校准，不能替代持续decode研究。
- 执行前由P0冻结整个仓库commit。README blob SHA不能代替仓库commit。
- 不直接使用其带清空目录效果的参数，除非已限制为批准的实验专属目录。

### [S4] 作者分支 OffloadingConnectorWorker

- [固定版本 worker.py](https://github.com/t348575/vllm/blob/d6eadf416bb5234047760bf55d532f2f038cf697/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py)
- 本轮读取初始化／布局注册，以及 `_submit_store_jobs`、`handle_preemptions`、`start_kv_transfers`、`prepare_store_kv`、`get_finished` 等路径。
- 关键事实为store按既有时序延期提交、提交成功断言、`jobs_to_flush`的wait、completed_jobs汇总和部分失败断言。
- 完整scheduler侧refcount/fence/allocator关系仍须P0在同一候选SHA核验，不能由worker单文件推断全部释放条件。

### [S5] py-kvcache reactor

- [固定版本 reactor.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/reactor.py)
- 依据本对话已读取的 `_pump_once`、共享slot、completion、`_schedule_work`、`_on_store_copy_done`、父任务汇总与TransferCoordinator路径。
- 支持异步多级执行、copy headroom、父子任务汇总、共享引用和读／写／预加载排序等事实。
- 函数名用作P0定位锚点，不能以旧行号替代实际checkout核对。

### [S6] py-kvcache 成本规划器

- [固定版本 load_planner.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/load_planner.py)
- 本对话已读取规划器输入、ADMIT／DEFER／DECLINE、待写入状态、预加载容量与等待截止逻辑。
- 本轮方案明确继承它，不声称现有规划器已经考虑本方案新增的所有资源释放依赖。

### [S7] py-kvcache vLLM 适配

- [固定版本 vllm.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/vllm.py)
- 本轮读取import fallback、`VLLM_AVAILABLE`、`PLAN_API_AVAILABLE`及单KV group检查。
- CPU单元测试能够导入不代表实际vLLM兼容，这一差别须写入验收。

### [S8] StagingPool

- [固定版本 staging.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/staging.py)
- 本轮读取 `compute_slot_count` 与slot申请／释放，最低slot数量可能使分配超出名义配置。
- 源代码无额外检查不自动等于运行缺陷，是否可能重复释放需要事件测试确认，不能仅凭缺少断言声称已有bug。

### [S9] profiler 包装和依赖清单

- [profiling.py](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/py_kvcache/profiling.py)
- [pyproject.toml](https://github.com/atlarge-research/py-kvcache/blob/3abba7a502d553f6e7e2e58b92086487e3395d7e/pyproject.toml)
- 本轮核对可关闭profiler包装及包定义。作者vLLM fork另有直接profiler import，不能只凭py-kvcache的noop后备认定整个组合不需要该依赖。

### [S10] 官方执行语义与安装说明

- [vLLM KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- [vLLM GPU安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- 用于核对现代offload路径、异步传输、页锁定内存及版本匹配要求。latest文档不自动适用于已锁作者fork，接口以P0实际代码为准。
- 不使用论坛AI回答作为兼容性证据。

### [S11] 模型官方配置

- [Qwen2.5-7B-Instruct config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [Mistral-7B-Instruct-v0.3 config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json)
- Qwen配置给出28层、4个KV头、hidden 3584、28个attention头，未启用sliding window。
- Mistral配置给出32层、8个KV头、hidden 4096、32个attention头，sliding_window为null。
- 这些支持理论KV字节估算，不构成GPU实际容量、模型下载许可或后端兼容证明。运行时还要冻结模型revision。

## 3. 审计到的风险与计划处理

| 风险 | V1处理 |
|---|---|
| 作者fork分支与py-kvcache是否完全匹配 | 候选版本明确标未验证，P0/P1跑真实集成 |
| import静默降级stub | 验证实际符号、能力和handler路径 |
| staging名义预算不等于实际分配 | 实际字节检查，所有实验臂一致 |
| GPU源块释放可能等parent完成 | 使用原协议闭包，不提前回报complete_store |
| 下游原本自动续接 | 保留续接，不为新策略制造额外停顿 |
| 已提交SSD操作不可任意重排 | 只调整尚未签发工作与后续压力 |
| shared cache可驱逐 | 优先原生无I/O回收，不虚构写回债务 |
| 磁盘没有完整淘汰 | 首版磁盘可容纳固定trace，容量guard，不声称SSD压力覆盖 |
| 调度器等待但策略又等新调度 | 加限流前先接通必要排空信号与测试 |
| 原失败协议较保守 | fail-closed报告，不虚称透明恢复 |
| token间隔与网络chunk混淆 | 按真实事件统计，缺失则显式标记 |
| 旧Source流水线与新目标不一致 | 只复用通用经验／测试，不导入旧执行语义 |

## 4. 尚未完成

没有下载或编译全部依赖，没有运行其GPU测试，没有确认目标服务器、GPU驱动、存储配置及现有本地修改，没有形成成本表、SLO数值或性能结果。

本轮未重新开展相关论文全量审计。实施完成后若进入论文写作，需要按实际算法与冻结代码补充新颖性比较，不能把本交接包当成“已经确认无人做过”的证明。
