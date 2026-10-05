# P1 成本曲线接口复核（CPU 只读）

日期：2026-09-27（Asia/Shanghai）。固定版本：py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e；kvcache-experiments 0e023a84a21246b9bbc06266fa8070397eccbdc9。路径均相对服务器项目 /root/autodl-tmp/prefix-io-v1-handoff/project。

## 结论与实际范围

固定仓库中没有现成 v2 转换器。原 emit_break_even.py 输出旧版标量阈值，当前 LoadPlanner 需要三条成本曲线及实际 KV 几何。已有 GPU Prefix smoke 不能变成这些曲线的测量数据。

本轮仅准备 CPU 候选任务、核验历史资格身份及格式边界。不创建曲线，不新增插值算法，不启动模型，不更改缓存引擎，不进入 P3。实际 io_uring 重查仍为 EPERM；完整 P1 未通过。

## 对应源码与证据边界

| 位置 | 核实事实 | 使用限制 |
|---|---|---|
| third_party/work/py-kvcache/py_kvcache/vllm.py:630,663–739 | _build_planner 要求 curves、真实 Plan API、预加载与共享 staging、正 lookahead、kv_bytes_per_token、max_model_len | model_name 取运行模型字符串；本地路径不能任意替换为仓库 ID |
| third_party/work/py-kvcache/py_kvcache/break_even.py:103–118,247–279 | 原 loader 允许缺失 model_name，dtype 不符只警告；根据 curves 判断 v2；golden 可空；KV 字节只检查正整数 | 原生解析成功不足以证明身份、真实 BF16 几何或测量来源 |
| third_party/work/py-kvcache/py_kvcache/cost_model.py:72–92,119–125 | golden 是 PCHIP 计算一致性自检；表外范围只警告 | 即使自造数据自洽，也不构成真实标定；空 golden 不产生证据 |
| third_party/upstream/kvcache-experiments/scripts/emit_break_even.py:65–80,122–128 | 输出模型/dtype/两个阈值/margin，未输出 curves/golden/KV 字节；g_mem 从 SSD 曲线与带宽模型推导 | 不可把旧表改版本号或把推导 g_mem 称作 RAM 实测 |
| third_party/upstream/kvcache-experiments/scripts/pareto_measure.py:114–162 | 可独立复用原 Job 和 build_job_plan；COLD_ONLY 早退只产生 baseline cold_prefill | 计划不等于执行；未调用 main/start_server/run_benchmark |
| third_party/upstream/kvcache-experiments/plots/pareto_plot.py:200–279 | 原解析筛选成功、串行的 cold-prefill 与实际 LOAD 请求 | 原始失败必须另保留，不能仅保存成功过滤后的结果 |
| third_party/upstream/kvcache-experiments/scripts/emit_break_even.py:83–91 | 无已求出交点时返回0 | 不可把0直接解释为“缓存始终有收益”；当前无数据，不在本轮修改算法 |

## 已准备候选与未完成条件

候选计划准备器只复用固定 SHA 的原生任务生成函数。名义 doc-size 来自原作者按词构造的参数，并非实际 token 长度；候选值低于旧 max_model_len 也不能证明请求一定容纳，执行前仍须补精确 token 适配与计数。

f(N) 的原生冷 prefill 在机制上不需要 SSD/io_uring，但当前原 launcher 建立新 session，且原驱动的预热、文本构造、离线模型选择与启动参数未形成可执行冻结配置。本轮没有为孤立 f(N) 消耗 GPU 预算。

完整 v2 还须真实、同模型/设备/精度/布局条件下的 cold-prefill 与缓存路径数据、实际 I/O 字节、原始 CSV/请求/trace、带宽与变换来源，以及有审查记录的导出过程。不能把原 smoke 墙钟转换为 f(N)，不能套用其他模型或显卡的曲线，不能跳过 LoadPlanner 强制准入。

下一允许阶段仍是 P1：先取得可用的原生 io_uring 环境，验证作者 ring 与实际 I/O，然后完成该环境上的原生校准与 staging/SSD/生产 KV 集成资格。候选计划永远不单独授权 GPU 运行。
