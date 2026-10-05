# 轻量前文 metadata 辅助 Source 选择

本策略是默认关闭的独立重排，不是 CFO、不使用历史 attention，不是质量概率或资格证明。
不修改修复算法、来源传播规则、候选读取集合、Source freeze、成本或资源准入。
原有冻结实验清单不自动启用它；比较此策略需要另行绑定配置与运行代码摘要。

## 实现入口

- `src/probekv/source_metadata_selection.py`：纯 CPU metadata 合同、共享引用、评分和重排。
- `ComparisonSessionV2(..., metadata_policy=...)`：在真实 K 比较结束、比较凭据签发前接入。
- `compare_native(..., current_prefix=..., historical_prefixes=...)`：提供本次前文及按 Source ID 映射的出生前文。
- 独立 `select_source` 只接收上游计算的状态差异，因而也适用于显式 K+V 消融；不自行读取 V。

未提供策略时沿用原 K-only 计算和确定性排序。已冻结 P0 调用方不变，GPU 实验不自动启动。

## 输入与完整性

`PrefixOccurrence` 保存 `content_key / role / token_count / order_index`，附加半开
`token_start / token_end` 来验证覆盖和不重叠。`PrefixManifest` 保存 tokenizer 签名、
目标起点、不可变 occurrence tuple 及 complete 标记。区间是原请求 token 行号。

前文必须完整覆盖 `[0, target_start)`：系统提示、指令、文档、分隔符都要计入，
目标自身、后续文本、padding 不得计入。角色与区间由请求 renderer 提供，不能靠
猜测或把未标注文本全当 document。首个目标前文真正为空可用空 tuple 与起点 0 表达。

`build_prefix_manifest` 从实际 token 切片构造精确内容摘要，最终对象不保留 token IDs。
相同 tokenizer 命名空间下同一精确内容的长度必须一致。角色默认精确相等才兼容；
可显式配置互不相交的角色等价组，不存在隐含的 system/document 通配兼容。

缺失 manifest 或覆盖有缺口返回 `valid=false`、`metadata_score=null`。
重叠、错误顺序、长度矛盾、越过目标、非法数字等抛出 `MetadataError`，不产生有效分数。
`complete=false` 即使区间看起来完整，也不能获得有效分数。

`SharedPrefixManifests` 是带字节预算的独立 sidecar；一次注册，多 Source 引用同一
内容寻址 manifest ID。支持 payload 导出/导入与摘要核验，不改变旧 Source catalog。
旧 manifest 没有完整角色分区时保留缺失，不能从历史前文 KV 或 attention 重建。
这里的字节预算计序列化 payload，不声称包含所有 Python 容器开销。

原生比较接入点还将 metadata 内容摘要和目标起点与已有共享出生输入 manifest、
当前真实请求核对。它只读取输入 metadata，不新增历史前文 KV 或 Prefix shadow。

## 分数

相同内容与兼容角色按出现顺序一一配对，额外 occurrence 不匹配。

`O = W_match / (L_old + L_new - W_match)`。

共同项按历史顺序排列；`Gamma` 是 `n_i*n_j` 加权逆序对之和除以所有共同项配对的权重和。
实现使用整数 Fenwick 累加，复杂度 `O(m log m)`，最后才转为比例。

`P = O * (1 - Gamma)`。

- 两边真实空：P=1，`both_prefixes_empty=true`。
- 只有一边空或无共同项：P=0。
- 共同项少于两个：Gamma=0。
- 不将缺失 metadata 当空前文或 P=0。

结果保留 valid、reason、L_old、L_new、W_match、matched_count、overlap、order_penalty、metadata_score。

## 选择规则与开关

```python
policy = MetadataSelectionPolicy(
    enabled=True,
    tau=config["metadata_state_tau"],
    delta=config["metadata_state_delta"],
)
```

没有内置 tau/delta；开启时两者都必须显式且有限非负。原生比较的 tau 必须等于
已绑定 ComparisonProfile 的 tau_reuse，不能另设更宽阈值绕过状态准入。

1. 保留其他已知准入条件通过且 D<=tau 的 E。
2. 构造 B：D<=min(E.D)+delta。metadata 不提前剪掉比较候选。
3. 单候选不读 metadata；多候选全部有效才按 P 降序选 winner。
4. 任一 B 成员 metadata 缺失/不完整，整个 B 回退原状态排序。
5. P 相同保留原状态排序；v2 为 D、generation、publication_epoch、source_id。
6. E 为空沿用原回退逻辑。v2 诊断凭据仍可保留原始 best 供 mismatch 审计；
   原生 preparation 继续根据实际 winner 的 D 拒绝不兼容来源。

纯策略函数的 `StateCandidate.admission_eligible` 可以表达已知的其他前置条件。
这不是成本凭据；输出始终标记 `requires_selected_source_admission=true`。
原生路径仍对重排后的实际 Source 获取租约、申请空间并检查最终准入，不能借用
原 winner 的成本或资源凭据。freeze 后失败不得改选第二名。

## 审计与边界

比较凭据增加不可变 `metadata_selection`，保存原 winner、新 winner、策略配置、
近似集合、分数及 manifest 引用。原始 K 分数、比较数量和 Source 签名仍保留。
metadata 用时包含在原比较 host wall-clock 内。策略在观察后改变会使凭据失效。
新模块加入 runtime binding digest；旧 GPU 结果不能为新代码自动授予资格。

未接入角色 metadata 的旧 runner 仍默认 K-only。仅打开开关而不提供完整 metadata，
会得到审计清楚的原选择回退，而不是假装本策略成功。尚未宣称 QA 或 TTFT 改善。

## CPU 验证

```powershell
$env:PYTHONPATH="$PWD/src"
python -m unittest tests.test_source_metadata_selection tests.test_source_metadata_integration -v
```

覆盖用户 1/3 示例、长度加权逆序、重复 occurrence、空/缺失/不完整、区间冲突、
角色兼容、共享引用与损坏、配置边界、稳定 tie-break、真实 store 比较后的 winner
重排、无额外 K 投影/完整 KV/lease/LRU 副作用，以及身份/配置变更拒绝。

2026-09-21 本地验证：新增 23 项全部通过；全量 unittest 1664 项中
1663 通过、1 项跳过，耗时 91.531 秒；compileall、合同 validator、git diff --check 通过。
这些结果只证明本地接口、算术和回归，不证明真实模型选择质量或 GPU 性能提升。
