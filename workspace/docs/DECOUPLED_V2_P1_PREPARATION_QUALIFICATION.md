# P1 准备资格消费者：原始证据、冻结数据与 Source 见证

新增 `src/probekv/p1_qualification_v2.py` 和
`scripts/qualify_decoupled_v2_p1_preparation.py`。这是只读的阶段准备消费者，不是模型执行器、GPU 授权服务或质量认证。

## 解决的缺口

已有 P0 数值验收与 P1 输入依赖解析，之前彼此独立；仅有文件不能证明当前模型、补丁、执行代码、历史出生和目标存储同时符合要求。现在统一核验：

1. 原始 P0 batch 的 manifest、记录链、原始 logits/KV 及冻结容差；要求 T20、T21、target-r1，以及 G1 发布/G2 拒绝证据。
2. 所有 batch 必须匹配显式 code/runtime/patch/model/tokenizer 身份。旧 runtime 不会隐式继承，不靠修改摘要中的 PASS 解锁。
3. 使用现有冻结输入解析器重新验证文件索引、content-group 隔离、512-token 目标、出生在消费之前、独立 S0 及配对 E/M1 配方。
4. 从静止、已存在的真实隔离池重新打开 G0/G1 见证对象，核验完整 backing/SelectionState、共享 manifest、完整历史 prompt、目标位置、逐层证明与真实发布动作。教师序列数值参照不得成为来源。
5. 检查读前读后 catalog/registry 摘要相同，不创建缺失池、不发 comparison/use 事件、不刷新 LRU。

消费者只记录独立 G0/G1 见证，**不能将它们当成全部 P1 Source 已构建**。精确与混合 Artifact 内容相同是合法零结果；来源身份不得合并洗白，也不从零差异推导质量结论。

## 避免资格与构建循环依赖

阶段分成：

```text
P0 原始数值 + G0/G1 磁盘见证 + 冻结输入图
  → BUILD_PLAN_PREPARED_EXECUTION_BLOCKED
  → 原生 P1 Source 构建 dispatch + 单独授权/preflight
  → 实际 Source build receipts
  → 对应 QA consumer 才具备输入条件
```

`build_plan_preparation_ready=true` 仅表示可以准备有界构建清单；不是已经获准启动构建，更不是 QA 通过。所有 P1 构建 ID 保留在 `source_builds_pending`，不会因为有一对 P0 见证就假装全部完成。

以下输出始终 false：`P1_execution_allowed`、`GPU_execution_allowed`、`P1_QA_execution_allowed`、`source_artifacts_for_P1_verified`、`paper_evidence`。返回码 0 只表示准备检查通过。

## 命令与契约

```text
python scripts/qualify_decoupled_v2_p1_preparation.py
  --contract /absolute/path/qualification_contract.json
  --contract-sha256 <外部冻结的文件SHA256>
  --output /new/path/preparation_report.json
```

契约 `kind=P1_preparation_evidence_v2` 包含：

- `runtime_identity`：code_commit、runtime_digest、patch_sha256、model_signature、tokenizer_hash。
- `prefix_mode=off`、`locked_test_accessed=false`；Prefix-on 不能继承本次资格。
- `p0_batches`：各 batch 的绝对路径、manifest 文件 SHA、完整原 binding。
- `frozen_inputs`：冻结根目录、外部文件索引 SHA、模型与两种 tokenizer 身份，沿用 `load_frozen_input_graph` 接口。
- `expected_input_graph_sha256`：已冻结的逻辑依赖图摘要。
- `p0_source_witnesses`：batch/action/target/Source 身份以及 catalog、registry、pool_spec 的路径和外部 SHA。

输出文件禁止覆盖。模型执行 runtime digest 不因为增加只读消费者而改变；消费者自身文件 SHA 单列。不能将这个约定用于隐瞒实际执行文件的变更。

## 边界与下一步

历史 `p0_stage_readiness_v2.py` 的 P1 入口仍失败封闭；本次没有删除阻塞项来冒充原生 P1 QA dispatch 已完成。新入口给出了具体准备项证据，后续需要接通：

1. 固定 legacy normalized K/V15、共同 first reuse layer=9 的真实 P1 dispatch。
2. 按冻结清单构建历史/S0/E/M1，逐对象验证实际回执与未来请求可见性。
3. 新阶段授权、设备/代码/环境 preflight、失败保留与计费；旧 P0 时限不是 P1 授权。

CPU 模拟/文件测试仅检验上述校验逻辑；已有真实 P0 数据的只读重验不算新增 GPU 动作，也不算新的 QA 样本。

### 后续构建接线

已新增 `p1_build_dispatch_v2.py` 的独立 Source 构建操作，见
`DECOUPLED_V2_P1_BUILD_DISPATCH.md`。这不是完整 P1 批次启动器或固定 Source QA 入口；准备报告的执行阻塞仍然保留。新包装器具有独立文件摘要，不从旧 P0 自动继承 GPU 资格。
