# P1 Source 构建接线：执行原语与批次启动分离

## 本次实现

`p1_build_dispatch_v2.py` 新增：

- `compile_p1_build`：重新解析冻结输入，按 build ID 编译一个出生动作；保留全部原始 prompt token 和位置，记录 request ID、max_new_tokens=1、prefetch_window=0 及 content-key namespace 适配。只减少出生阶段的解码，不截断输入，不另跑 capture forward。
- `validate_build_context`：在模型前向前检查请求、运行文件摘要、实际池和快照。精确出生必须使用独立空池；M1 出生池只能包含已验证的 G0 父对象。父回执自身 hash 不够，实际 backing 的共享 manifest 必须与冻结父出生完整输入相符。
- `execute_p1_source_build`：精确来源调用已有 `execute_p0_request` 的 cold-miss 全量 capture；M1 调用已有 `execute_lineage_birth`。这是直接调用原生执行路径，不接受外部 passed 回调。执行后要求实际发布、完整 backing、正确 G、目标位置及清理回执；失败保留原 audit，不补跑。

## 构建隔离

每个精确出生使用独立的实验池。M1 池先构建其 G0 parent，后构建 M1；配对 E 单独构建。四个历史来源和 S0 是五个实验 arms，不将 S0 当作第五个在线版本，也不提高线上 K=4 上限。

目前复用已有 `TargetSourceStoreV2` 的隔离诊断格式（历史 purpose 字段仍为 `isolated_P0_diagnostic`），不将其重标为生产池，不把参考构建插入在线命中统计。实验队列/角色由外部 P1 操作及 build receipt 明确绑定。

读取与验证已构建对象的时间应计入外层动作总耗时；内部 request TTFT 不包含这段 post-request 验证，两者不得混用。下游消费需要独立重建/加载相应实验 Source，不能依赖前一个 arm 的可变工作状态。

## 数值资格不自动继承

新包装器使用两段身份：

```text
base_runtime_digest             已有 P0 模型/执行依赖
build_dispatch_sha256           本次新执行包装器
```

底层依赖没有修改不表示新包装器已经有 GPU 证据。GPU preflight 和操作日志必须绑定两者；不能只记录旧 base digest 后声称新入口已验证。

CPU 测试验证输入、路由、失败与回执合同；其中模拟 forward 不是模型数值结果。编译出的 89 个操作也不表示产生了 89 个 Artifact。

## 后续接线进度（2026-09-22）

1. 已增加独立 P1 单动作入口和固定指定 Source QA 消费代码，详见 `DECOUPLED_V2_P1_QA_ENTRY.md`。不是自动运行完整矩阵的批次授权。
2. 仍需新包装器的 GPU 小批确认、实际构建回执及 QA 证据收集；不能用 CPU 测试继承 GPU 资格。
3. d1/d2/完整 checkpoint 选择质量的独立观察与 QA oracle 聚合仍未执行。

因此本次保持 `P1_execution_allowed=false`。旧准备消费者的 `BOUND_NATIVE_P1_DISPATCH` 阻塞不能仅因为新增本文件就删除。
