# 自然 512-token 输入的 P0 上机准备

## 范围

新增的是 P0 诊断桥接，不是 P1 QA runner。先验证已冻结自然输入的捕获与独立 Source
发布链路。每个配方严格三个动作：capture-off、capture-on、普通完整请求出生。
两个参照 forward 不入池；出生请求不使用 teacher、不为构建重复生成完整答案，只生成
一个 token 后走原有发布流程。所有动作成本分别保留，不能当生产 TTFT。

`p0_natural_input_v2.py` 调用已有 `build_controlled_p0_manifest` 和原生 P0 runner，
没有新增绕过准入的执行器。每个配方必须使用全新空的隔离池，不能批量叠加历史 Source
后继续假设候选为空。metadata 辅助重排本轮关闭。

## CPU 准备与服务器绑定

1. `prepare_decoupled_v2_natural_p0.py` 读取冻结文件索引的外部 SHA，重新核验
   原请求、历史顺序、完整 arm、分区与 Source 构建依赖。spec 显式提供 identity、build_id、
   teacher_token_ids（31个，对应32个预测位置）、comparison_profile、upper_seconds_by_arm。
2. 保持原 prompt tokens、目标位置、全部 QA 文本不变。P0 明确覆盖 `prefetch_window=0`；
   原清单 content_key 保存在审计，实际 key 由隔离 store 的模型/tokenizer/权限域派生。
   两者不得静默当成同一命名空间。诊断生成长度32、普通出生长度1均记录。
3. 输出 controls.json 不是运行授权。服务器使用原有 init-pool，再执行：

```text
prepare_decoupled_v2_controlled_p0.py freeze-batch
  --natural-controls <controls.json>
  --recipe <仅含 authority/limits/registry_budget/numerical_policy 的文件>
  --native-manifest <当前实际附件> --native-manifest-sha256 <实际SHA>
  --store-root <新空池> --registry-root <新registry>
  --instance-id <当前实例> --output <新目录>
```

4. 原生资产/补丁/导入检查、模型词表/上下文长度、工作区哈希、资源及当前授权仍由原有
   preflight 和 runner 检查。必须采用32位置、token一致、relative-L2≤1e-4的已批准标准。
   旧授权窗口不能复用；控制文件不填GPU UUID、价格或执行成功的占位值。
5. 上卡后 raw numerical pair 与普通出生 publication 分别验收，保留失败，不覆盖旧目录。

普通出生的事后验收使用 `verify_natural_p0_birth`：核对完整出生请求 tokens、目标绝对位置、
每层完整执行行、层数、G0、独立目标存储及 KV/SelectionState 文件摘要。不仅比较目标文本，
防止同一目标在不同前文中产生的 Source 被混认。只在请求/租约全部释放后运行，不刷新LRU。
该磁盘与配方校验本身不证明真实GPU执行；必须与独立 raw batch 数值核验一起报告。

## 仍然禁止

- 不自动执行完整89次 Source构建和189次P1消费；不把该配方称为自然 workload 收益。
- `mixed_M1` 不能走 exact 编译器。其真实 parent、执行来源和配对参考必须独立绑定。
- P1 legacy fixed15消费入口、实际自然 Source 文件资格及P1独立准入未完成前，P1保持阻塞。
- CPU PASS不是新版本GPU PASS；旧9批GPU证据只保留其原runtime/输入范围。

可准备租卡做此有界P0，不等于全部P0已验收，更不等于可以启动正式P1或论文实验。
