# 独立 layer3 边界诊断（2026-09-22）

## 授权与解释边界

用户在明确“当前 P1 为 layer9、目标主路径为 d2/layer3”的差别后要求继续。
因此新增有界的边界敏感性诊断，不改写首次交接任务书、冻结 P1-E/P1-M
或 P4 晋级规则。不是完成 P4，不冻结 Profile，不进入生产准入、增长或多段实验。

只使用已完成六路对照的第一个冻结 fit 请求及其四个历史 Source、独立 S0。
该请求的 layer9 输出已经曝光，本补充是探索性配对诊断，不是新 held-out 证据。
不挑新样本、不调阈值、不改答案/scorer、不降低 repair ratio，不重建 Source。

## 固定动作（最多九个消费动作）

1. 同输入、无 Prefix 的 dense teacher，32 个预测位置。
2. 历史 Source1，d2 完整后 layer3 开始 r=1，使用相同 teacher inputs。
3. dense 自由 greedy QA，原 32-token 上限及停止规则。
4. 历史 Source1，layer3 r=1 自由 greedy QA。
5–9. 历史 Source1/2/3/4/S0，各一次 layer3 fixed15 K/V QA。

teacher 输入固定为原 prompt 最后 31 个 token；不是标准答案，也不根据模型
输出选择。它只用于条件一致的数值检查，不计入 QA。全部输入及动作在启动前
生成并摘要冻结。Source1 是原清单第一项，不由分数或答案选定。
teacher 数值请求不携带自由 QA 的答案停止契约；所有自由 QA 请求原契约保持不变。

第 2 项必须满足每个位置 logits relative-L2 <= 1e-4、32 个预测 token 相同；
第 4 项必须与第 3 项自由生成 token IDs 相同，才能执行第 5–9 项。
任一执行、身份、数值或资源检查失败，保留失败并停止后续 QA；不放宽容差。

## 对齐与隔离

- 每次使用新的请求上下文、独立复制的静止 Source store，只能看指定 Source。
- 沿用已验证的模型、patch、底层 executor 和 legacy normalized K/V mask 公式。
- `completed_depth=2`，观察 block3 投影，在 block3 attention 前提交；不假称
  block3 QKV 投影免费或整层均已稀疏。
- 全部32层记录实际 active-before/after；层1–2完整，fixed15 从层3起目标保留
  ceil(512*0.15)=77 行，非目标和 suffix 保留。r=1 全部行持续完整。
- 只有独立诊断入口可做规定 Source 的受控提交；保留物理 lease、HBM、ready、
  Source/destination integrity 检查。生产 FinalCommit 不被绕过或宣称通过。
- 2 GiB comparison、6.875 GiB HBM 池、4 GiB safety，沿用已批准预算。
- layer9 原始证据与代码保持不变；新入口绑定独立文件 SHA 和旧 runtime digest。
- 源构建、layer9 输出可读复用；layer3 执行和数值结果必须新测。所有旧 CPU/GPU
  证据保持原范围，新 CPU 测试不能冒充 GPU 通过。
- 单动作最多 180 秒计算、600 秒初始化、120 秒清理；失败不自动继续。
  用户取消总时长限制不等于无限扩展矩阵。费用未知保留 null，不自行关实例。

## 输出与停止

输出 raw logits/token IDs、层账本、comparison receipt、mask、Source before /
destination / after digest、资源释放、实际时间、输入/代码/动作/模型摘要。
先完成数值门，再完成五 Source QA，最后只报告这一请求在 layer3/layer9 下
是否同答、F1、分数及 mask 差异。不能据一题宣称多 Source 有用/无用、d1/d2
选得准或端到端加速。不同边界原始数值和时延不合并成性能样本。
