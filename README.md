# Prefix Cache research source snapshot

本仓库与 majic-mk/PageShare 并列，是独立的公共仓库，代码直接保存到默认 main。

- prefix_io_v1/：服务器当前项目源码、作者 py-kvcache/vLLM 副本、历史增量修改、测试、配置、脚本、交接文档与本轮报告。原服务器项目根目录在此目录对应。
- workspace/：本地研究工作区的已跟踪源码、未提交源码修改、补充测试与 Prefix I/O 开发候选快照，保留原路径。
- SOURCE_MANIFEST.json：逐文件原始字节数、SHA-256 和来源，供备份核对。
- SOURCE_EXPORT_INFO.json：范围、排除项和真实实验结论。

当前方法代码位于 prefix_io_v1/artifacts/prefix_io_v1/server14-dr-long-20261005/。其中 runtime_source_v3/native/py_kvcache 是本轮两臂共同使用的原生私有源码；runtime_source_v2/control/prefix_io_control 保存 D_R 的有限依赖排序、观测与恢复预测组件；dr_p316_original_adapter_v5.py 沿用原 P316 runner。

## 最新真实模型结果

同一 Qwen2.5-7B-Instruct/BF16、RTX 5090，原 mixed_readwrite 10 请求，每条最终输入 16257 tokens、完整输出 128 tokens。相邻单对 U04 → F8+D_R01：

| 指标 | U | F8+D_R |
|---|---:|---:|
| 整批完成秒 | 28.295357870 | 33.101601666 |
| 实际 SSD 改序 | 0 | 0 |
| 完整输出 | 1280 tokens | 1280 tokens，全相同 |

组合本次耗时增加 16.98598%。D_R 没有形成可执行排序机会，分类 NOT_EXERCISED；不能宣称依赖排序带来提升，不能把单对组合退化归因于有效 D_R 改序。原生 I/O、模型 shutdown 与 OS 会话均正常闭合。没有显著性、生产 SLO 或完整 D/I/J 性能声称。

完整报告：prefix_io_v1/artifacts/prefix_io_v1/server14-dr-long-20261005/SERVER14_U_F8_DR_FINAL_REPORT_02.md。实际原始配对分析和逐请求/机制补充记录在同目录 ACTUAL_MAIN_PAIR_ANALYSIS_01.json 和 ACTUAL_MAIN_PAIR_MECHANISM_LATENCY_SUPPLEMENT_01.json。本仓库保存报告内容中的原始证据引用；历史大轨迹和运行资产留在原服务器/本地备份。

## 使用范围

这是源码和配置备份，没有模型权重、KV 缓存、虚拟环境、CUDA 私有 SDK 二进制或临时编译产物。依赖版本、资产路径和冻结源锁保留；换环境后需按已有交接文档准备依赖与授权，不能把旧 GPU 权限、历史资格或机器绑定锁直接当作新机器资格。上传过程没有启动 GPU 实验，也没有改动原服务器或本地研究源码。

复制的作者源码、LICENSE/NOTICE 和归属说明保留在各 third_party 目录。本仓库不重新授予作者代码许可证。源码整理由 Codex 辅助完成。

## 克隆

git clone https://github.com/majic-mk/prefix-cache.git

浏览源码和克隆无需登录。SOURCE_EXPORT_INFO.json 保留首次上传时的私有状态，当前仓库已公开。进入 prefix_io_v1 后按 docs/prefix_io_v1/04_CODEX_EXECUTION.md 与已有阶段文档阅读；模型和执行器沿用作者实现。
