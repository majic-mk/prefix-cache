# 本包参考测试

这些脚本只执行独立的Python规则，不导入ProbeKV、PyTorch、vLLM，不读取模型、不调用GPU、不修改远端。

`provenance_reference.py`实现来源代数、目标全层执行账本、入库原因与快照可见性的抽象规则。实际引擎必须提供真实账本、输入依赖与资源证据。该参考不能证明混合来源的数值误差或任务质量受控。

`test_provenance_reference.py`有24项规则测试，包含正例和反例。

`validate_package.py`检查JSON契约、阶段依赖、文档结构、关键不变量，并对若干错误契约做拒绝测试。它不是仓库的contract validator。

Python 3.10及以上、无第三方依赖：

```bash
python -m unittest discover -s reference -p 'test_*.py' -v
python reference/validate_package.py
```

第二条命令会生成/覆盖包根目录的`07_本包校验记录.json`。重新执行后文件hash可能因Python环境等信息变化；发行版checksum只对应随包发出的原文件。

主模型质量、近似传播幅度、真实Source提取、额外IO、BF16/GQA/RoPE内核和TTFT仍待Codex在真实工程环境执行。
