本轮已完成服务器上的独立 raw 分词工具和输入合同准备，状态为 PASS_CPU_TOOLS_AND_SOURCE_ONLY_ASSETS_UNBOUND_REAL_INPUTS。尚未选择、解析或分词任何真实自然数据，也没有生成正式 family receipt、manifest 或 GPU 配置。

实际新增源码位于服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-natural-input-cpu-20261005/tokenizer_producer/raw_tokenizer_cpu.py；SHA-256为 2d252d0647f6a1617a4222fd86009da27344d29391c109fa3ccecf7cd4f68c63。工具使用原作者完整接受顺序和现有本地 Qwen tokenizer，不应用 chat template、不截尾、不挑性能有利的请求。它只产 raw token IDs，不生成 prefix_family、split、arrival 或 deadline。

本轮修复了默认 Python loader 可能读取未冻结 .pyc 的源码闭合缺口。协议和 tokenizers Python 模块改为直接从核验源字节执行；C扩展仍使用原已核验库。已有缓存文件、已安装包、作者代码和旧锁均未修改或删除。3项新字节码缓存回归测试包含在最终24项测试中。

实际服务器 CPU 测试 24/24 通过，CLI help退出0；数据集为空的真实模板检查退出78，返回 UNBOUND、tokenizer_imported=false，且未写 raw result。旧64项服务器测试、本地重复测试和修复前21项测试未累加进本轮24项。

真实 source-only CPU 资产诊断02通过：同一原 tokenizers 0.22.2 后端、22个包/发行元数据来源、英文/中文/特殊标记3条声明为synthetic的测试文本，编码与解码检查正常。它不是自然数据或正式资格。早期默认import诊断01和特殊模式检查保留为历史，执行源码来源以02为准，不将其错误提升为正式证据。

实际执行均使用 CUDA_VISIBLE_DEVICES=空字符串和 .venv/bin/python -B -I -S。测试入口是 test_raw_tokenizer_cpu.py；CLI检查入口是 raw_tokenizer_cpu.py；资产诊断入口为 tokenizer_producer/root_diagnostic/probe_actual_tokenizer_assets_cpu_v2.py；冻结入口为原准备目录新增 freeze_prerental_sources_v12.py。所有实际argv/cwd/stdout/stderr/退出码/耗时见本目录 *_COMMAND.json、*_RESULT.json、*_STDOUT.log、*_STDERR.log，以及 NATURAL_CPU_DELIVERY_REPORT.json。

V12已实际流式核验 4953 项来源、16085064210 字节，严格保留V11全部4,919个引用。V12源锁SHA-256为 23ab34887c7f4340a37f2814936c639c46890ca31e4b3ba1b07359d41fc13f77。冻结锁含模型和SDK的实际外部文件引用；CPU发布工具由交付ZIP独立SHA清单记录，不把来源检查当GPU资格。

本轮GPU操作0次，原GPU预算账本空闲且字节未变；剩余约 42.96 分钟。未改变GPU电源状态，也未做新设备查询。此前的功能和单个精确GPU成本单元仍保留，但正式策略效果尚未验证。PRIMARY实际可用约 48.54 GiB；本轮无需扩容。

下一步只需先提供真实请求文件的服务器路径。格式、来源及数量限制见 HOW_TO_PROVIDE_REAL_INPUTS.md 和 input_contract_review/NATURAL_INPUT_CONTRACT_REVIEW.md；RAW_TOKENIZATION_REQUEST_UNBOUND_TEMPLATE.json中dataset_ref刻意保持null。现有有界目录检查未找到真实trace，不能据此声称全盘不存在数据，也没有拿旧资格请求或测试文本充当实验输入。

raw分词成功后仍须闭合真实会话/文档/公共前缀来源、固定完整分区和真实独立控制deadline。实际开发声明必须对应运行时host/boot/CLOCK_MONOTONIC；切换模式或重启改变时钟域后不能沿用旧boot声明。开发诊断可按原合同处理未绑定服务SLO，正式效果仍需要事先服务目标和真实native reserve/普通候选覆盖。

完成上述输入后才进入原guard下的正式开发 U/off，关闭证据完整后再运行 I/shadow。当前不需要GPU，也没有有效的性能提升或论文收益结论。本轮未下载、删除数据或修改驱动/系统。

交付包只包含CPU源码、测试和证据及来源元数据，不包含完整模型、tokenizer二进制/完整tokenizer.json、SDK或私有缓存载荷；这些仍在服务器。需要保留原CPU桥接ZIP和GPU功能/成本ZIP，不能用本包代替整机备份。
