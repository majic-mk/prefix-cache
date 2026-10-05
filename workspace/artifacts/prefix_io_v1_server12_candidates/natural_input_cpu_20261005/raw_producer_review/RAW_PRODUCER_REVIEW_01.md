只读审查发现一个需要在冻结前修复的实际来源闭合缺口。

raw producer 的无 dataset 短路、原作者完整保序、独立 raw schema、synthetic fixture 隔离、有限输入整体拒绝和 append-only 输出均符合预期。已读取实现、21 项测试及对应本地测试记录；没有重新执行测试，没有导入真实 tokenizers/模型，没有 RPC 或 GPU 操作，没有改 producer 或旧源码。

阻塞点是已有 .pyc：sys.dont_write_bytecode=True 禁止写缓存，不禁止读取缓存。实际 tokenizers 导入和 load_protocol 均使用 Python 默认缓存 loader，而包叶核验不包含 .pyc。合法 timestamp/size 的旧 bytecode 可被执行，module.__file__ 仍显示已锁 .py。已通过本机 CPython 3.12 的 _bootstrap_external.py 第 1128/1135 行确认此行为；无需运行真实分词器。

已向根代理和实现代理报告，建议协议单叶直接从已核验字节 compile/exec，tokenizers Python 叶采用 scoped source-only loader，二进制保留原固定 loader；也可以显式拒绝已有缓存但不得删除。需要 CPU fixture 覆盖已有有效恶意 pyc 被忽略/拒绝。当前审查 SHA 19b9bbffe9fa4501c638988c548e79cc7fd9cf8d56e1d742cda2d9f9bc000d90 尚不能通过来源闭合审查；真实自然输入仍 UNBOUND。
