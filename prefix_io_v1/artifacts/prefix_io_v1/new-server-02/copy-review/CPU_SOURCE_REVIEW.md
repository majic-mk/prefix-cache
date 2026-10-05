# 作者 swap_blocks_batch 最小资格脚本：CPU 源码审查

本次仅准备脚本并完成 CPU 源码/参数入口检查。未加载扩展、未执行 ABI op、未执行 GPU、未下载模型。以下内容是源码确认和测试计划，不是真实复制合格结论。

## 固定来源与真实接口

- 作者 vLLM HEAD：817a7e3124f817cd6e549581d3e5483207a753a4。
- py-kvcache HEAD：3abba7a502d553f6e7e2e58b92086487e3395d7e；读取的是已有 observer 预集成工作树。
- csrc/torch_bindings.cpp:287–292 注册 _C_cache_ops::swap_blocks_batch 的 CPU dispatch；参数为三个 Tensor 和默认 False 的 bool。
- csrc/cache_kernels.cu:78–93 要求三个 CPU int64 向量并检查长度，n == 0 在 getCurrentCUDAStream 之前返回。
- csrc/cache_kernels.cu:99、126–143 使用当前 CUDA stream。CUDA 12.8 编译路径尝试动态查找 cuMemcpyBatchAsync；163–173 的旧驱动/旧 CUDA fallback 是作者实现自身逻辑。
- vllm/_custom_ops.py:2681–2703 的包装器直接调用同一 op。is_src_access_order_any=True 仅在源没有并发 GPU 写入时安全；脚本固定 False，与当前 py-kvcache/reactor.py:2028–2032、2086–2090 的默认调用一致。
- canonical GPU KV 存储要求 int8（py-kvcache/transfer.py:137–146）；该 op 实际按裸地址和字节数复制，脚本使用 int8 GPU 存储及 uint8 pinned CPU 存储。

没有导入 vllm、vllm._custom_ops 或 platform；指定扩展必须通过 torch.ops.load_library 直接加载。脚本拒绝已注册的同名 op，记录指定扩展的 SHA-256，并严格比对 schema。未实现替代 op、替代引擎或模型执行器。

## 模式和证据含义

CPU 模式：

```bash
CUDA_VISIBLE_DEVICES='' .venv-prefix/bin/python \
  experiments/prefix_io_v1/scripts/qualify_author_copy.py \
  --library /exact/path/to/author/_C.abi3.so \
  --abi-only --output /new/evidence/abi.json
```

路径只是调用格式，必须由构建方换成真实锁定作者扩展。零长度 CPU int64 调用分别检查默认 bool 与显式 False；三个 int32 参数和长度不匹配必须被真实作者 op 拒绝。任何 schema、参数、动态库、dispatch 异常均使资格失败。检查 torch.cuda.is_initialized 保持 False。ABI 通过只能证明注册/参数路径，不代表真实 H2D/D2H。

真实复制模式必须由外层权限/预算 runner 调用；该 runner 先把 CUDA_VISIBLE_DEVICES 设置为唯一获准的完整 GPU UUID，再传同一 --gpu-uuid。脚本不会自行设置设备可见性，不会扩充授权。未精确匹配在 import torch 之前失败。脚本还要求恰有一个可见 CUDA 设备，并只使用 cuda:0。

真实复制数据路径：
1. 1024 B GPU int8 payload；3 份各 1024 B 的 pinned CPU payload。CPU 元数据向量均为 CPU int64，地址/字节数均来自固定、校验过的区间。
2. GPU 在专用当前 stream 填充 0xA5 guard；作者 op 把 3 个不等长、不同偏移的 CPU 已知字节区间写入 GPU，共 334 B。
3. 作者 op 把完整 1024 B GPU 区间读回 pinned witness，逐字节确认 payload 和所有 guard。
4. 同一作者 op 将 GPU 的 3 个区间复制到另一份不同偏移的 CPU 目标，共 334 B；再次逐字节确认 payload 和 guard。
5. finally 中同步 stream；所有 payload 与元数据引用保留到同步结束。正常完成才记录 stream_synchronized_before_exit=True。

总真实 payload 传输是 H2D 334 B、D2H 1358 B，3 次非空作者 op 调用。GPU context、动态库、驱动和 allocator 开销不包含在 1024 B payload 数字中，仍须由外层预算管理。外层超时强制终止不能当作同步成功或测试通过。

该测试是组合 H2D/D2H 与 guard 的字节资格检查，不能单独定位两方向中的所有可能故障；不证明使用了 batch driver 分支而非作者 fallback，不给性能结论，不包含 io_uring、reactor、模型、Prefix Cache 命中或生命周期验收。

## 本次实际 CPU 检查

- Python AST 解析、禁止 vLLM import 语句检查通过。
- 作为模块导入脚本未导入 torch，未执行 main。
- 纯 CPU 已知字节计划、三个源/目标区间和总字节数检查通过。
- 5 种越界/非正长度输入被拒绝。
- --help 通过。
- GPU UUID 与环境不一致的入口按预期 exit 2；这是拒绝路径测试，不是 GPU 执行。

证据：commands.jsonl、cpu-source-check.txt、cli-help.txt、cli-uuid-binding-rejection.txt、source-manifest.json。真实 ABI 和 GPU 输出须由主线程后续在独立证据文件中记录；不得覆盖上述准备证据。

实际新增脚本：experiments/prefix_io_v1/scripts/qualify_author_copy.py。没有修改作者源码、权限、锁或其他模块。
