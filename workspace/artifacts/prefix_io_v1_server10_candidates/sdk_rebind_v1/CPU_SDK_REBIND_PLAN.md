# Server10 CUDA13 CPU 资产重新绑定

已有 CUDA13 SDK 与原 `cuda13_sdk_overlay.py` 保持原字节。迁移工具只重新确认现有资产，并在新私有目录生成新服务器的 inventory 和真实 CPU 编译链接凭证。

原 helper 要求凭证有四条成功命令，不能直接把旧 595 驱动凭证改成 580 驱动后宣称通过。工具在新目录中重做小探针的 `nvcc --version`、SM120f CPU 编译、`c++ -shared` 链接和 `readelf -d`。原先小探针编译约 0.7 秒，链接约 0.07 秒；这些历史时间不是新服务器的实测值。

现有 217,111,529 B SDK 树仅核验及建立符号链接，不复制。原 54 B `probe.cu` 经旧 SHA 校验后复制到新目录，生成的小型 `.o` 和 `.so` 不执行、不加载。真实驱动只被读取校验和链接器读取，不修改驱动、不导入框架、不运行任何 GPU kernel。

服务器最少命令（在 project 根目录）：

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server10-sdk-rebind-v1-20261003/test_rebind_cuda13_cpu.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server10-sdk-rebind-v1-20261003/rebind_cuda13_cpu.py
```

输出新目录：`artifacts/prefix_io_v1/server10-cuda13-cpu-20261003`。最终 `SDK_REBIND_RESULT.json` 只有在未修改的原 helper 完整验证新 inventory/proof、树、驱动、cudart 和三份探针输出后才报告 PASS。

本地已执行四项 CPU 边界测试：同大小内容漂移拒绝、布尔型大小拒绝、已存在收据不覆盖、CPU 子进程清除继承的库/编译器注入环境；4/4 PASS。服务器真实编译结果应以现场生成的收据为准。

该凭证只说明工具链能够在 CPU 完成编译与链接。GPU 实际可用性、完整模型输出、缓存路径数值一致性和策略效果仍需分别依据后续已授权 GPU 运行结果，不能由本凭证推断。
