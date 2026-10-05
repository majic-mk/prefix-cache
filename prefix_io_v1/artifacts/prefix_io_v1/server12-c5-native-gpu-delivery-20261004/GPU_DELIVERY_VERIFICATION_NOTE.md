# GPU 交付归档复核补充

实际服务器归档 263 个源/证据文件，264 个 tar 成员，原始 payload 27653558 bytes。
压缩文件 GPU_EVIDENCE.tar.gz：1995186 bytes，
SHA-256 ebb849264071d984649f3eaab13924786864293af65c0e634d1f69396fa2a39f。
manifest：66862 bytes，
SHA-256 2abf76cbfae347865a29d4144f5d9b0a0ffae51a21586d05d5aaeb3e41da47dc。
已完整下载到本机，263/263 原始文件逐字节 SHA 核验通过，
LOCAL_GPU_EVIDENCE_BYTE_VERIFICATION.json 保存原服务器路径与本机映射。

12_PACK_COMPLETED_GPU_EVIDENCE 首次 CPU 归档失败，因为辅助脚本误用
experiments/prefix_io_v1/permissions.yaml；实际冻结授权路径是
experiments/prefix_io_v1/configs/permissions.yaml。
首次尚未创建 manifest/archive；追加 pack_gpu_delivery_v2.py 修正这一字符串后，
13_PACK_COMPLETED_GPU_EVIDENCE_CORRECT_PERMISSION_PATH 通过（0.482 秒）。
原失败代码、命令和错误日志均保留；无 GPU 重跑、源锁或授权更改。

本机首次按完整原始深层目录展开 G/common 时触发 Windows MAX_PATH。
改用原路径摘要与原文件名映射到 recorded_files 后，完整逐文件验证通过；
tar 内原始服务器路径保持不变，不改系统长路径设置、不覆盖不同字节文件、
不删除首次已展开的文件。新标定和辅助交付目录仍直接按原文件名展开。
仅 GPU 原始记录与源码/元数据/日志归档，不复制大型模型、SDK 二进制或私有缓存；
这些全都留在服务器，不是数据清理操作。
