# P4 下一有卡阶段准备（本轮无卡）
本轮仅执行 CPU 准备与命令预览，不探测 GPU，不创建 GPU 预算预留，不下载或编译，不删除或合并缓存。原生准备脚本 --check-launch 已实际退出 78；当前仍是 CPU-only，明天有卡的描述不授予今天 GPU 启动权限。

已准备实际新脚本 `experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py`：
- `--mode off|shadow --name SAFE --dry-run` 纯标准库预览。
- `--check-launch` 在 GPU 导入或预算操作前拒绝。
- `--execute` 只接受独立的 GPU scope 恢复记录、最终源码锁和真实既有 run_gpu_stage reservation/session/argv。
- 复用原 P316 的真实 CanonicalKV、TransferCoordinator、handler store/restore/shared 路径；没有缓存或模型执行器重写。
- 新 native/control/Python author workspace 均固定为 P4-02。Python 源码来自新 author fork；只有逐个 SHA 锁定的旧 _C.abi3.so 和 _C_stable_libtorch.abi3.so 可作为 binary fallback，不复制或编译它们。兼容性未实测，仍是明天第一项 GPU 验证。

原生资格每次限时 180 秒，沿既有预算 guard 额外保留 20 秒；存储预留 128 MiB、实际 staging 配置 16 MiB、iodepth=1。off/shadow 不新增普通额度，因此不把它们宣称为 I/D/J 或零普通额度进展的完整 GPU 资格。真实 GPU 四组动作与精确 KV、AIO/drain 结果尚未执行。

`plan-v2.json` 保存真实现有 CLI 的 AST 声明、源字节绑定及真实 argv：
- 可复用的作者 copy primitive。
- 可复用的共同模型/native Prefix sanity 使用旧已编译 author 路径，明确不授予新 P4 资格。
- 新 P4-02 native off/shadow 资格入口。
- 旧 P3 calibration/model/capacity runner 的硬绑定与所需薄适配位置，不能仅改变 PYTHONPATH 冒充已迁移。

最终源码仍在 CPU 开发中。root 必须冻结 `gpu-next-day/final-source-lock.json` 的 files=[{path,bytes,sha256}]，覆盖新 native、P4 control、新 author offloading common/scheduler/worker、两个新 runner 和旧精准 binaries，再生成带 source-lock 的最终预览。未冻结保持 BLOCKED。

GPU scope 恢复记录需要 status=USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION、schema_version=1、显式 GPU booleans、allowed_modes=[off,shadow]、allow_model_downloads=false、new_executor=false、permission 文件字节/哈希绑定、当前批准 GPU UUID、source-lock 路径与字节 SHA。本轮没有生成任何伪授权记录。

空间只读检查：当时 PRIMARY free=11,003,219,968 bytes；8 GiB floor 下 128 MiB primitive 可容纳，原 model round 的 3 GiB 预留不足。AUX used=20,095,848,448 bytes /20 GiB cap；3 GiB 也不足。每次明天运行必须重做 preflight，不能据此自动调低模型轮次预留或合并缓存。

34 项新增 CPU 测试通过、0 失败、0 跳过。测试进程没有导入 torch/vllm/py_kvcache/cupy；GPU账本 SHA 保持 31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1。首个 shell log 目录错误发生在测试启动前，原负回执保留；通过 XML 以 cpu-tests-03 为准，不把独立重复运行叠加计数。

下一阶段仍须 GPU 授权/设备/源码/预算/空间 gate，实际 native/模型/paired interference/load/ETA/开销资格和 P4 效果对照。正式 P5 的 SLO 仍为 null，没有启动 P5–P7；目前没有方法提升结论。
