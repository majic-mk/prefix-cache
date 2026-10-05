V3 是原 native_runtime_v2.py 的 CPU 可审阅薄接线。建议部署到原 preparation/runner/native_runtime_v3.py；不覆盖旧版本、作者文件或模型执行器。

新增正式 development/off/U 和 effect/off/U 分派。U 使用原 U/off 引擎、相同原 CUDA/host 观测与原 shutdown/排空；真实私有成本表仅用于运行身份验证，不构建 FiniteStartup、不安装 I controller。development/shadow/I 和 effect/on/I 的 startup 参数与关闭流程保持原路径。qualification 与无表 shadow 保留旧规则和旧 collector。

两个 effect 臂均先重放原 activation.verify、原 GPU 成本发行器和原 reserve_join.verify_existing_native_reserve。CPU helper 只返回原纯 JSON，并清理自身新增的 prefix_io_control 模块和 sys.path；不会把表或新授权转交给下游。真正守卫的 runtime 再次通过原发行器取得同一私有表。独立 deadline/SLO、真实 native development reserve、源码祖先、设备和共同执行域字段缺失即拒绝。

正式流的全部实际 token-ID 前端输出和 CUDA frames 由原 reserve_join._capture 逐一校验。UNKNOWN、unsupported、缺 CUDA witness、输出不全均保守拒绝，同时仍执行原 finally 关闭/排空。新父证明接口要求实际 frontend/capture 字节 refs 冻结闭合，并重新检查原 native tail，不能靠 PASS 标记代替原始证据。

正式 effect 的命名空间 peer 可来自 U/off 或 I/on，以保留预先固定的 AB/BA 顺序。两个 peer 都要完整原 CUDA/输出/tail 纯读重验；I peer 另需实际 owner install、运行 identity、原 shutdown 后 startup 恢复证据，并重新原 CPU 私有表发行/实际 development reserve 校验。development 的 off prerequisite 仍只接受实际 U。第一版 14 项 CPU 证据保留在原记录，加入 generic peer 契约后以 `_02` 记录为准。

测试只检查 CPU 分派、拒绝边界、原函数/关闭 AST 未变、call spy 参数与 import 恢复。Spies 不产生私有 CostTable、模型执行、GPU frames 或真实 development receipt。16 项本地 CPU 测试通过不构成 GPU 资格，不构成性能提升。

当前真实正式输入、独立 SLO 和正式 development U/I 证据尚缺；此目录没有制造这些文件。本轮 GPU 操作数为 0，未证明正式混合工作负载的全流观测或效果。服务器源码冻结、测试和打包由根 agent 执行。
