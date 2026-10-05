# 有界模型下载执行器 CPU 审查

审查对象：experiments/prefix_io_v1/scripts/download_pinned_model.py。所有实现与测试在服务器执行。没有真实网络连接、GPU 操作、模型下载或对真实共享账本的写入；测试仅使用临时项目、显式假 HTTP 字节流和独立账本。socket.connect 在本进程测试中被强制拒绝。

## 发现与修复

1. 默认 urllib HTTPRedirectHandler 会在跳转前调用 fp.read()，完整消费跳转响应 body，绕过执行器的 actual/amount 计费。先行测试真实调用标准库 http_error_302 复现这一缺陷。
   当前修复对所有 GET redirect 显式失败并关闭原响应：即使目标是当前白名单也不跟跳、不读取 body。URL 仍先受精确主站白名单约束。没有扩大 CDN 主机名单。实际 GET 出现新跳转须另行审查；失败按既有完整预留收费。
2. 经主线程确认，实际 ModelScope HEAD 200 没有 Content-Length。执行器现在允许缺少该头，但提供时仍必须与 manifest 大小完全一致。无头时仍只读取 expected+1 上限，最终要求精确大小与哈希；短响应/多一字节都不发布。

只修改上述两处逻辑；没有修改真实权限、GPU runner、共享账本、正式 docs/锁或模型 manifest。

## CPU 结果

先行 39 项测试：35 passed / 4 failed。修复后 **39 passed / 0 failed / 0 skipped**，0.46 秒。见 before.txt/xml、after.txt/xml。未将此结果计为实际下载资格或模型验证。

覆盖范围：
- 下载器与原 GPU runner 使用同一个 flock 文件；在下载持锁期间运行真实 GPU runner 代码的临时副本，证实其子命令未启动。没有运行任何 GPU 工作。
- 任意未结算 active_reservation 在请求前拒绝，不改变账本。
- 累计剩余额度而非原始预算、探测字节预留、既有 GPU 历史不变。
- 正常文件按实际 payload 结算；短响应、异常、错误哈希/长度/编码都保守收费并保留 partial；显式新尝试不返还旧失败消耗。
- Content-Length 缺失时完整成功、短响应拒绝、超长最多读取 expected+1。
- 原生 urllib 跳转缺陷复现与失败关闭；外域、HTTP、userinfo、非443与尚未批准的 CDN 拒绝。
- 越界/软链接路径、已存在错误文件、初始/中途磁盘空间不足拒绝。
- 冻结 revision/model/file URL/hash 类型绑定。
- 预留在网络 open 前已持久化/fsync；结算持久化失败保留旧 active 预留，后续运行被拒绝。
- 全量完成后再次运行只校验并复用所有文件，没有额外请求/费用。

补丁 0001-bounded-response-download.patch 包含两处实现差异与测试。正向、反向 git apply --check，以及 whitespace=error 校验通过，见 verification.json。

## 账本与后续限制

真实账本的前后哈希可能不同：主线程明确通知，审查期间已合法结算 metadata 115928 字节，并可能随后启动实际下载。审查未写该文件；verification.json 保留实际前后哈希，不把主线程活动错误标记成我们的修改。权限和 GPU runner 哈希保持一致。

本次不证明真实 provider 路径、网络性能、断点恢复或模型执行。下载器不做 Range resume；失败 partial 保留，重新请求另占预算。预算口径为 HTTP payload 与失败保守预留，不是 TLS/TCP 物理链路计费。模型 smoke 与完整 P1 仍取决于实际下载校验和既有阶段门禁。
