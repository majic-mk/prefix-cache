# 无卡阶段：明日模型实验空间恢复只读审查

结论：当前已注册、已完成实验私有 KV 缓存的合法重复项不足以恢复原定 3 GiB 模型轮次预留。没有实际回收空间，也没有新审批或任何合并操作。

审查覆盖 61 个有成功 result.json、engine_shutdown=completed 和 storage-source-manifest.json 的私有缓存根，并复用 5 份以前的精确合并审计。共核对 222,948 个注册路径别名、13,318 个唯一 inode，元数据错误为 0。没有全磁盘扫描，也没有读取模型权重、二进制库或用户共享数据。

先做元数据分类，再只读取候选不同 inode 的真实 KV bytes。共 16 个 inode、14,680,064 bytes（14 MiB），每个 inode 只读一次 SHA-256；全部与注册摘要一致。symlink/非 canonical path/非 KV .bin/不支持目录范围均被拒绝。读取前后 inode/size/link count/mtime/ctime/allocated bytes 稳定，完成 receipt 和注册清单字节再次核验未变，GPU 预算账本未变。/proc FD 名称检查未发现注册私有 KV payload 的打开 FD。此为时间点只读证明，不是未来写入锁；未来即便获准也必须重新核验。

保守 nlink=1 方案只有 8 份 AUX 私有重复缓存、8 个组，可能回收 7,340,032 bytes（7 MiB）。保留全部路径和内容，未来仅限定目标 inode 会变化；注册 P3 只读源不会被替换。已经合并的旧大组拥有多个路径但只有同一个 inode，不能再次算回收收益。未发现另外可提供收益的完整闭合私有 shared-inode 别名备选。

本次观察时间：2026-10-01 02:28:04（Asia/Shanghai；原始 UTC 存在 result.json）。
原定每轮模型预留：3,221,225,472 bytes（3 GiB）。
最低空闲：8,589,934,592 bytes（8 GiB）。

PRIMARY：
- 真实 statvfs 空闲 10,889,969,664 bytes。
- floor 上可用 2,300,035,072 bytes。
- 3 GiB 预留不足 921,190,400 bytes，约 0.858 GiB。
- 本限定方案在 PRIMARY 可回收 0 bytes，所以仍阻塞。

AUX：
- 真实 statvfs 空闲 11,250,208,768 bytes。
- cap=21,474,836,480 bytes（20 GiB）。
- 本轮已用 20,095,848,448 bytes 沿用 root 2026-10-01 任务消息中提供的实际元数据；本审查没有重扫 AUX 全目录，未把这个用量称为新测量。它未在 gpu-next-day/plan-v1/v2/v3 或 p4-01/p4-02 的小 JSON 里找到同数值原始记录，因此明确保留来源边界。
- cap 剩余 1,378,988,032 bytes，floor 上余量 2,660,274,176 bytes，cap 为当前更紧的限制。
- 原 3 GiB 预留不足 1,842,237,440 bytes。
- 即便未来获准且实际回收全部 7 MiB，仍缺 1,834,897,408 bytes，约 1.709 GiB。
- 必须以未来实际 cap/free preflight 为准，不能据这个只读 proposal 直接启动模型轮次。

冻结清单：
private-cache-space-proposal-frozen.json
bytes=53,225
SHA-256=4dcff26836aabbd1b77f6244eb62dfb74f30efae3f4ccc8dbe3eb2a063ff718b

证据：
read-only-space-review-result.json、actual-candidate-inode-hashes.json、
metadata-scope-complete.json、metadata-batch-{0,16,32,48}.json、
registered-completed-run-inventory.json、旧审计 inventory、执行脚本和命令汇总。
metadata-state*.json.gz 只是本次元数据图，没有包含任何 KV payload 或模型数据。

本次真实动作计数：
GPU=0；硬链接创建=0；删除=0；移动=0；缓存写入=0；保护源替换=0；新授权=0。
仅在 project artifacts 写了审查证据。没有调用 consolidation apply 工具，没有申请把旧批准扩到新清单。

下一阶段：root 按这个真实不足结论准备明日命令门禁。原定 3 GiB 模型轮次仍须先解决空间；任何改变固定预留都需要另有具体、可复核的容量依据，不能静默降低 guard。仅靠本次私有重复项不能解除阻塞。
