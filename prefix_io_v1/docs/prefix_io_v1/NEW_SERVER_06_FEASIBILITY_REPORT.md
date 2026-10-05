# 新服务器 06：实际环境验证
目标：connect.westd.seetacloud.com:40168；主机 autodl-container-mz3u56hqpn-f855e672。
结论：当前实例 NO_GO。仅说明当前环境不能运行固定作者 I/O 路径，不等于研究系统原理不可行。

## 本轮实际检查
2026-09-27 UTC，Linux 5.15.0-78-generic，x86_64，root，memlock 无限制。
- Python 标准库原始 io_uring_setup(depth=2, flags=0)：退出 2，errno=1 EPERM。
- 现场 gcc -O2 -Wall -Wextra -Werror 编译：退出 0；C 原始 setup：退出 2，errno=1 EPERM。
- 作者原生 LiburingRing(2)：退出 2，errno=1 EPERM；源文件 SHA256 6a8995ca6e5f49ae470e8c49dbf3d98caf2d09dd091cb6f08f9e892c051414f5。
- C 的无效 fd enter/register 诊断均 EPERM；register 不作为作者路径的额外要求。
- Seccomp=2，filters=1。结果与容器策略过滤相符，但未取得宿主过滤规则，不能唯一归因。
- 本内核未提供读取的 io_uring_disabled/group sysctl 节点；缺失不表示禁用。

## 实际改动和证据
仅新增 artifacts/prefix_io_v1/new-server-06/ 的诊断记录、报告及交付包；更新执行状态指向新实例，并保存旧状态。
每个探针 JSON 含完整 command、stdout、stderr、退出码及耗时。identity.json 含目标、内核、授权范围及输入文件哈希。
未改业务源代码、驱动、系统 CUDA、内核或安全策略；permissions.yaml 和预算账本保持不变。
未运行测试套件；本轮结果为三项独立环境探针，不能计为原有回归测试重新通过。

## GPU 与下一阶段
GPU 运行 0，模型加载 0，下载 0，未启动真实文件 I/O：创建 ring 的必要条件已失败。
历史 GPU Prefix 成功记录来自旧实例，不能作为本机 GPU 验证。新机器 GPU 身份及适用授权未验证。
P1 仍阻塞，P2–P7 不开启。下一允许步骤是取得能通过原始 setup 和作者原生 ring 的环境后，先完成小规模真实文件读写，再按 GPU 身份授权及剩余预算恢复 P1。
连续换普通实例未消除同类阻塞；不建议继续盲目换 GPU 型号或重新安装 CUDA。可继续 CPU 开发，但无法据此宣称完整系统可行。
