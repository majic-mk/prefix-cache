# 第五轮可行性裁定：当前容器不可运行原生 SSD 路线

日期：2026-09-27（Asia/Shanghai）。服务器 connect.westc.seetacloud.com:24801，容器 autodl-container-67e44a9c3a-cb95ca9e，项目 /root/autodl-tmp/prefix-io-v1-handoff/project。

## 结论

**NO_GO_CURRENT_CONTAINER。** 截至本轮检查，当前容器拒绝原生 io_uring 必要调用；通过本任务可访问的 SSH 环境，未发现用户可控、平台支持的最小修复入口。因此，在当前配置和可访问管理权限下，指定作者 py-kvcache + vLLM 的完整原生 SSD 缓存路径不可运行。

这是当前部署环境的不可行结论，不是对 py-kvcache／vLLM 架构或研究策略的否定。系统在具备正确原生 I/O 能力的环境中是否完整可用、策略能否产生收益，仍未验证。P0 完成；P1 部分通过、完整验收失败；P2–P7 未启用。

本轮不再仅以 permissions.yaml 禁止系统变更为停止理由：用户已给予有限例外授权，已实际检查可实施修复条件；当前障碍是必要系统调用被拒绝且没有已提供的管理入口。增加项目授权文本不会自行改变这些服务器能力。

## 本轮实际检查

| 项目 | 实际结果 | 证据 |
|---|---|---|
| C 探针编译 | gcc -O2 -Wall -Wextra -Werror，exit0 | compile-probe.json |
| 直接 io_uring_setup | 系统Linux头文件、结构体120字节、depth=2、flags=0；return=-1、errno=1/EPERM，exit2 | raw-syscalls.json、io_uring_syscall_probe.c |
| 辅助入口诊断 | enter(fd=-1) 与 register(fd=-1) 均EPERM；无I/O提交 | raw-syscalls.json |
| 作者原生 LiburingRing(2) | 固定文件路径与SHA；在setup阶段抛EPERM，exit2 | native-ring.json、native_ring_probe.py |
| 内核开关与内存锁 | io_uring_disabled=0；memlock软/硬限制均不限 | platform-audit.json |
| 继承的过滤状态 | 当前进程与PID1均Seccomp=2、Seccomp_filters=1 | platform-audit.json |
| 管理入口 | 未发现docker/podman/ctr/nerdctl；所检查的Docker/containerd/Podman本地socket均不存在 | platform-audit.json |
| 权限补充的兼容性 | 现有严格schema仍有效，解析后的原权限和预算值与之前完全相同 | permission-validation.json |

作者 liburing_file.py:138–166 的构造首先调用 setup，失败立即抛异常，尚未进入 mmap；enter 在:263。这项必要前提失败已经足以否定当前容器中的固定路线，无需加载GPU模型再次重复证明。

作者原生文件SHA256：6a8995ca6e5f49ae470e8c49dbf3d98caf2d09dd091cb6f08f9e892c051414f5。

本轮是实际CPU平台诊断，不是mock、GPU测试或新单元测试回归。原生import出现Transformers弃用与CUDA不可见提示，原始stdout/stderr均保留；没有据此改版本、安装依赖或运行GPU。

## 限制来源与可修复性

已确认当前进程处于seccomp过滤模式。直接C调用与作者库均失败，排除了“仅Python封装失败”的解释；入口拒绝现象与容器策略限制一致，但未获得宿主机过滤规则，不能声称seccomp是唯一根因。

/proc/*/attr/current 的 docker-default (enforce) 属于AppArmor／LSM记录，不是已读取的seccomp规则。能力集合缺少SYS_ADMIN/SYS_PTRACE说明当前管理与诊断边界；普通flags=0的ring并非因此必然要求SYS_ADMIN。

Linux文档说明fork/exec继承已有seccomp过滤器，追加过滤器只能进一步收紧；在容器内另起进程或再加allow规则不能正常解除继承的拒绝。[Linux内核说明](https://docs.kernel.org/userspace-api/seccomp_filter.html)

Docker文档列出默认策略限制io_uring相关调用；AutoDL说明其容器实例不支持在实例内使用Docker。这些公开说明与本轮管理入口检查一致，但不等于平台运营方绝对不能调整配置。[Docker说明](https://docs.docker.com/engine/security/seccomp/)；[AutoDL环境说明](https://api.autodl.com/docs/env/)

未尝试提权、进入宿主机命名空间、关闭全局seccomp、替换I/O后端或安装嵌套容器运行时。现有可访问入口下，没有找到可以在限定授权范围内直接实施的系统修复。

## 授权如何记录

新增 experiments/prefix_io_v1/configs/authorizations/io_uring_20260927.json，记录本轮用户原话、允许范围、目标容器、排除项与预算。

permissions.yaml 只增加指向该有限例外的注释，保留 allow_driver_or_system_changes=false：原字段是合并的宽泛权限，不能用true准确表达“只允许该容器内平台支持的io_uring最小修复”。现有解析器严格限制键集合，因此没有随意添加新字段或扩大驱动/全局权限。配置验证确认解析值未变。

没有实际系统修改；无需回滚系统。已有缓存/模型执行代码、研究策略和GPU运行结果未改。

## GPU和预算

本轮GPU作业0，模型下载0，实际I/O提交0。共享预算账本逐字节未变，活动预留为空。

累计仍为285.247483194秒（0.0792354120 GPU小时）／8小时；模型保守扣费19,422,798,722 B（18.088890912 GiB）／20 GiB。余额约7.920764588小时、2,052,037,758 B。

第三轮Qwen2.5-7B BF16原生GPU Prefix的0→112 cached tokens及相同16-token输出是历史通过结果；本轮未重跑。真实staging／SSD／生产KV往返／性能收益仍没有通过证据。

## 下一步与重新进入条件

当前容器内停止追加GPU校准或后续策略实现，保留代码、模型和所有失败证据。下一步取决于外部条件变化：

1. 平台为目标实验容器提供受支持的最小权限调整或具有该能力的环境。
2. 新环境先通过原生setup及作者ring/mmap，再以真实enter/CQE验证项目私有路径的文件读写。
3. 然后继续P1的实际资源预算、合法成本曲线、staging／SSD／生产KV往返与末尾drain；不能仅因setup成功就宣布系统可行。

固定作者当前只使用setup与enter。register的无效fd探针仅辅助诊断，不能据它失败要求额外放开未使用的调用。不要求--privileged或全面关闭安全策略。

给平台的具体材料见 PLATFORM_IO_URING_FEASIBILITY_REQUEST.md；本任务没有向平台发送消息、创建新实例、购买资源或重建容器。若平台需要重建／重启，仍须先明确具体目标、数据持久化与恢复方式。

## 文件与命令

本轮证据目录 artifacts/prefix_io_v1/new-server-05/，包括原始检查JSON、C和Python探针、有限授权、独立复核、预算一致性与可行性裁定。精确命令见 REPRODUCE_NEW_SERVER_05.md。

本轮交付是独立可读的可行性证据包，不是已经完成P1–P7的系统发布包。第三、四轮代码包继续保留；没有清理旧目录或覆盖历史结果。
