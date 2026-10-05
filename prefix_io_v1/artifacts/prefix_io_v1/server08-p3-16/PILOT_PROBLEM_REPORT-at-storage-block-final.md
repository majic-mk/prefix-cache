# P3 当前实施交付：存储授权阻塞，阶段未完成

当前状态是 **P3_NOT_COMPLETE_STORAGE_AUTHORIZATION_BLOCKED**。在新服务器 20739 上，作者原生 CUDA/LinuxAIO 和完整模型已实际运行成功。当前阻塞是磁盘资源预留检查；不是没有 GPU，也不是已证明方案不可行。P4–P7 保持关闭。

本轮新增共同修复限于 reactor.py 与 staging_cache.py：紧凑 staging/pin 计数、共享同一 slot 的消费者去重，以及 STOP/异常时等待原 CUDA/AIO 排空证据再回收。fixed/pressure 控制仍使用已有 simple_stage_policy，不新增缓存引擎、模型执行器、joint scheduler 或第二释放队列。原精确 prefix、成本准入、共享 staging、预加载、复制合并和异步流水线保留。dispatch_controller=None 直接回到原 stage 签发；共同安全修复保留。修改函数及行号见 lifecycle-and-modification-map.json。

CPU 去重资格：1583 passed、16 skipped，包含 12 个冻结历史协议 fixture；不能把它们都称为当前生命周期测试。guarded 执行矩阵、finite/capacity、补丁往返和容量分析已完成；此前失败和一次无 CUDA guard 的通过收据全部保留。旧失败：首轮 compact fixture 不合法、旧裸构造 fixture 缺 _stop、旧大 registry fixture 缺新增计数导致保守 unknown；修复及新资格没有回写旧结果。补丁往返覆盖共同增量和研究 patch 的正反应用、七阶段 31 文件身份及 CPU common-only/off，不替代 GPU 运行时回退证明。

本轮真实 GPU 共 29 次，全部 wrapper/child exit0、无超时、会话已排空。本轮用时 2653.93 秒；全程累计 15753.81 秒（4.376 小时），8 小时上限余 13046.19 秒。GPU UUID 为 GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8，RTX5090，实际驱动595.71.05。

已完成 16-case 原生简单阶段资格、80-member registry 的两类真实共享/staging资格、两轮 decode标定及独立验证（各28 windows）、六次 U/F16/P4 重复、F8首轮、两个容量域各六阶段点资格、首轮GPU-hot U以及四次观测开关对照。case/window不是额外GPU run；旧P315的失败all-hit定义及未执行尾部保留。

|共同 1GiB 候选|已完成次数|cohort含排空的中位数/秒|解释|
|---|---:|---:|---|
|U|2|32.566494|冻结的正常开发流；描述性，无置信区间|
|F16|2|34.856058|冻结的正常开发流；描述性，无置信区间|
|P4|2|36.410556|冻结的正常开发流；描述性，无置信区间|
|F8|1|32.931849（单次）|缺重复，不能选赢家|
|P512|0|未运行|存储阻塞|

F16、P4 的两次中位数均慢于 U；尚未证明方法提升。F8真实出现8990次 performance deny decision，包含重复检查，不能称8990个独立parent，也不能继续称全部额度均未binding。普通 U 两次真实 pending flush 为0.408055/0.641355秒；目标存在，不能以“没有普通等待目标”作为止损理由。

两个容量域 cap960-l1 与 cap1024-l2 已分别通过原数值/介质/成本门槛；各自 loaded-U 敏感性尚未执行，不能汇入共同候选赢家或称容量调优完成。512/768/896MiB 因整段prefix所需1016文件与可用slot上限不相容而结构剪枝；未改变prefix长度或放宽门槛。容量v1计划曾被误改元数据，已恢复原SHA并保留事件；后续只用新增v2/full128参考。

稀疏干扰七个条件cell已通过原25%条件（验证loaded误差、重复spread、首尾drift）；该表仅支持owned/conditional校准域，production-state lookup未完成、失配返回None。没有正式收益、真实DMA overlap或完整状态泛化结论。

GPU-hot/观测4次对照测量SSD读写均为0，原流水线开启，warmup写入单独计入。optional观测on/off的cohort中位数 10.170148/10.046723秒，描述性差值+1.2285%。这只隔离安静域可选sink，不能推断normal-I/O总观测成本≤2%；共同parent/capacity计数两臂均开启，thread CPU窗口也不等于model总CPU或墙时窗口。

资源释放能力保持保守：CUDA原事件query/同步、AIO完成及完整parent闭包才可回收；原事件没有证明完成时保留owners/Futures并返回unknown。诊断snapshot明确gpu_release_credit=false；旧立即GPU复用UNKNOWN仍为UNKNOWN。host-hold event facade实际调用原query但可延后可见true，只证明软件等待可见证据，不证明真实DMA忙时延或性能改善。

当前PRIMARY空闲11784204288B，下一模型固定预留3221225472B，扣除后小于8589934592B底线，原preflight拒绝。仍缺3次finite（P512/P512/F8）、2次capacity loaded-U、3次GPU-hot B/B/U，共8次。10269个旧P315私有重复缓存清单SHA e692ad4495548d9729423d956ceea405bbcac4202d5b9e50bee2db4ac424e062，预计可回收9421848576B，仅是预计值。该清单没有收到新的精确人类grant，未创建授权或合并；旧8226/8231清单授权不能复用。共享源、模型、代码、日志不在目标内。

已重新逐字节核对3048份注册源，共2796552192B，全部SHA匹配、无新增bin；冻结08的2599输入全部匹配。实际版本metadata/作者HEAD/16个ELF字节身份已记录，ELF只读hash不构成独立ABI验证或整个安装环境快照。完整原GPU资格依各wrapper/result判断。

实际执行命令均可查GPU result.json、out316/*-command.json、CPU XML与 p3-reproduction-at-storage-block.json。GPU必须经原run_gpu_stage.py、唯一label、同UUID空闲/source冻结/预算/磁盘预留校验；不要裸执行child argv。新closeout文件是本截点报告，不能替代未来完整P3验收。

下一允许阶段仍是 **P3剩余8次对照**：收到精确缓存grant后，在GPU完全idle时重审此冻结清单、dry-run、应用并保存10269个verified及完整journal，再逐字节核对所有路径/内容并记录真实df回收量；随后继续原冻结候选。未获grant不降低3GiB预留、不删除缓存、不启用P4，也不写P3完成。

补充：最终CPU报告字段修正版v3通过30项守卫测试及50项子检查；先前v2的28项为重复验证，不再次相加。CPU去重总数为1613，仍保留16skip和基数中12个历史fixture的范围限定。v3确认核心模型、真实普通等待、稀疏条件标定、quiet观测、原生transfer/shutdown、CPU及补丁往返门禁通过；未满足项仍来自8次GPU未执行及有限开发未闭合。首次/二次报告适配失败原样保留，不改任何真实GPU结果。

本报告表中的P4仅指pressure reserve4Q候选，不代表实施阶段P4获准开启。quiet观察PASS仍受安静域限制，不授予normal-I/O≤2%的工程资格。

缓存合并dry-run实际exit0、apply=false、仍10269目标/9421848576B预计回收；六根被审计，目标替换分布在五根，第一根作为canonical保留。没有auth或applyjournal。待精确人类授权。

冻结输入快照v2含2599项，ZIP SHA dcefc31e51208f8ff4e91de34389bd032642d20f43946595617506c07c42fa59；其中一份已锁定AUX历史参考用external_aux_inputs映射，仅复制读取，未写AUX。另有16个ELF完整流式SHA身份审计；不打包模型/私有cache/库二进制。
