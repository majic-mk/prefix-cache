# P3 实验与实施最终证据报告

全部预注册 P3 GPU 工作已经实际执行。此报告提供验收输入；阶段关闭须在严格closeout与root审查实际通过后记录，最终裁决见P3_FINAL_ROOT_REVIEW.md。固定路线继续在作者py-kvcache和配套vLLM上增量修改；没有重写缓存引擎或模型执行器。

共同修改限于reactor.py/staging_cache.py的紧凑容量及pin计数、共享同slot消费者去重，以及STOP/异常时按原CUDA/AIO完成与parent闭包等待排空再回收。研究fixed/pressure策略与共同修复分离，dispatch_controller=None回到原stage签发；精确prefix、成本准入、共享staging、预加载、复制合并和异步流水线均保留。运行代码15个native文件，测试16个，不能将31个均称运行代码。完整位置、版本、生命周期见lifecycle-and-modification-map.json与current-environment-version-lock.json。

CPU资格为1613个去重passed、16 skipped；50 subtests单列，12个冻结历史fixture适用范围保留。无CUDA初始化guard、CPU协议矩阵、有限候选/容量、容量分析、补丁正反往返与strict closeout schema资格都有命令/XML/guard。未重复累加先前28个schema测试。此前失败及无guard通过收据保留，不冒充新的生命周期资格。

真实GPU共38次stage尝试，其中37次成功、1次失败；不是38次模型成功。成功包括2次原生资格、3次条件标定、12次容量点资格及20次完整模型cohort，每个cohort的10请求/1280输出均与full128 golden一致。全部尝试已自然排空、无timeout。P316累计GPU 3420.680539 秒；全程累计 16520.562887秒（4.589045小时），8小时额度剩余 12279.437113秒。实际RTX5090 UUID GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8，driver595.71.05。

|共同1GiB候选|独立运行数|cohort含排空中位数/秒|相对U时长变化|
|---|---:|---:|---:|
|U|2|32.566494|+0.0000%|
|F16|2|34.856058|+7.0304%|
|P4|2|36.410556|+11.8037%|
|F8|2|33.372540|+2.4751%|
|P512|2|34.947063|+7.3099%|

按事前最小中位数规则 **B=U**。本冻结开发工作负载没有证明新策略提升；F8/F16/P4/P512中位数均慢于原路径。每臂两次仅作描述性判断，不提供置信区间、正式SLO、未见测试收益或全局最优结论。F8首轮8990次performance deny是重复决策次数，不能称独立parent。普通U重复pending flush为0.408055/0.641355秒，目标存在，不以无等待目标作为止损理由。

容量960MiB/horizon1和1GiB/horizon2各六点资格及一轮loaded-U均通过原门槛。其loaded cohort时长分别为 34.365089/33.582873 秒；两域不汇入共同候选赢家。容量与horizon同时不同，不能归因于单一参数。512/768/896MiB保持原整段prefix几何剪枝，没有缩短prompt或放宽门槛。

cap960第一次尝试因冻结计划错误引用permit.json而非capacity-permit.json，在模型启动前失败；13.852598秒已计入预算，旧失败wrapper/log/v2计划保留。只新增v3计划修正路径，并通过禁CUDA CPU实际门槛/SHA守卫；参数、门槛、预算和reserve不变。960成功使用mixed-U-02，1024成功使用mixed-U-01；不是把旧失败重写成成功或科学剪枝。

GPU-hot按原U/B/B/U位置完成；由于B=U，实际四行均U/off/observation-on。每行完整prefix16256、golden128、SSD读0，generation写入按原上限计费，预热写入单列，外部流水线开启。不是一个新策略热缓存效果对照。可选观测off/on/on/off四次中位数差约+1.23%，只隔离quiet optional sink，不能推广为normal-I/O总观测开销≤2%；共同计数两臂均开启，wrapped-thread CPU不等于全模型总CPU。

稀疏干扰七个conditional cell在原25%门槛内通过；production状态lookup尚不具备资格，unsupported返回None。真实资源释放继续依赖原Event/AIO/parent完成，unknown保持unknown、gpu_release_credit=false；host事件可见性延后只证明软件保护，不证明真实DMA忙时延、overlap或提前释放收益。

限定合并依据人类对前一条具体10269/e692清单动作的“继续”回复，原冻结清单重新验证后完成10269文件合并，2060 canonical SHA、每个target同inode/size、无临时残留、完整intent/verified journal均通过；全部路径内容保留，共享源、模型、代码、日志不作为合并目标。实际净回收9409294336B（约8.76GiB），不是预计9421848576B。授权JSON记载原始人类回复及具体上下文，未扩大其他清单授权。

最终重新逐字节核对3048份注册源/2796552192B，全部SHA一致、无新增bin。11号冻结锁2608项全部一致；全部38个P316 session不再存活、ledger无active reservation、GPU空闲。PRIMARY空闲11449671680B，8GiB底线保留。版本与16个ELF字节身份记录不构成整个安装环境snapshot或独立ABI验证；真实原生资格依GPU收据判断。

下一允许动作是审阅P3结论并封存交付。P4-P7保持关闭；没有自动启动P4策略的授权，也不通过新增延迟、改变候选或混合容量域制造收益。
