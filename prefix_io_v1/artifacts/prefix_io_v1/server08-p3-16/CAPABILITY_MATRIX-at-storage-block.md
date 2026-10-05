# 当前能力矩阵与版本边界

|能力|实际证据/状态|限制|
|---|---|---|
|精确Prefix Cache/完整模型|7次正常模型+5次GPU-hot各1280输出通过full128 golden|P3开发域，不是未见测试/SLO|
|成本准入|原1GiB permit+两个独立容量point门槛通过|新容量loaded-U未跑|
|共享staging/复制合并/异步流水线|原API与实际AIO/CUDA/完整排空保留，80成员共享资格通过|host query gate不能当真实DMA时延|
|preload|lookahead1在实际模型路径保留；lookahead2 point资格通过|lookahead2 loaded效果未验证|
|fixed/pressure控制|核心重复已完成，pressure普通registry已知；F8首轮真实额度deny|F8重复及P512两次未跑，无有限域赢家|
|资源释放依赖|原事件/AIO/parent边界释放；未知保守保留|没有新的GPU source-safe ACK/提前复用协议|
|稀疏干扰额度|七cell conditional table原阈值通过|未获生产状态资格，unsupported返回None|
|观测|真实thread CPU包裹+quiet开关4次|normal-I/O optional开销未隔离；共同计数常开|
|有限候选调度|P3简单策略有限候选冻结、部分已跑|P4 dependency/joint策略未开启|
|版本锁|214metadata+作者HEAD+16ELF SHA；2599source/input身份|不含模型/库二进制本体或完整安装环境|
