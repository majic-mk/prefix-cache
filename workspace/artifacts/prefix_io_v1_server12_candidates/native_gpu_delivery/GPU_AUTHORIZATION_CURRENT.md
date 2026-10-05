# 当前项目 GPU 授权
用户于 2026-10-04 明确指示：“授权gpu，只要有gpu就用，以后无需授权”。

此直接指示已记录于 PROJECT_GPU_AUTHORIZATION_AMENDMENT.json，并由
normal_runtime_authority.py 为现场配置产生独立 context、effective permission、
source binding 和 original GPU guard 接线。以后同一项目的 GPU 阶段、常规修复、
新冻结来源和服务器迁移不再要求逐轮人工授权；仍需现场硬件、来源、模型、
生命周期和阶段先决条件实际核验。

原 8 小时累计 GPU 预算保留，未因新授权增加。当前已用
24,155.170197121333 秒，剩余 4,644.829802878667 秒。实际账本为
experiments/prefix_io_v1/gpu-budget-ledger.json，本说明不是替代账本。

此 GPU 指示不授权系统/驱动修改、下载、删除数据、付款/租用实例、远端推送，
也不改变研究范围、性能门槛、成本上界或实验对照。permissions.yaml 原文件、
既有冻结源码、历史授权和失败结果保持原样；本轮现场 effective permission
比原 permissions.yaml 的可选能力更窄。

本轮 normal off01 已完成真实 GPU 执行和正常 shutdown/会话排空；成本迁移失败：
选中步 16,893,473 ns 高于原冻结上界 16,238,752 ns。QUALIFICATION_off.json
明确不允许下一 mode，所以未启动 shadow/on。这是实验门槛，不是缺少用户授权。

历史报告中的“下一轮需授权”和旧 README 的逐臂授权描述只是旧交付时的状态；
本文件及对应用户原文 amendment 记录当前持续授权，不覆盖历史证据。

