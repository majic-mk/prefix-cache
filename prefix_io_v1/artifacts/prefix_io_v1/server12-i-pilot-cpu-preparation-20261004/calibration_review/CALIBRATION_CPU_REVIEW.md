# 前瞻校准入口的独立 CPU 复核

实际结果为 21/21 PASS，0 failures，0 errors，0 skips；没有发现可绕过真实生产入口的授权、预算、UUID 或原来源门的问题，无需为了 helper 的独立 API 更换生产候选。复核未编辑 root 创建的 calibration_v2 文件，未连接服务器、探测 GPU、运行模型/SDK、修改账本或生成真实 live context/grant。

新 test_standing_calibration_review_cpu.py 执行 standing 模块、C.bind 和 B.load_binding 的真实函数。文件/大资产 closure 验证在定向 shape 测试中以明确的 CPU 内存值代入，所有 subprocess.run/Popen 被禁止，因此这些 shape 测试不等于一次真实站点 binding 或大资产复核。唯一临时 GRANT 路径文件内容是无效的 CPU placeholder，位于 unittest 自己的临时目录，只让错误 existing-grant 分支被执行；不存在可被真实站点使用的授权文档。

已验证：

- 只能派生已有 human_user 的持续 GPU 指令；预算仍为 28,800 秒，原 permission 和 amendment 文件 SHA 固定。
- amendment root/base-ref、来源、重复 JSON、非有限数字、路径穿越、symlink 和 SHA/长度漂移均拒绝。
- C.bind 缺 context 时先失败，尚未执行资源 probe；错误 existing grant 在写 effective permission 或 site binding 之前失败。
- B.load_binding 和 C.bind 的最终 grant 要求 max_attempts=1、max_processes=6、seconds_limit=1200、reserved_seconds=1220 且为确切 int；float、bool、扩大、缺失和外站 UUID/指令均拒绝。
- 有效 permission 的上述 job limits 同样要求 exact 类型；driver/system 权限扩大、GPU UUID 替换或累计小时数超过 8 均拒绝。
- resource shape 对无卡、GPU 占用、错误 driver/UUID、CPU/内存/显存/存储不足保持拒绝；context 对原 ledger active reservation 或剩余预算不足 1220 秒保持拒绝且不写 live context。

standing helper 只用于 C.bind 派生来源信息，本身不是 GPU authority。context 的 max_attempts/max_processes 是只读资源快照附带的提示值；即使缺失或填写别的值，派生 grant 仍固定为 int 1/6。context seconds 的数值相等不用于扩大预算，最终 grant、permission、原 guard 命令和原 ledger 全部固定整型 limits。不要把单独调用 helper 返回的对象当作可执行资格；真正入口仍需真实 resource、完整 revision/site source、持续授权引用和原 guard reservation 全部通过。

## 源及校准语义

逐函数原 AST 对比结果：run_native_cost_experiment.py 的 26 个顶层函数和 native_conditional_cost.py 的 18 个顶层函数全部保持相同；receipt 的 _calibration_a_budget 同样未改。serializer 唯一改变的顶层函数 create_plan 对新 families/seeds 作事先一致性约束；receipt 唯一改变的 load_verified_single_file 是新的源路径元数据，成本公式与准入预算没有改变。

新 prompt-first 为 40100/41100/42100，seeds 为 4029/4030/4031；runner 与 serializer 双侧一致。AB/BA 两组 calibration 和独立 AB validation 拆分不变。receipt 的 verifier 和 serializer FileRef 长度/SHA 逐项等于真实新文件，SDK、原 guard、祖先 source 和 common candidate 的固定 pin 未放宽。全部新候选文件在本复核前后字节一致。

本轮新校准入口仅准备新的独立样本，不保证新 U 会低于原 A-only 预算，不能掩盖旧三次 off 的覆盖失败。native common runner 仍以 bridge=None 执行，尚未测量 on 观测/策略开销；新的校准不是 P4/P5 的收益对照，也不是独立服务 SLO。

## 实际命令及证据

```text
C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe -B -I -S artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/calibration_review/test_standing_calibration_review_cpu.py --output artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/calibration_review/LOCAL_STANDING_REVIEW_FINAL_V2.json
```

命令实际 exit=0。21 项 unittest 日志、前后 source rows 和明确限制保存在 LOCAL_STANDING_REVIEW_FINAL_V2.json；此脚本可放入新服务器 scope 的 calibration_review 子目录直接以原 venv 的 -B -I -S 运行。服务器应使用自己完整的 retained native-recalibration 路径，比对真实文件；本子代理尚未运行服务器命令。

下一 GPU 阶段必须仍受原 8 小时预算、原 guard、完整 source proof、真实设备/原 shutdown/session drain 及阶段协议约束。CPU 来源/参数审核通过不能授予 P4 效果实验资格。
