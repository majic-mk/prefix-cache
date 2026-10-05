# 本截点复现说明

当前P3未完成；不要把证据文件的存在当stage验收。

工作区：/root/autodl-tmp/prefix-io-v1-handoff/project；SSH当前20739，凭据不写入交付包。

每个已执行GPU命令由 p3-reproduction-at-storage-block.json 精确索引，29个原wrapper保留出入时刻、elapsed、exit、drain与命令。对应*-plan.json是唯一可供guarded runner复现的计划；新复跑必须新label/output重新冻结与预留。

CPU完整矩阵用run_aio_cpu_tests.py禁CUDA guard，见master qualification所列4processes与XML；finite/capacity03、patch-roundtrip guards、capacity analysis各命令收据完整保留。单次无guard通过不单独作为guarded资格。不要把旧历史fixture作用域扩大到当前生命周期。

本次closeout analyzer只做纯stdlib元数据核对：.venv/bin/python -I -S experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --manifest artifacts/prefix_io_v1/server08-p3-16/p3-closeout-manifest-at-storage-block.json --output artifacts/prefix_io_v1/server08-p3-16/p3-closeout-analysis-at-storage-block.json。预期P3_INCOMPLETE，缺8次GPU且lineage未闭合；分析器status不自动授予P4。

注册source3048份、model权重/私有cache payload及16个库二进制不打包；源与compiled身份通过SHA绑定现服务器。证据包附精确文件manifest，解压后验证每项SHA。source snapshot保留08所有冻结输入实际内容及alias映射；尚未是完整P3最终snapshot。
