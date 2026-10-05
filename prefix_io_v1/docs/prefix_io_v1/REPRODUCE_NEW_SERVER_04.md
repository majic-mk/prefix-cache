# 新服务器第四轮：实际执行命令

工作目录 /root/autodl-tmp/prefix-io-v1-handoff/project。以下为本轮已执行的 CPU 命令；没有 GPU launch 命令。输出文件保持只增不覆写；如需重跑必须换新的 artifact 路径。

## 环境复查

~~~bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
/root/miniconda3/bin/python -I -S artifacts/prefix_io_v1/new-server-03/platform-probe/io_uring_setup_probe.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m prefix_io_control.preflight --controller docs/prefix_io_v1/templates/controller_spec.yaml --permissions experiments/prefix_io_v1/configs/permissions.yaml --capabilities artifacts/prefix_io_v1/new-server-04/capability-report.json
~~~

探针 exit2/EPERM；preflight exit2/BLOCKED。完整 stdout/stderr/argv 分别保存于 new-server-04/io-uring-recheck.json 与 preflight-command.json。还实际执行了 git status --porcelain=v1、git rev-parse HEAD、git branch --show-current，及四个固定 upstream 的 git -C <repo> rev-parse HEAD / status --porcelain=v1 --untracked-files=no，结果见 workspace-*.json 和 source-preservation.json。

## 本轮新增 CPU 测试及候选准备

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/prefix_io_v1_calibration -q -p no:cacheprovider --junitxml=artifacts/prefix_io_v1/new-server-04/calibration-preparation/cpu-tests.xml

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py --smoke-dir experiments/prefix_io_v1/runs/native-prefix-04/details --doc-sizes 64 128 192 --repeats 3 --output artifacts/prefix_io_v1/new-server-04/calibration-preparation/candidate-plan.json

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/prefix_io_v1/new-server-04/calibration-preparation/verify_cpu.py
~~~

32 passed，0 failed/skipped；实际 prepare exit0，生成3个原生候选 job、共9个拟测请求。verify_cpu.py 完成编译、stdlib导入边界和CLI help检查。这些没有生成任何校准秒数、运行真实模型或启动GPU。精确命令记录在 calibration-preparation/commands.json。

--output 使用排他新建，以上原路径已经存在，重放时会拒绝覆盖。不要把 candidate-plan.json 当 v2 曲线或直接交给 LoadPlanner；不要直接运行原作者默认 Pareto benchmark。实际 token、预热、同session launcher、启动配置与真正成本数据仍未具备。

## 交付完整性

~~~bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/python artifacts/prefix_io_v1/new-server-04/pack_delta.py
~~~

打包脚本逐项记录大小和SHA256，并重开ZIP逐文件校验和检查CRC。交付目的文件排他创建，不覆盖旧包。模型、环境和历史完整证据沿用第三轮基础包；本轮包是增量，base archive SHA写在manifest中。本地仅下载和再次校验交付文件，没有本地开发或研究测试。

下一允许阶段仍为P1；平台原生io_uring及真实缓存资格恢复之前，不将此候选准备提升为P2/P3。
