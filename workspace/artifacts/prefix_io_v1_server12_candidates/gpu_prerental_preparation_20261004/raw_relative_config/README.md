本次限定追加修复父子进程配置路径：真实首 child 尚未加载框架便被原 `R.safe` 拒绝，因为父进程 source patch 生成 `str(CONFIG_PATH)` 绝对路径。原严格路径门禁保留。

新增 `runner/strong_native_cost_runner_v3.py`，相对 v2 仅 `source_patch` 一处 replacement target 改为 `CONFIG_PATH.relative_to(root).as_posix()`。原 source/UUID/minor/permission/guard/budget/storage 门禁、原 executor/cache/native I/O、原六个 fresh process、原统计 verifier 全部保留。原16文件、v2、节点 helper 不变。

实际 CPU 回归执行原 patched `execute_parent`，在第一条 child argv 已生成后由明确 CPU-only subprocess 拒绝 stub 截获并 abort，没有创建子进程、加载模型或发出正 GPU receipt。验证 child config 处于项目内、为 POSIX 相对路径、与父 original guard expected command 的 config 一致、继承原 session；另验证项目外路径未到 subprocess 即失败。所有临时原 parent fixture 文件均在临时目录丢弃，不能作为 GPU 证据。

服务器 CPU 命令（project cwd，D 为服务器新准备目录）：

```
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S D/raw_relative_config/test_relative_parent_config_cpu.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S D/runner/strong_native_cost_runner_v3.py --help
```

本机实际 3/3 CPU 回归与 v3 CLI PASS，GPU query/RPC/GPU runs 均 0。root 将新增 v3/test 纳入 cal02 新完整源锁、配置、intent、作业名后才可启动；不复用已失败 cal01 或改其证据。
