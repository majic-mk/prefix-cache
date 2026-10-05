# 第五轮实际命令

服务器工作目录 /root/autodl-tmp/prefix-io-v1-handoff/project。本轮只执行CPU诊断；没有模型/GPU启动、系统设置修改、依赖安装或模型下载。

~~~bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
gcc -O2 -Wall -Wextra -Werror artifacts/prefix_io_v1/new-server-05/io_uring_syscall_probe.c -o artifacts/prefix_io_v1/new-server-05/io_uring_syscall_probe
artifacts/prefix_io_v1/new-server-05/io_uring_syscall_probe

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/work/py-kvcache .venv/bin/python artifacts/prefix_io_v1/new-server-05/native_ring_probe.py

/usr/sbin/capsh --decode=00000000a80425fb
~~~

结果：C编译exit0；C探针exit2、setup EPERM；作者ring探针exit2、setup EPERM；capsh解码exit0。分别见compile-probe.json、raw-syscalls.json、native-ring.json、capability-decode.json。全部命令stdout/stderr已保存，失败没有删除。

平台审计用Python标准库读取/proc/self/status、/proc/1/status、uid/gid maps、namespace标识、attr/current、相关sysctl和选定mountinfo；用shutil.which与文件stat检查常见本地容器管理入口，未连接管理socket、未扫描远端端口、未读取进程环境或平台凭据。结果platform-audit.json。

有限授权记录已保存。权限验证通过现有prefix_io_control.config.read_yaml和validate_permissions对更新前后文件进行比较：严格schema通过，解析后的权限/预算值相等。完整代码argv和结果在permission-validation.json。

诊断源码可重复执行；采集JSON若重跑须用新目录以保留本轮结果。C二进制由本轮服务器编译，交付包提供源码与哈希，不依赖传递该二进制。

下一步不是继续在同一配置下重复探针，而是取得平台支持的具体最小修复入口。授权补充不等于宿主机控制能力，当前环境NO_GO不等于架构在所有环境中不可行。
