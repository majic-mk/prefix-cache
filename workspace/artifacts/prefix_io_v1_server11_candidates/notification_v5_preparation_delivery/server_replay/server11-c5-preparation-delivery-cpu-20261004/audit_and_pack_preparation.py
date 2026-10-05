"""Audit unchanged sources and seal four new CPU-only directories; no GPU calls."""
from __future__ import annotations
import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

NAMES = {
    'runtime': 'server11-c5-runtime-preparation-cpu-20261004',
    'review': 'server11-c5-runtime-preparation-review-cpu-20261004',
    'resource': 'server11-c5-cpu-resource-audit-20261004',
    'delivery': 'server11-c5-preparation-delivery-cpu-20261004',
}
LOCK_SHA = '8a8dbf5226cefabfa5100f24aabe9bc4bf48010996d389a96c516c6288164071'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_bytes())

def write_new(path, doc):
    data = (json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(data)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def checked_file(root, rel):
    p = PurePosixPath(rel)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe source reference')
    path = root.joinpath(*p.parts)
    require(path.is_file() and not path.is_symlink(), 'not regular source: '+str(path))
    require(path.resolve().is_relative_to(root.resolve()), 'source escaped root')
    return path

def verify_ref(root, row):
    data = checked_file(root, row['path']).read_bytes()
    require(len(data)==row['bytes'] and digest(data)==row['sha256'], 'source changed: '+row['path'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args=parser.parse_args()
    root=args.root.resolve(strict=True)
    base=root/'artifacts/prefix_io_v1'
    dirs={k:base/v for k,v in NAMES.items()}
    for path in dirs.values():
        require(path.is_dir() and not path.is_symlink() and path.resolve().parent==base, 'invalid new directory')
    delivery=dirs['delivery']
    before=read_json(delivery/'SESSION_BEFORE_PREPARATION.json')
    native_before=read_json(delivery/'NATIVE_V6_BEFORE_PREPARATION.json')
    anchors=read_json(delivery/'ANCHOR_REFS_BEFORE_PREPARATION.json')
    rows=before['source_refs']+native_before['source_refs']+anchors['refs']
    for row in rows:
        verify_ref(root,row)
    ledger_path=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'
    ledger_data=ledger_path.read_bytes();ledger=json.loads(ledger_data)
    require(digest(ledger_data)==before['ledger_sha256'], 'GPU ledger changed')
    require(ledger['gpu_wall_seconds']==before['gpu_wall_seconds'], 'GPU wall budget changed')
    require(ledger.get('active_reservation') is None and before['active'] is None, 'active GPU reservation')
    lock_path=dirs['runtime']/'PREPARATION_SOURCE_LOCK.json'
    lock_data=lock_path.read_bytes();lock=json.loads(lock_data)
    require(digest(lock_data)==LOCK_SHA, 'preparation source lock changed')
    require(lock['gpu_launch_allowed'] is False and lock['gpu_qualified'] is False, 'GPU lock qualification')
    candidate=base/'server11-p4-notification-candidate-v5-cpu-20261003'
    for row in lock['files']:
        verify_ref({'preparation':dirs['runtime'],'candidate':candidate}[row['scope']],row)
    require(len(lock['files'])==80, 'unexpected server preparation closure')
    cpu_path=dirs['runtime']/'SERVER_CPU_RESULT_01/CPU_RESULT.json'
    cpu=read_json(cpu_path)
    require(cpu['status']=='PASS_CPU_WIRING_ONLY' and cpu['tests']==22 and cpu['failures']==0 and cpu['errors']==0 and cpu['skipped']==0, 'CPU wiring failed')
    require(cpu['source_files_verified_before']==80 and cpu['source_files_verified_after']==80 and cpu['source_lock_sha256']==LOCK_SHA, 'CPU wiring source closure')
    require(cpu['actual_gpu_runs']==0 and cpu['actual_vllm_install'] is False and cpu['forbidden_modules_imported']==[], 'CPU wiring native activity')
    independent_path=dirs['review']/'SERVER_REVIEW_01/TEST_RESULT.json'
    independent=read_json(independent_path)
    require(independent['status']=='PASS' and independent['passed']==22 and independent['failed']==0 and independent['skipped']==0, 'independent CPU review failed')
    require(independent['location']=='server_cpu' and independent['sources_unchanged'] is True and independent['source_before']==independent['source_after'] and independent['preparation_source_lock_sha256']==LOCK_SHA, 'independent source mismatch')
    require(independent['gpu_workloads_run']==0 and independent['actual_vllm_collector_install'] is False and independent['forbidden_modules_imported']==[], 'independent native activity')
    for row in independent['source_after']:
        path=Path(row['path'])
        require(path.resolve().is_relative_to(root), 'review reference escaped project')
        verify_ref(root,dict(row,path=str(path.relative_to(root))))
    guards=[]
    for tag in ('GPU_RUN_ENTRY_BLOCK','GPU_CONTROL_ENTRY_BLOCK','GPU_VERIFY_ENTRY_BLOCK'):
        result=read_json(dirs['runtime']/(tag+'_RESULT.json'))
        stdout_path=dirs['runtime']/(tag+'_STDOUT.log')
        doc=read_json(stdout_path)
        require(result['exit']==2 and doc['status']=='GPU_BLOCKED_NO_C5_RECEIPT_AND_CPU_ENVIRONMENT_QUALIFICATION' and doc['gpu_started'] is False and doc['gpu_qualified'] is False and doc['native_cost_receipt'] is None and doc['performance_claim'] is False, 'GPU entry guard failed')
        require((dirs['runtime']/(tag+'_STDERR.log')).read_bytes()==b'', 'GPU entry unexpected stderr')
        guards.append(dict(tag=tag,expected_exit=2,actual_exit=result['exit'],status=doc['status']))
    resource=read_json(dirs['resource']/'SERVER_REPLAY/RESOURCE_AUDIT.json')
    require(resource['verified_raw_count']==132 and resource['scored_pairs']==42 and resource['audit_runs_benchmark'] is False and resource['audit_GPU_operations']==0, 'resource replay mismatch')
    preserved=resource['preserved_qualification']
    require(preserved['decision']=='RETAIN_OFF_STOP_OR_RESOURCE_LIMITED' and preserved['CPU_candidate_qualified'] is False and preserved['GPU_runs_allowed']==0 and preserved['P4_complete'] is False, 'old qualification changed')
    nodes=[str(x) for x in Path('/dev').glob('nvidia*')]
    quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    git=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,check=True,capture_output=True,text=True).stdout
    require(nodes==before['GPU_nodes'] and git==before['tracked_git_status'], 'device or tracked source status changed')
    require(quota==before['cpu_max'] and memory==before['memory_max'], 'cgroup settings changed')
    free=shutil.disk_usage(base).free
    require(free>=8*1024**3, '8 GiB free space floor')
    audit=dict(status='PASS_CPU_PREPARATION_AND_UNCHANGED_BOUNDARIES',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        server_root=str(root),new_directory_scope=NAMES,source_references_verified=len(rows),unique_prior_references=len({x['path'] for x in rows}),
        frozen_prior_sources_and_anchors_unchanged=True,source_lock_sha256=LOCK_SHA,preparation_source_count=len(lock['files']),
        server_cpu_wiring_tests=cpu['tests'],server_independent_tests=independent['tests'],independent_source_count=len(independent['source_before']),
        GPU_entry_guard_checks=guards,GPU_runs=0,formal_benchmark_runs=0,actual_vllm_collector_install=False,native_execution_verified=False,
        cpu_timing_qualified=False,new_probe_cost_measured=False,old_score_covers_new_runtime=False,gpu_qualified=False,P4_complete=False,performance_claim=False,
        CPU_candidate_qualified=False,preserved_score_decision=preserved['decision'],raw_results_read_only_verified=132,scored_pairs=42,
        gpu_ledger_sha256=digest(ledger_data),gpu_wall_seconds=ledger['gpu_wall_seconds'],gpu_budget_remaining_seconds=8*3600-ledger['gpu_wall_seconds'],active_gpu_reservation=None,
        GPU_nodes=nodes,cpu_max=quota,memory_max=memory,tracked_git_status=git,disk_free_bytes=free,
        result_refs=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in (cpu_path,independent_path,dirs['resource']/'SERVER_REPLAY/RESOURCE_AUDIT.json')])
    write_new(delivery/'SERVER_FINAL_AUDIT.json',audit)
    report='''# C5 运行入口 CPU 准备交付（2026-10-04）

本轮结论：CPU 接线与独立审查通过；未运行 GPU，未完成 P4 真机或性能验证。

实际改动限于新的准备目录：复制并接入原启动器、控制器和验证器，增加有界值观测和 CPU 反例；冻结 C5 reactor/collector、C4、原 runtime_v4/native_v6 与旧实验未修改。没有重做缓存引擎或模型执行器，没有扩展候选、干扰额度或研究任务。

## 修复内容与语义

- 将 capture 的 owner run ID 与 bridge 对齐，保持 frontend request ID 和原 native_request_id、128 帧检查。
- 只在 on 接入已有 C5 通知等待；off/shadow 不安装等待和 Queue.get 包装。Queue 实例、FIFO、原 get 参数、timeout、返回值和异常保持。记录上限为 16 条值观测、8 条错误，持有弱引用；未写资源完成或释放状态。
- 匹配 wake 可以在 get 之前到达并清 registration；该竞态只有真实匹配 token 能通过，旧 wake、一般消息或 deferral 均不足以证明通知已运行。
- detach 发布唤醒后保留原 drain 流程，待其返回才关闭和导出观测。异常路径也在原 drain/shutdown 后关闭，未伪造 arm 清理。
- 新 GPU run/control/verify 三个入口在解析配置、触碰预算或导入模型前无条件阻断。旧 C4/v6 成本凭据不能给新 C5 启动器资格。

## 真实服务器执行

全部在当前 server11 项目执行，Python 3.12.3，`CUDA_VISIBLE_DEVICES=''`，使用 `.venv/bin/python -B -I -S`。

- `freeze_cpu_preparation.py --candidate-root <C5> --runtime-v4-root <runtime_v4>`：冻结 80 个文件，锁 SHA-256 为 `'''+LOCK_SHA+'''`。
- `run_cpu_preparation.py --candidate-root <C5> --author-root <project> --output-dir SERVER_CPU_RESULT_01`：22/22，失败/错误/跳过均为 0，前后核验 80 个冻结文件。
- `run_review_cpu.py --preparation-root <new-runtime> --preparation-lock-sha256 <lock> --candidate-root <C5> --native-root <native-v6> --race-review-root <C5-review> --author-root <project> --output-dir SERVER_REVIEW_01 --location server_cpu`：独立 22/22；前后实际源码相同。
- 新 run、control、verify 脚本分别运行：退出码均为预期 2，返回 GPU_BLOCKED_NO_C5_RECEIPT_AND_CPU_ENVIRONMENT_QUALIFICATION，未启动 GPU。
- `audit_resource_results.py --benchmark-root <old-C5-benchmark> --output-dir SERVER_REPLAY`：只读验证旧 132 份结果、42 个计分对；没有重跑正式计分。

精确绝对路径、参数、环境、退出码、耗时、stdout/stderr 均保存于各目录 `*_COMMAND.json`、`*_RESULT.json`、`*_STDOUT.log`、`*_STDERR.log`。两组测试分别报告；本地和服务器重复执行不相加为额外覆盖。event/native/128 帧输入是明确的 CPU 合成元数据，未执行真实 collector.install、模型、CUDA 或物理 I/O。

## 证据与边界

运行结果在 runtime 的 `SERVER_CPU_RESULT_01/CPU_RESULT.json`、review 的 `SERVER_REVIEW_01/TEST_RESULT.json`、resource 的 `SERVER_REPLAY/RESOURCE_AUDIT.json`。封存审计在本目录 `SERVER_FINAL_AUDIT.json`，备份 receipt 在 `SERVER_BACKUP_RECEIPT.json`。

审计逐字节核验 519 个旧 Python 源引用及 6 个权限/协议/历史结果锚点；旧 GPU 账本和跟踪代码不变。冻结 manifest 中服务器结果为 null 的字段是冻结时状态，保留原值，实际运行记录由独立结果引用。归档只封存四个本轮目录，不复制模型、私有编译缓存或旧 GPU 数据，不删除已有数据。

## 当前结论与下一允许阶段

旧基准在 40 ms 合成场景测得 worker+producer CPU 中位数从 33.081 ms 到 8.747 ms（约 73.6%），但完整 CPU 资格因 80/100 ms 对照受半核配额限流而失败；132 个样本全部保留。120 ms 总 wall 包含 producer 固定持有时间，不等于 deadline 晚 20 ms。新 Queue 观测/完整启动器未计分，旧 73.6% 不能转作本轮入口的性能资格，更不能证明 GPU、TTFT、吞吐或论文收益。

本轮 GPU 0 次，正式计分新增 0 次；原累计 GPU 用量 22380.561257688794 秒，原 8 小时预算剩余约 6419.44 秒（1.78 小时），没有预留变化。实例仍无 NVIDIA 设备，CPU 配额 `50000 100000`（0.5 核）、内存 2 GiB；仍不具备本轮 GPU 实验条件。

下一允许 CPU 工作是构建绑定新 C5 完整入口与观测的严格成本凭据工厂并冻结；后续需要足够 CPU 配额下的完整入口成本资格。真实原生校准及 off→shadow→on GPU 验证须另有新代码、新作业和限定预算的明确授权。保持现有模型、128 tokens、单文件任务和阈值，不自动租 GPU、不扩大研究范围。现阶段不能宣称 P4 完成或已有论文级提升。
'''
    with (delivery/'CPU_PREPARATION_REPORT.md').open('x',encoding='utf-8') as f:f.write(report)
    write_new(delivery/'SEAL_COMMAND.json',dict(command=[sys.executable]+sys.argv,GPU_runs=0,formal_benchmark_runs=0))
    content={}
    for top in dirs.values():
        for path in sorted(top.rglob('*')):
            require(not path.is_symlink(), 'symlink in new evidence')
            if path.is_dir():
                continue
            require(path.is_file() and path.resolve().is_relative_to(top), 'non-regular evidence')
            require(path.stat().st_size<=10*1024**2, 'evidence too large')
            content[str(path.relative_to(base))]=path.read_bytes()
    require(len(content)<=300 and sum(map(len,content.values()))<=30*1024**2, 'bounded archive limit')
    manifest=dict(scope='FOUR_NEW_CPU_PREPARATION_DIRECTORIES_ONLY',data_file_count=len(content),raw_bytes=sum(map(len,content.values())),
        GPU_runs=0,formal_benchmark_runs=0,files=[dict(path=n,bytes=len(b),sha256=digest(b)) for n,b in sorted(content.items())])
    manifest_data=(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
    archive=delivery/'C5_PREPARATION_CPU_EVIDENCE.tar.gz'
    with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
        for name,data in [('ARCHIVE_CONTENTS_MANIFEST.json',manifest_data)]+sorted(content.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;info.mtime=0
            tar.addfile(info,io.BytesIO(data))
    require(archive.stat().st_size<=20*1024**2,'compressed archive too large')
    for name,data in content.items():
        require((base/name).read_bytes()==data,'evidence changed during seal')
    require(ledger_path.read_bytes()==ledger_data,'GPU ledger changed during seal')
    for row in rows:verify_ref(root,row)
    receipt=dict(status='PASS_SEALED_CPU_PREPARATION',archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=digest(archive.read_bytes())),
        manifest=dict(file='ARCHIVE_CONTENTS_MANIFEST.json',bytes=len(manifest_data),sha256=digest(manifest_data)),data_files=len(content),raw_bytes=manifest['raw_bytes'],
        GPU_runs=0,formal_benchmark_runs=0,source_lock_sha256=LOCK_SHA,prior_references_verified=len(rows),gpu_ledger_sha256=digest(ledger_data))
    write_new(delivery/'SERVER_BACKUP_RECEIPT.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
