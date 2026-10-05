"""Collect actual CPU evidence, without GPU/model/backend imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
REL = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LEDGER_SHA = '60368b94c494d15227eb528273e7b4a1a20119c6f371767ffde5cc96aa8f52f9'
def require(ok, reason):
    if not ok:
        raise ValueError(reason)
def read(path):
    return json.loads(path.read_bytes())
def put(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')
def ref(root, path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(root).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
def unittest_count(text):
    counts = [int(x) for x in re.findall(r'Ran (\d+) tests? in ', text)]
    require(counts and 'FAILED' not in text and 'skipped=' not in text, 'actual unittest completion')
    return sum(counts)
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', type=Path, required=True)
    root = p.parse_args().project_root.resolve(strict=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only command')
    d = root/REL
    ledger_raw = (root/LEDGER).read_bytes()
    require(hashlib.sha256(ledger_raw).hexdigest() == LEDGER_SHA, 'unchanged original ledger')
    ledger = json.loads(ledger_raw)
    require(ledger['active_reservation'] is None, 'idle original ledger')
    suites = [
        ('protocol','CPU_PRERENT_PROTOCOL_TEST_01','SERVER_PROTOCOL_CPU_RESULT.json',30),
        ('activation','CPU_PRERENT_FINITE_ACTIVATION_TEST_01','SERVER_ACTIVATION_CPU_RESULT.json',19),
        ('host_control','CPU_PRERENT_HOST_CONTROL_TEST_01','SERVER_HOST_CONTROL_CPU_RESULT.json',29),
        ('review_protocol','CPU_PRERENT_PROTOCOL_REJECTION_REVIEW_01','SERVER_PROTOCOL_REJECTION_REVIEW_CPU_RESULT.json',7),
        ('review_ids','CPU_PRERENT_ORIGINAL_ID_REVIEW_01','SERVER_ORIGINAL_ID_REVIEW_CPU_RESULT.json',6),
        ('review_capability','CPU_PRERENT_FINITE_CAPABILITY_REVIEW_01','SERVER_FINITE_CAPABILITY_REVIEW_CPU_RESULT.json',9),
        ('review_capacity','CPU_PRERENT_OBSERVER_CAPACITY_REVIEW_01','SERVER_OBSERVER_CAPACITY_REVIEW_CPU_RESULT.json',3),
        ('review_raw_ids','CPU_PRERENT_RAW_ID_REVIEW_01','CPU_PRERENT_RAW_ID_REVIEW_01_EVIDENCE.json',3),
        ('review_host_refusal','CPU_PRERENT_HOST_REFUSAL_REVIEW_01','CPU_PRERENT_HOST_REFUSAL_REVIEW_01_EVIDENCE.json',6),
        ('review_host_fallback','CPU_PRERENT_HOST_FALLBACK_REVIEW_01','CPU_PRERENT_HOST_FALLBACK_REVIEW_01_EVIDENCE.json',2),
        ('runner','CPU_PRERENT_FINAL_RUNNER_TEST_01','runner/SERVER_CPU_RESULT.json',47),
        ('entry_v3','CPU_PRERENT_ENTRY_V3_TEST_01',None,10)]
    records=[]
    for name,tag,evidence_name,expected in suites:
        result=read(d/(tag+'_RESULT.json'))
        command=read(d/(tag+'_COMMAND.json'))
        require(result['exit']==0 and result['GPU_runs']==0 and command['CUDA_VISIBLE_DEVICES']=='','actual CPU PASS '+tag)
        evidence=read(d/evidence_name) if evidence_name else None
        if evidence:
            require(evidence['status'].startswith('PASS'),'actual evidence PASS '+name)
            for key in ('failures','errors','skips','test_failures','test_errors'):
                require(evidence.get(key) in (None,0,[]),'zero failure/error/skip '+name)
        if name=='runner':
            require(len(evidence['commands'])==3 and all(c['exit_code']==0 for c in evidence['commands']),'three runner suites')
            count=sum(unittest_count(c['stderr']) for c in evidence['commands'])
        elif name=='entry_v3':
            count=unittest_count((d/(tag+'_STDERR.log')).read_text())
        else:
            count=evidence['tests_run']
        require(count==expected,'actual test denominator '+name)
        records.append(dict(suite=name,tests=count,actual_server_CPU=True,
            command_ref=ref(root,d/(tag+'_COMMAND.json')),result_ref=ref(root,d/(tag+'_RESULT.json')),
            evidence_ref=ref(root,d/evidence_name) if evidence_name else ref(root,d/(tag+'_STDERR.log'))))
    require(sum(r['tests'] for r in records)==171,'unique final denominator')
    lock=read(d/'PRERENT_SOURCE_LOCK_V3.json')
    proof=read(d/'PRERENT_SOURCE_PROOF_V3.json')
    preflight=read(d/'SERVER_PRERENT_ENTRY_PREFLIGHT.json')
    plan=read(d/'STRONG_RAW_CPU_PLAN_TEMPLATE.json')
    lock_ref=ref(root,d/'PRERENT_SOURCE_LOCK_V3.json')
    require(proof['status']=='PASS_FULL_CPU_SOURCE_BYTES' and proof['source_lock_ref']==lock_ref and
        len(lock['files'])==proof['source_count']==4851 and proof['actual_gpu_runs']==0,'actual final freeze')
    require(preflight['status']=='PASS_CPU_BOUNDED_STRONG_U_QUALIFICATION_ENTRY' and
        preflight['source_lock_ref']==lock_ref and preflight['actual_GPU_runs']==0,'actual CPU entry')
    require(plan['source_lock_ref']==lock_ref and plan['gpu_uuid'] is None and
        plan['cpu_preparation_only'] is True and plan['GPU_qualification_issued'] is False,'unbound CPU template')
    protections=[read(d/('SOURCE_PROTECTION_'+phase+'.json')) for phase in ('BEFORE','AFTER')]
    for phase,record in zip(('BEFORE','AFTER'),protections):
        require(record['status']=='PASS_COMPLETE_SOURCE_BYTES_AND_UNCHANGED_IDLE_LEDGER' and
            record['phase']==phase and record['complete_source_rows_verified']==4816,'old protected source bytes')
    for key in ('source_lock_ref','source_map','actual_completed_GPU_archive_ref','original_ledger_ref'):
        require(protections[0][key]==protections[1][key],'old protected bytes unchanged')
    lb,la=read(d/'LOCAL_USER_WORK_BEFORE.json'),read(d/'LOCAL_USER_WORK_AFTER.json')
    require(la['status']=='PASS_ORIGINAL_LOCAL_USER_FILES_UNCHANGED' and la['files_verified']==14 and
        lb['files']==la['files'],'local user work unchanged')
    sdk=read(d/'PRERENT_PROJECT_SDK_CPU_ASSET_RESULT.json')
    require(sdk['status']=='PASS_PROJECT_ASSET_BYTES_DRIVER_DEFERRED','asset-only SDK status')
    for tag in ('CPU_PRERENT_FULL_SOURCE_FREEZE_V3_01','CPU_PRERENT_ACTUAL_RAW_TEMPLATE_03',
        'CPU_PRERENT_ENTRY_PREFLIGHT_V3_01','CPU_PRERENT_PROJECT_SDK_AUDIT_01','CPU_PRERENT_SOURCE_AFTER_01'):
        result=read(d/(tag+'_RESULT.json'))
        require(result['exit']==0 and result['GPU_runs']==0,'actual final CPU closure '+tag)
    require((root/LEDGER).read_bytes()==ledger_raw,'ledger unchanged during collector')
    forbidden=sorted(n for n in sys.modules if n.split('.')[0] in ('torch','vllm','py_kvcache','cupy','numpy'))
    require(not forbidden,'no model/native backend imported')
    free=shutil.disk_usage(root).free
    state=subprocess.run(['git','diff','--name-only'],cwd=root,capture_output=True,text=True,check=True).stdout.splitlines()
    require(not state,'tracked server source unchanged')
    decision=dict(schema='actual_server_gpu_prerental_CPU_completion_v1',
        status='PASS_CPU_READY_FOR_ONE_BOUNDED_STRONG_OFF_QUALIFICATION',completed_UTC=datetime.now(timezone.utc).isoformat(),
        artifact_namespace_date='20261004',actual_GPU_runs_this_preparation=0,
        compiler_executions_this_preparation=0,actual_model_or_shared_library_loads_this_preparation=0,
        data_deletions=0,system_or_driver_modifications=0,final_unique_server_CPU_tests=171,
        failures=0,errors=0,skips=0,final_test_records=records,superseded_or_repeated_tests_added_to_denominator=False,
        final_source_lock_ref=lock_ref,final_source_proof_ref=ref(root,d/'PRERENT_SOURCE_PROOF_V3.json'),
        complete_source_files_verified=4851,complete_source_bytes_verified=proof['total_bytes_verified'],
        original_protected_source_files_unchanged=4816,original_local_user_files_unchanged=14,
        actual_completed_historical_GPU_archive_unchanged=protections[1]['actual_completed_GPU_archive_ref'],
        original_idle_ledger_ref=ref(root,root/LEDGER),original_GPU_remaining_seconds=28800-ledger['gpu_wall_seconds'],
        PRIMARY_free_bytes=free,current_NVIDIA_device_nodes=sorted(str(p) for p in Path('/dev').glob('nvidia*')),
        project_SDK_bytes_verified_driver_deferred=True,full_SDK_runtime_qualified=False,
        natural_service_trace_bound=False,independent_development_deadline_ns=None,
        actual_strong_domain_GPU_cost_cells_qualified=0,actual_private_GPU_capabilities_issued=0,
        actual_native_controller_reserve_receipts_issued=0,method_effect_verified=False,formal_goodput_allowed=False,
        CPU_entry_preflight_ref=ref(root,d/'SERVER_PRERENT_ENTRY_PREFLIGHT.json'),
        CPU_raw_template_ref=ref(root,d/'STRONG_RAW_CPU_PLAN_TEMPLATE.json'),
        CPU_failure_history_preserved=['CPU_PRERENT_ACTUAL_RAW_TEMPLATE_01','CPU_PRERENT_ACTUAL_RAW_TEMPLATE_02'],
        next_allowed_GPU_phase=dict(label='server12-strong-u-qual-off01',mode='off',arm='U',phase='qualification',
            maximum_execution_seconds=300,cleanup_reserve_seconds=20,maximum_total_reserved_seconds=320,
            required_live_checks=['actual idle device and UUID','real mounted driver and original SDK',
                'same full source closure','original cumulative idle ledger','PRIMARY storage'],
            human_standing_authorization_reused=True,new_human_approval_required=False,automatic_cloud_rental=False),
        subsequent_exact_cell_acquisition=dict(conditional_on_actual_off_guard_and_drain=True,
            maximum_execution_seconds=900,cleanup_reserve_seconds=20,maximum_total_reserved_seconds=920,
            storage_reserve_bytes=512*1024**2,not_started=True),
        remaining_effect_requirements=['actual strong off complete outputs and shutdown/native/OS drain',
            'real finite-cell GPU costs and independent heldout','independent predeclared development deadline',
            'actual finite-table development shadow and eligible preview','actual closed host-control reserve',
            'natural heldout for formal service effect'],
        unsupported_observation_shapes_remain_unknown=['zero scheduled metadata ordinal','empty output prefill'],
        tracked_server_changes=state,forbidden_backend_modules_imported=forbidden)
    put(d/'FINAL_PRERENT_CPU_DECISION.json',decision)
    lines=[]
    for _,tag,_,_ in suites:
        argv=read(d/(tag+'_COMMAND.json'))['argv']
        lines.append('CUDA_VISIBLE_DEVICES=\'\' '+' '.join(argv))
    for tag in ('CPU_PRERENT_FULL_SOURCE_FREEZE_V3_01','CPU_PRERENT_ACTUAL_RAW_TEMPLATE_03',
        'CPU_PRERENT_ENTRY_PREFLIGHT_V3_01','CPU_PRERENT_PROJECT_SDK_AUDIT_01','CPU_PRERENT_SOURCE_AFTER_01'):
        lines.append('CUDA_VISIBLE_DEVICES=\'\' '+' '.join(read(d/(tag+'_COMMAND.json'))['argv']))
    report=f"""# 租用 GPU 前置 CPU 交付

本轮可在无卡阶段执行的实现、服务器 CPU 测试、完整源码冻结和原数据保护核验已完成。下一允许步骤是单次强基线 U/off 资格验证。CPU 完成不等于系统 GPU 可行性、P4 策略效果或论文收益已经成立。

完成时间 UTC：{decision['completed_UTC']}。20261004 是沿用的本轮固定目录命名。
服务器 connect.westd.seetacloud.com:24828；项目 {root}。密码未写入交付材料。

## 实际改动

runner/ 沿用作者 LLMEngine、py-kvcache、LoadPlanner、精确 Prefix Cache、共享 staging、预加载、复制合并及原异步流水线；只增加薄运行接线、完整输出/生命周期检查和有限候选绑定。普通 preload 保留原 ready、owner 和资源路径，仅加一次有限决策；原批量复制合并保留。

activation/ 为独立私有成本资格接线，CPU 表、公开字典、布尔 GPU 标记均不能开启策略；当前模型/内核/条件不匹配或不在真实覆盖集合时回到原路径。activation/control_observation/ 记录 scheduler、sampling、output、controller 四类原方法的紧凑 host 区间和实际 preview 尝试。host wall 可含原等待，禁止拿 host wall 减 CUDA 时长编造纯 CPU 控制成本。reserve 只能由真实 development shadow、独立 deadline、已结束的原 guard 和实际 eligible 窗口连接后发行。

protocol/ 使用原 P3 的 12 个请求作资格负载，原 prompt token IDs、到达顺序和原 0.5 请求/秒保持；每个请求要求完整 128 输出 tokens。它是受控机制数据，不能作为自然业务 heldout 或正式收益证据。

entry_control_v3.py、freeze_prerental_sources_v3.py、audit_project_sdk_cpu.py 提供源/资产检查和明确限额入口；复用人的持续 GPU 授权和原 8 小时账本，实际出现设备后才绑定 UUID。review/ 独立复核实际原 API、拒绝条件、回退及成本来源。共同接线修复与研究策略分开；关闭新策略保留原执行路径，共同观测仍有开销，不能称零开销回退。

## 服务器实际测试与命令

| 最终套件 | 通过 |
| --- | ---: |
| 协议和原受控资格负载 | 30 |
| 有限成本资格拒绝/原估计器数学 | 19 |
| host 观测/reserve 接口 | 29 |
| 原执行器/有限当前条件/激活接口 | 47 |
| 根入口 v3 | 10 |
| 七组独立复核 | 36 |
| 合计 | 171 |

最终套件 **171 通过，0 失败，0 错误，0 跳过**。旧版本和重复测试没有加入分母。CPU fixture 不能签发真实 GPU 能力。每条命令完整 argv、cwd、CUDA_VISIBLE_DEVICES、stdout、stderr、退出码和耗时保存在对应 CPU_PRERENT_*_COMMAND.json、_STDOUT.log、_STDERR.log、_RESULT.json。

实际最终命令如下（均只在服务器 CPU 执行）：

~~~sh
{chr(10).join(lines)}
~~~

完整冻结实际读取 **4,851 文件 / {proof['total_bytes_verified']:,} 字节**，源锁 SHA-256：{lock_ref['sha256']}。原保护清单 **4,816 文件**、历史真实 GPU 交付归档和原账本 BEFORE/AFTER 字节相同；本地原有 **14 用户文件** SHA 相同。没有删除已有实验数据或缓存。

实际 prepare-plan 前两轮拒绝日志保留：01 为调用方传绝对路径；02 为源锁漏列已有强基线配置。v3 补齐真实原文件，并由 thin runner.REQUIRED 检查必需执行/生命周期闭包；03 实际生成成功。模板 gpu_uuid=null、cpu_preparation_only=true、不可执行，不能充当 GPU 实验或真实成本资格。

## 真实 GPU 情况、资源与下一允许阶段

本轮 **GPU 作业 0、编译器运行 0、模型/共享库加载 0、数据删除 0、驱动/系统修改 0**；NVIDIA 设备节点 {decision['current_NVIDIA_device_nodes']}。没有租赁、支付或下载模型。

PRIMARY 当前剩 **{free/1024**3:.2f} GiB**，首轮只预留 128 MiB 并保留 8 GiB 底线，目前无需扩盘。原 8 小时累计 GPU 预算剩 **{decision['original_GPU_remaining_seconds']:.3f} 秒，约 {decision['original_GPU_remaining_seconds']/60:.2f} 分钟**，账本 SHA 未变。

项目内 SDK 字节已核验；无卡模式的宿主驱动为零字节占位，完整 SDK runtime 未资格通过。切 GPU 模式后必须核验真实驱动、实际 UUID/空闲资源、完整源码、存储和剩余预算，不匹配即停，不修改系统/驱动、不使用 stub 库冒充驱动。

切当前实例到 GPU 模式后，下一入口为：

~~~sh
.venv/bin/python -B {REL}/entry_control_v3.py launch --project-root {root}
~~~

此 launch **本轮未执行**。入口复用已有授权，实时核验后仅执行一次 server12-strong-u-qual-off01，强 U/off 最多 300 秒，加 20 秒收尾。真实完整输出、原 shutdown 返回、native/OS 会话排空及原 guard 闭合全部通过后，才允许单有限成本单元采集（最多 900 秒加 20 秒收尾，存储预留 512 MiB）。后续仍受原累计预算限制，不保证现有余量可以完成全部阶段。

真实强域成本、有限策略动作、controller reserve、性能收益均未验证。正式效果阶段还需要独立预声明 deadline/SLO、实际 development shadow 和自然 heldout 数据；不从 U 的慢步骤反推有利预算，不把旧单文件诊断当当前强基线成本表。零 scheduled metadata/空输出 prefill 等不支持形状保留 UNKNOWN，不能丢弃后宣称全负载通过。现阶段不承诺提升或论文结论。

## 证据

本报告与 FINAL_PRERENT_CPU_DECISION.json 位于 {root}/{REL}。同目录保留全部本轮代码、配置、实际 CPU 原始日志、源锁/证明、保护清单和独立复核。交付归档只包含小型代码/报告/配置/日志，不嵌入模型或 SDK；实际归档及本地逐文件 SHA 核验另见 PRERENT_CPU_DELIVERY_RESULT.json 和 LOCAL_CPU_AUDIT_BYTE_VERIFICATION.json。
"""
    with (d/'FINAL_PRERENT_CPU_REPORT.md').open('x',encoding='utf-8') as f:
        f.write(report)
    with (d/'FINAL_LEDGER_SNAPSHOT.json').open('xb') as f:
        f.write(ledger_raw)
    print(json.dumps({k:v for k,v in decision.items() if k!='final_test_records'},ensure_ascii=False))
if __name__=='__main__':
    main()
