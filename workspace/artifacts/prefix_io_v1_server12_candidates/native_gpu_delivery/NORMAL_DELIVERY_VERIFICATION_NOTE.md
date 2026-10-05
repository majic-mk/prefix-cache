# Normal off01 交付归档核验

本轮实际 normal GPU guard 成功完成128测量输出 tokens、原 shutdown和会话排空，
但选中步16.893473 ms高于冻结16.238752 ms上界0.654721 ms，
原资格审核保持失败，shadow/on未执行。持续 GPU授权已生效，
后续无需逐轮人工授权；技术先决条件和原8小时预算仍必须满足。
当前累计已用24155.170197121333秒、剩余4644.829802878667秒，无 active GPU作业。

服务器 29_PACK_COMPLETED_NORMAL_OFF_EVIDENCE 初次封包因原 runtime建立的模型 alias
symlink拒绝，未写 archive或manifest。这个 alias指向现有固定
models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444。
pack_normal_delivery_v2.py仅在完整target匹配时跳过这一个模型alias；
unknown symlink继续拒绝，不跟随、归档或删除模型alias/weights。
29B_PACK_COMPLETED_NORMAL_OFF_EVIDENCE 真正完成封包，0.336385876秒CPU；
没有任何 GPU重跑。初次失败源码和日志保留。

实际归档：
- NORMAL_NORMAL_DELIVERY_off01-final.tar.gz：898542 bytes /
  SHA256 05cde5d6bfbd9202114460cc07eee02aa1f66050b60da210ddcd4b797d72acdf。
- manifest：43273 bytes /
  SHA256 3ebfb68af3fa9ab6295b499d9a91eff796129efe5ed6acaec1630d0ad8d2ee90。
- 原始成员170个，其中169份文件和末尾manifest；
  文件payload7168627 bytes，所有server文件封包前后完整byteSHA一致，
  原GPU账本前后SHA 7015729867cac4d285bf6277433fd395e8d721bcd7dc71d92e92bf3cd416637f
  完全相同。
- 本机 verify_normal_delivery.py已真正运行，
  PASS_SERVER12_NORMAL_DELIVERY_LOCAL_BYTE_VERIFICATION，
  169/169文件校验通过，真实 prior GPU_EVIDENCE.tar.gz字节重新校验通过。
- 本机完整original serverpath→shortlocalpath映射：
  normal_off01_verified/LOCAL_NORMAL_DELIVERY_BYTE_VERIFICATION.json。
  原始路径保留在tar及manifest；实际本机文件位于recorded_files/pathhash12_basename64，
  没有修改Windows全局路径设置、使用extractall或覆盖已有文件。

源码／实际命令／107 normal CPU检查及20 prototype服务器检查／metadata与真实计时／
用户持续指示／normal原始guard、raw、process.log／失败qualification／对照v1+v2
全部封包。前一single6完整原始证据保持独立GPU_EVIDENCE.tar.gz SHA绑定依赖，
本包不重复复制已保存校准原始数据、模型、SDK二进制或private KV/cache。
所有原文件均保留服务器。

CPU复用原型真正公开loader调用一次，随后两个get为同对象/类型，
完整源码/authority/SDK/ledger/session重查通过；它尚未集成 active GPU启动，
也没有GPU提速证明。不能用它改写成本失败或开下一mode。

FINAL_NORMAL_OFF_GPU_REPORT.md给出实际改动、每项CPU/GPU命令与结果、
数学限界、证据位置和下一允许阶段：有限成本稳定性与迁移诊断，
以及CPU active-entry校验去重设计和新来源冻结。
旧失败样本继续保持未通过，不事后调高上界、改变步预算或挑选seed重跑至通过。
NEXT GPU权限不阻塞；当前被阻塞的是失败的阶段资格。

