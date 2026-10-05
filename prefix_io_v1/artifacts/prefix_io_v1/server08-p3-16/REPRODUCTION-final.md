# P3 最终复现索引

服务器工作区：/root/autodl-tmp/prefix-io-v1-handoff/project。凭据不入报告。

实际 38 次 GPU stage 命令、各自 wrapper SHA、返回码、session 和预算用时见 p3-reproduction-final.json，其中37次成功、1次在模型启动前因容量资格文件路径错误失败。每次复跑都必须经原 run_gpu_stage.py、唯一新label、当前source冻结、GPU空闲、8小时累计预算和原存储预留校验；不要裸运行 child argv。

本次 CPU 保留1613个去重passed、16 skipped；50 subtests不另加；12个冻结历史fixture只支持其历史协议。资格、XML、CUDA禁止初始化guard及补丁正反往返均由closeout manifest逐SHA绑定。纯元数据路径修正通过actual-capacity-qualification-binding-preflight.json验证，没有运行GPU或新增测试计数。

最终严格元数据验收命令：

```bash
.venv/bin/python -I -S experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316_v3.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --manifest artifacts/prefix_io_v1/server08-p3-16/p3-closeout-manifest-final.json --output artifacts/prefix_io_v1/server08-p3-16/p3-closeout-analysis-final.json
```

这是只读stdlib证据核对，不能代替实际CUDA/AIO资格。最终phase裁决以P3_FINAL_ROOT_REVIEW.md为准；模型权重、3048份共享源payload、私有cache和已安装二进制不打包，其身份及资格分别由注册源SHA、compiled byte identities及GPU收据绑定。P4-P7关闭。
