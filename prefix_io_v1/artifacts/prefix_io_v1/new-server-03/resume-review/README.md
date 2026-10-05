# 显式单文件 Range 续传：CPU 预备与审查

本子任务没有真实网络请求、GPU 操作、真实模型文件或共享账本写入。新增 experiments/prefix_io_v1/scripts/resume_pinned_model.py 和 tests/prefix_io_v1_download/test_resume_boundaries.py；原 download_pinned_model.py 未改，SHA256 保持 c39562e4b2a6450f355b52210dc06ebd5ae499e6693349509ce6532718e08ded。

## 真实超时前提与边界

主线程实际 qwen-ms-02 被既有 1800 秒 timeout 终止，exit=124。旧脚本 main 已安装 SIGTERM/SIGINT/SIGHUP handler，抛 InterruptedError；流式请求内会被 except BaseException 捕获并按完整当前文件预留结算。信号可能打在应用层 read 返回与写文件之间，计数只代表应用层已返回读取字节，不承诺网络 wire 或 TLS 缓冲字节精确值。SIGKILL 或结算持久化中断仍可能遗留 active reservation，续传一律拒绝，绝不自动清空或退款。

实际失败快照由主线程保存于 ../download-timeout-settlement.json：原第4分片大小 3,556,377,672 B，partial 3,321,888,768 B，剩余 234,488,904 B；原失败 event id ec8173ee56a7473aab9395168b295933，原失败完整收费 3,556,377,673 B 已保留，active=null，旧进程已确认消失。指定 partial SHA256 为 533b7c828ecc2fa7641c61c5063b4efb3037e0153d81d3a90204ad52d3976214。以上真实输入由主线程提供，本子任务没有读取/复制该真实 partial。

## 最小实现

复用现有下载器的 validate_manifest、real_project_path、权限解析、VerifiedRedirect/make_opener（默认 TLS 验证）、verify_file、atomic_json 及同一个账本 flock；没有第二个主机白名单或另一套重定向机制。当前入口只允许固定 ModelScope 官方 manifest 的 SHA256 权重，不允许任意 URL、目录或 offset。

- 必须指定 failed-label、精确 file、原 partial、offset、partial SHA256 和全新 label。原 result 必须 FAILED，目标/revision/source/manifest SHA 一致；其中唯一目标失败事件须逐字段等于共享账本中已结算的同 id 事件，且保守收费与原文件大小+1 一致。
- offset 必须等于原 partial 当前大小并在文件内部、不得超过原失败应用层读取计数；原文件和每级路径拒绝 symlink。新结果、目标和 newlabel.partial 不覆盖旧数据。
- 持同锁，拒绝任何 active reservation，先检查累计 20 GiB 剩余额度。保留原 partial，只将其有界复制到全新 partial；复制时校验明确的 partial SHA 和源文件身份/大小/时间稳定性。磁盘先要求足以新增一个完整文件加 2 GiB，再请求前检查剩余写入空间。
- 网络打开前持久化 remaining+1，即本例 234,488,905 B 预留。只发一次 Range: bytes=3321888768-；原 helper 自动取得本次官方 CDN 签名。五种已支持跳转在 CPU 测试中均保留 Range、Accept-Encoding 与 _prefix_download_expected 元数据，仍最多 3 跳、精确对象绑定、不读 redirect body。
- 必须 HTTP 206、Content-Range 精确等于 bytes 3321888768-3556377671/3556377672、Content-Length 精确等于 234488904、identity encoding。200、缺失或不一致头均在任何 body read 前拒绝并关闭响应；真实 Range 支持尚未验证。
- 只读 remaining+1 上限；不足/超额/异常/全文件 SHA 错误都保留新 partial，并按整个 remaining+1 保守收费。成功才校验整个重组文件大小/SHA并发布，按本次实际应用层返回 payload 收费；原失败 charge 与原 partial 不删除。
- 返回 VERIFIED_RESUMED_FILE_ONLY，只证明该文件。它不会接着下载小文件，也不会自动重试或声称整个模型可用。后续主线程须用原全量下载器的新 label 复用校验全部文件并下载余下 tokenizer.json/vocab.json。

## CPU 结果及变更验证

实际执行命令（服务器项目根目录）：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1_download --junitxml=artifacts/prefix_io_v1/new-server-03/resume-review/first.xml
~~~

109 passed / 0 failed / 0 skipped，0.76 秒。原有 65 项 + 新增 44 项；不要与原 89 项重复相加。后续模型来源/KV metadata/serialization 最终 suite 已扩为 49 项（含旧 24 项），交付脱敏另 15 项，因此当前本轮总计 109+49+15=173 unique；历史 39/65/24 项均包含在其对应最终 suite 内，不能重复相加。测试 socket.connect 被禁用，所有流、文件、权限与账本均在临时测试项目。

新测试覆盖：成功整文件SHA+原失败/partial保留、预留fsync先于请求、共享锁、任意active拒绝、额度含探测字节、错误failed事件/manifest/URL/计费绑定、offset/partial路径/软链接/hash、禁止覆盖、两次磁盘检查、200或错误206 headers零body读取、短/超长/错误hash/InterruptedError全预留收费、结算失败保留active，以及5种redirect继承Range与元数据。

AST/源编译检查和新增补丁正反向 git apply --check、whitespace=error 均通过。证据 first.txt/xml、verification.json、0001-explicit-resume.patch。脚本 SHA256：86b1f6ad71e0f5fbdea689ea2efe7ae092407ee4b3b1a9bbb096b45ffe6a0ddc；补丁 SHA256：37f3d0016756ef19f42196aa69a6ca47fd5c2a71f976e33ff8d2f3bd623071c2。

## 仅交主线程执行的精确命令

此命令未由本子任务执行。主线程在核对同一冻结 partial、原失败已结算、无活动作业、余额有效后，选择已要求的新 label qwen-ms-resume-03。下面 900 秒为有界外层时限，未修改原 1800 秒任务；按新 label 保存实际 launch/log/exit/result。

~~~bash
timeout --signal=TERM --kill-after=15s 900s .venv/bin/python experiments/prefix_io_v1/scripts/resume_pinned_model.py --manifest artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --label qwen-ms-resume-03 --failed-label qwen-ms-02 --file model-00004-of-00004.safetensors --partial models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444/model-00004-of-00004.safetensors.qwen-ms-02.partial --offset 3321888768 --partial-sha256 533b7c828ecc2fa7641c61c5063b4efb3037e0153d81d3a90204ad52d3976214
~~~

真实执行结果：PENDING。若服务端返回200或不符合严格范围语义，失败关闭并保守收费，不默认重试、不放宽范围检查。原完整引擎未改、GPU未运行、P1 io_uring EPERM 门禁仍在。
