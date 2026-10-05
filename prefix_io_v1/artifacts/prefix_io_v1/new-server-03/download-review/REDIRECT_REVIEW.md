# 精确 CDN 路线跟随：CPU 复核补充

本文件取代 README.md 中“当前所有 redirect 均失败”的阶段状态。原39项测试、前后日志与 0001 补丁保持原样。

## 实际依据与范围

主线程首次真实权重 GET 被旧的失败关闭机制拒绝。该次没有读取body，但按设计消耗完整文件预留；其计费不能在本修复后退还。随后 safety_review 对4个固定官方URL分别执行受控GET获取Location，再单独HEAD该Location；8次真实请求只属于该审查任务，完整证据在 ../redirect-review/，不由本次CPU审查重复执行。

保存证据确认唯一 hostname 为 cdn-lfs-cn-1.modelscope.cn，4个HEAD均200、无二跳、长度等于manifest；TLS验证保持开启。该证据允许如下最小增量：

- 初始 manifest URL 仍只能是固定 ModelScope/Hugging Face 官方主站，不允许直接从CDN开始。
- 新增唯一精确 CDN hostname；仅 ModelScope 官方主站→该CDN、该CDN→自身的路线可跟随。没有通配符，不允许 Hugging Face→该CDN或其他主站互跳。
- 301/302/303/307/308 使用同一处理，最多3跳。
- 每跳先验证目标，再关闭旧响应；从不调用标准 urllib http_error_302 的 fp.read()，不消费redirect body。
- 每跳要求 CDN内容地址路径对应 manifest SHA256，query 的 namespace/repository/revision/filename/tag 全部匹配。绑定随请求传播。临时 auth_key 来自本次官方GET，不复用或固定历史签名。
- 仍保留最终payload大小/hash验证、失败保守收费与原始partial行为。

## CPU 结果

原39项加26项新增，共 **65 passed / 0 failed / 0 skipped**，0.49秒。新增覆盖全部5种redirect状态、3跳成功/第4跳拒绝、关闭旧响应先于下一请求、精确host与route限制、缺Location、相对同CDN路线、禁止manifest直接指CDN、内容地址或query任一关键字段不符早拒，以及经过假CDN的完整文件只按payload收费一次。

另将 safety_review 保存的4个真实Location传入实际处理器，parent.open 替换为只记录请求的stub；4条均通过精确route与对象绑定。此校验离线执行，没有新请求、模型下载或GPU。见 recorded-route-validation.json。

早期 redirect-before 结果属于首次新增用例尚未实现时，保留用于对照；最终 after 覆盖全部65项。0002-evidenced-cdn-redirect.patch 是在0001之后应用的增量，包含处理器及新增测试；正反向 git apply --check 和 whitespace=error 通过。脚本与补丁哈希见 redirect-verification.json。

## 保护与限制

本次没有写真实共享账本、权限、GPU runner 或正式报告/锁。主线程可在重读实际剩余额度、确认无未结算预留后进行一次显式下载重试。没有隐式网络重试；任何新的host、对象不符或超限跳转继续失败并按原计费规则结算。测试不代表实际下载已成功，不改变io_uring阶段门禁。
