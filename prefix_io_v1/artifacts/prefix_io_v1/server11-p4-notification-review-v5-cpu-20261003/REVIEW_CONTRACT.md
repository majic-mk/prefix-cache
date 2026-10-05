# V5 CPU 通知候选：独立协议审查合同

状态：等待候选实现，本文不构成 PASS 或 GPU-ready。

依据 `04_CODEX_EXECUTION.md`：保留原 engine/executor、Queue、owner 和生命周期；off 保留 C4 路径；通知仅传递标量状态变化。不得将 host end 通知当作 CUDA 完成、资源释放或成本资格。

必须用源层和真实 Python Queue/线程反例验证：

1. registration 前已 end、registration 内交错 end、registration 后至 get 前 end、get 后 end，均不漏唤醒；回调不等待原 reactor。
2. 原 run/capture/step/generation 身份匹配；旧消息、重复消息及取消后消息不清除新 registration，不刷新 snapshot/open_start。
3. 等待时间最多到原 open_start + max_wait 与原 freshness 截止的最小值；严格保留原 100 ms 策略时限。到期回到原路径，未发生 end 也不能永久等待。
4. 同步状态检查之后，mandatory、STOP、原工作提交仍通过原 Queue 及时唤醒且保持 FIFO。STOP 不新增阻塞，资源不提前释放、不重复释放。
5. capture fail/detach 与 bridge fault 的线程归属和通知路径分别检查。不得用 owner 只写假设遮盖实际允许的跨线程 mutation。
6. scheduled-load 原接口只改共享标量，没有入队；新 publication 与同序冲突清 frame 都需停车失效闭环。不可假设原接口会 Queue.put。
7. active parent、inflight read/open、copy pending/ready、其它 ready、preload pending、mandatory waiter、未知 drain、容量变化等任何可推进原生工作必须拒绝停车。
8. collector/capture 锁和原 submit 锁不得反序；锁内不调用会等待 owner 或获取反向锁的通知操作。
9. 一步至多一次有意义注册和对应唤醒；不得对每次轮询/检查 enqueue 空控制消息，防止用通知队列形成新的空转。
10. off 不注册、不读通知状态、不新增 Queue 消息/计时/条件等待；_has_work、STOP cleanup、原 pump 顺序、原预留/释放及发布者不变。

范围限制：CPU mock 与真实 Python 线程测试只能证明协议的这些场景，不能证明 GPU 正确性、性能或成本迁移。任何失效路径未闭环则保持 CPU-only。

具体已发现的实现前风险已发送实现代理：scheduled-load 无原唤醒；capture/submit 锁反序；bridge.fail 与 capture.fail 是不同失效来源。
