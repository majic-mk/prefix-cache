目前只需要提供实际已有请求文件的服务器路径。本目录的 RAW_TOKENIZATION_REQUEST_UNBOUND_TEMPLATE.json 将 dataset_ref 留为 null，CLI 会拒绝执行；它不是已经验证的输入或 GPU 配置。

原作者解析器支持 .jsonl 的逐行字符串/对象、.json 的列表或 data 列表、其他后缀的非空文本行。对象读取优先 prompt、text、input、content；ShareGPT conversations 只拼接 human/user 的 value。OpenAI messages/body 结构未自动支持。工具按作者的全量接受顺序处理，不截尾、不随机挑性能有利的请求。原 parser 可能跳过其不支持的记录；因此结果会保留作者接受计数，不能声称全部原始行都已执行。

分词工具只产生 raw token IDs。它不会给请求编造 prefix_family、分区、时间戳或控制 deadline；独立 raw 结果不能当作原正式 receipt。还需要原始会话、文档或公共前缀来源，且要证明原记录位置、作者接受序号和 prompt SHA 对应。新增一个侧车字段并不会自动获得旧协议的证明能力。

全部数据限 32 MiB、作者接受数不超过 96；正式三分区各不超过 32，必须全量保序且没有跨区家族或相同 prompt 泄漏。原 engine 当前为 max_model_len=1024、完整输出128tokens，正式绑定还会检查每请求上下文及全分区4096步骤上限。超过任何上限会拒绝整份输入；不能静默截断。

控制 deadline 必须来自真实独立需求，不能从此次成本测量、A_max 或效果结果倒推。开发诊断是否可留空服务 SLO，仍由既有开发合同决定；正式效果需要预先确定的 TTFT/ITL 服务目标和真实 native reserve。主机/boot_id/CLOCK_MONOTONIC 声明必须对应实际开发运行；若切换模式、重启或换机器改变 clock scope，需要在开发开始前重新声明，不能沿用旧 boot 的时间证据。

GPU 无需为这些 CPU 步骤保持开启。保留现有模型、SDK、原 GPU 原始证据和缓存文件；CPU 交付包不是整机备份。
