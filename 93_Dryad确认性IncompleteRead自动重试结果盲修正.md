# Dryad确认性IncompleteRead自动重试结果盲修正

日期：2026-09-29  
修正ID：`dryad_confirmatory_incomplete_read_retry_v1`

S18和S20的远程范围读取分别出现`http.client.IncompleteRead`。断点工具没有写入不完整分块，已有压缩字节保持可继续，但原异常集合未把该异常交给内部12次重试循环，因而需要人工重新启动参与者运行。

新增v2提取入口只将`IncompleteRead`加入与超时、连接错误和URL错误相同的自动重试集合；范围起止、分块大小、持久化断点、CRC、未压缩大小、BDF SHA-256、技术输入哈希和全部科学派生均不变。修正制定时S20尚未完成，组水平分析继续关闭。
