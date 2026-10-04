# Dryad确认性统一主张释放规则

版本：v1.0-pre-results  
日期：2026-09-29  
状态：16/35参与者级派生完成；总体效应未访问

## 目的

主要ITPC、行为、条件内耦合和样本外预测各自有独立判定，但论文最终主张必须由它们的组合决定。不能因为某一个次要终点阳性，就跨过主要终点或行为—耦合链条提升摘要和标题。

`scripts/assemble_dryad_confirmatory_claim_release.py`把四套结果盲分支合并成唯一机器可读的主张上限，不形成综合评分，也不把不同量纲合并为一个效应量。

## 主张上限

1. **target engagement + acute behaviour + condition-specific concurrent coupling**：主要ITPC优效、急性任务表现优势和同一预设结局的条件特异耦合均解锁。仍不能称为因果机制或中介。
2. **target engagement + acute behaviour，缺少条件特异耦合**：支持神经与行为的平均主动对照差异，但证据链未延伸到条件特异同期耦合。
3. **只有target engagement**：支持刺激锁定神经差异，不支持行为—机制链。
4. **行为优势但缺少主要target engagement**：必须把神经主要终点与行为结果并列报告，不能以行为阳性救援神经假设。
5. **测量与转化边界**：主要神经—行为连续证据链未成立，论文转向严格的测量学和推断边界。

预测增量只作为独立修饰语。即使双比较器闸门通过，也不能提高因果主张、长期疗效、患者推广或临床处方上限。

## 永久锁定的禁止升级

无论四项结果如何，以下字段始终为`false`：

- causal mechanism claim；
- chronic cognitive efficacy claim；
- clinical treatment-selection claim；
- patient generalisation claim。

原因是Dryad队列为健康老年人的急性任务实验，没有时间先后明确的中介操纵、长期随访、认知障碍患者刺激样本或独立临床验证。
