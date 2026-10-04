# Dryad确认性主要结果反向分支补全

版本：v1.0-result-blind-amendment  
日期：2026-09-29  
状态：16/35参与者级派生完成；总体效应未访问

## 发现的问题

原结果回填方案预设了三种主要ITPC结果：theta+显著更高、操作性等效、不显著且不等效。但双侧检验还允许第四种情况：主要ITPC显著低于非节律同频主动对照。若不在结果前单独定义，该结果可能被错误归入“不确定”，或者在结果后临时解释。

## 新增的第四分支

当主要配对差`theta+ − non-rhythmic < 0`且双侧`p < 0.05`时，机器分支为：

`theta_plus_lower_opposite_direction`

允许的核心表述为：主要ITPC差异方向与theta+更强target engagement的假设相反，theta+低于非节律主动对照。必须同时报告95%区间、配对效应量和ROI通道敏感性方向。

该分支：

- 反驳“theta+在主要ITPC上更高”的方向性假设；
- 不等同于没有神经响应，因为比较的是两个条件的相对差异；
- 不允许改用显著的次要SNR指标救援主要终点；
- 不允许升级为抑制机制、认知损害或临床无效；
- 若ROI敏感性方向不一致，应明确写为稳健性受限。

## 机器判定

`scripts/classify_dryad_confirmatory_primary_branch.py`在总体分析及推断审计完成后读取唯一主要ITPC行和ROI敏感性行，将结果分为：

1. `theta_plus_higher`；
2. `theta_plus_lower_opposite_direction`；
3. `operationally_equivalent`；
4. `inconclusive_not_equivalent`。

只有第一分支且ROI敏感性方向一致时，才设置`theta_plus_superiority_claim_unlocked=true`。该补全不改变双侧检验、显著性阈值、等效界值、主要估计量或任何参与者级派生。
