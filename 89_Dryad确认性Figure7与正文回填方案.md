# Dryad确认性Figure 7与正文回填方案

版本：v1.0-pre-results  
日期：2026-09-27  
状态：35人总体结果未生成；本文件只锁定呈现结构和分支语言，不预设结果方向。

## 1. 正文中的功能定位

确认性Dryad分析不应作为又一个“刺激有效”案例，而应承担全文第三、第四条推断转换的直接检验：

1. 平均主动对照差异能否升级为刺激锁定神经target engagement；
2. target engagement能否升级为同一被试、同一条件内的神经—行为联系；
3. 群体与同期关联能否升级为新参与者的个体获益预测。

正文保留Figure 1–6的多队列证据链，Figure 7从“作者发布表格的候选结果”升级为原始EEG确认性终点图。发布表格中的2 Hz比较、基线调节和旧预测结果移至补充材料或作为前置观察简述，不能与新的theta+—非节律同频主动对照混为一个估计量。

## 2. Figure 7主图结构

### Panel A：设计、冻结和样本流

- 44名公开参与者；5名永久开发集；39名一次性技术验证；4名按冻结规则技术失败；35名确认性样本。
- 标出主要对比：`f_theta_plus − non_rhythmic`。
- 标出共同分析频率：`1.33 × individual theta`，避免把不同读出频率误写成条件效应。
- 图中明确：技术验证阶段没有访问条件效应。

### Panel B：唯一主要神经估计量

- 横轴为每名参与者的ITPC差值，显示35个点、均值、95%区间和bootstrap区间。
- 同时显示`±dz 0.30`换算后的原始尺度等效界值，但视觉上区分95%优效区间与90%等效区间。
- 至少4个额中ROI好通道的敏感性作为较小的第二行估计，不替代35人主要估计。
- 不以星号数量作为主要视觉编码；直接标注差值、区间、双侧p值和等效判定。

### Panel C：行为效应与条件内耦合

- 上半部分显示准确率、正确RT和逆效率的theta+—非节律配对差及Holm校正结果。
- 下半部分显示正确率和正确RT模型在两个条件中的相位对齐简单斜率及95%区间。
- 速度方向统一为“负值代表更快”，准确率方向统一为“正值代表更好”；图注说明量纲不同，不能比较效应高度。
- 一致/不一致trial结果放入补充图，不在正文选择更有利的一层。
- 即使简单斜率阳性，也不允许写成中介、因果链或认知调控机制。

### Panel D：严格样本外预测增量

- 显示`baseline_plus_neural`相对训练折均值和基础特征模型的MAE改善及participant-bootstrap 95%区间。
- RMSE与Spearman增量作为次级符号或补充表，不用相关系数掩盖绝对误差没有改善。
- 只有两个MAE区间下限均大于0且两个RMSE点估计均改善，才标记prediction gate passed。
- 闸门未通过时不能写成“个体获益不可预测”，只能报告本样本没有一致的增量预测证据。

### 补充图和补充表

- Supplementary Figure S5：evoked、total和induced local log-SNR三指标及Holm校正。
- Supplementary Figure S6：奇偶trial、时间半分的内部一致性及bootstrap区间。
- Supplementary Table S6：35人技术与分析流、每人可用trial数和ROI通道数，不含可识别信息。
- Supplementary Table S7：全部主要、次要、敏感性、GEE和预测估计量及多重校正家族。

## 3. Results回填顺序

固定为以下六段，不按显著性重排：

1. 技术验证与35人确认性样本；
2. 主要ITPC对比及等效/不确定性判定；
3. 次要SNR成分与内部一致性；
4. 行为主动对照效应；
5. 条件内神经—行为简单斜率；
6. 样本外预测增量。

## 4. 英文结果分支模板

### 4.1 主要ITPC为正且显著

> In the frozen 35-participant confirmatory cohort, frontocentral ITPC at the common analysis frequency was higher after theta-plus than after non-rhythmic stimulation (mean paired difference [ESTIMATE], 95% CI [LOW] to [HIGH], paired d<sub>z</sub>=[DZ], two-sided p=[P]). The direction was unchanged in participants retaining at least four frozen-QC frontocentral channels. This result supports stronger stimulus-locked target engagement under temporal regularity; it does not by itself distinguish endogenous entrainment from repeated or temporally predictable evoked activity.

### 4.2 主要ITPC等效

> The primary ITPC contrast was not statistically different from zero, and its 90% interval fell entirely within the prespecified operational bounds of ±0.30 d<sub>z</sub> ([LOW] to [HIGH]). Thus, the data excluded a target-engagement difference of the prespecified magnitude for this contrast. The bound is measurement-oriented rather than a clinical minimum important difference and does not establish treatment inefficacy.

### 4.3 主要ITPC不显著且不等效

> The primary ITPC contrast was not statistically different from zero, but its 90% interval crossed at least one prespecified ±0.30 d<sub>z</sub> bound. The confirmatory result was therefore inconclusive rather than evidence of equivalence or absence of a neural response.

### 4.4 行为效应

> Theta-plus versus non-rhythmic stimulation changed [ENDPOINT] by [ESTIMATE] (95% CI [LOW] to [HIGH], Holm-adjusted p=[P]). Accuracy [DID/DID NOT] worsen, and the inverse-efficiency estimate was [FAVOURABLE/UNFAVOURABLE/INCONCLUSIVE]. We therefore describe [AN ACUTE TASK-PERFORMANCE ADVANTAGE / NO CLEAR ACTIVE-CONTROL BEHAVIOURAL ADVANTAGE], not cognitive rehabilitation or durable efficacy.

### 4.5 条件内耦合

> After centering leave-one-trial-out phase alignment within participant and condition, the simple slope for [OUTCOME] was [ESTIMATE] in theta-plus trials (95% CI [LOW] to [HIGH]) and [ESTIMATE] in non-rhythmic trials (95% CI [LOW] to [HIGH]). [THE/NEITHER] prespecified Holm-corrected interval supported concurrent within-participant coupling. These associations do not identify mediation or causal ordering.

### 4.6 预测

> Adding neural features changed held-out MAE by [DELTA] relative to the baseline-feature model and by [DELTA] relative to the training-fold mean. The corresponding bootstrap intervals [DID/DID NOT] exclude no improvement for both comparisons, and the RMSE guard [WAS/WAS NOT] satisfied. The prespecified prediction gate therefore [PASSED / DID NOT PASS]; failure is interpreted as no consistent incremental prediction in this sample, not proof that individual benefit is intrinsically unpredictable.

## 5. Discussion回填逻辑

### 若主要ITPC、耦合均阳性

可写为原始EEG支持target engagement并提供同期被试内联系，但仍明确缺少时间先后、操纵中介、长期结局和患者样本。创新重点是“从测量到机制主张的逐级检验”，不是泛化的刺激疗效。

### 若主要ITPC阳性、耦合阴性

这是最能支撑全文中心论点的组合之一：规则刺激可以提高刺激锁定响应，但该响应没有自动成为行为变化的被试内机制代理。必须避免把耦合阴性写成主要神经效应无意义。

### 若主要ITPC阴性或等效

将论文重心收敛到多队列测量学与转化边界：作者发布的平均行为优势不能保证冻结ROI和主动对照下存在可推广的神经差异。次要SNR、行为或预测结果均不能救援主要终点。

### 无论结果如何

不能推广到认知障碍患者、长期治疗、疾病修饰或临床处方。Dryad队列为健康老年人的急性任务实验；认知障碍分期数据在全文中承担外部表型和可推广性边界，而不是同一干预试验的治疗终点。

## 6. 防止结果驱动呈现

- 不根据p值改变Panel顺序、颜色突出或正文段落顺序。
- 不删除阴性主要终点或把最显著的SNR指标提升为主终点。
- 不把作者发布表格和本研究原始EEG派生结果混合成同一个“复现”估计。
- 不用单人轨迹或极端参与者作为机制故事；个体点只用于显示异质性和数据完整性。
- 主图、图注、Results、Discussion和摘要必须使用同一结果分支判定。
