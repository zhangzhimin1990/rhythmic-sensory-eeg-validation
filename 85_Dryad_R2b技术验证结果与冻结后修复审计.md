# Dryad R2b技术验证结果与冻结后修复审计

版本：v1.0  
日期：2026-09-24  
决策范围：技术可分析性；不包含任何刺激条件效应、行为效应或认知结论。

## 1. 冻结与样本隔离

五名开发参与者固定为S2、S13、S23、S27和S37。其余39名参与者构成一次性技术验证集。验证前冻结包包含19项资产，清单SHA-256为`fcb8a934be29882795fe37a510e96cee448a4c6e9363f911a7c41a45be837fa6`。技术输出schema明确禁止条件效应、行为结局、目标频率响应和预测字段。

## 2. 技术验证结果

- 精确验证集：39人，无重复、无缺失、无开发集混入。
- 全部技术规则通过：35人。
- 未通过：4人（S1、S15、S25、S41）。
- 通过率：35/39 = 89.74%；Wilson 95% CI 76.42%–95.94%。
- 预设总体要求：至少32/39人通过。
- 决策：技术Go。

失败机制如下：

| 参与者 | 未通过的冻结技术规则 |
|---|---|
| S1 | 总体epoch可用率低于90% |
| S15 | 坏道数超过12（可用通道少于冻结要求） |
| S25 | 最小条件单元epoch可用率低于80% |
| S41 | 总体epoch可用率低于90%，且最小条件单元低于80% |

事件覆盖、四个条件单元存在性和分析窗安全间隔均未造成额外失败。失败不是按结果方向或行为表现定义。

## 3. 独立重算

正式汇总之外，另从39份单人技术JSON直接读取六个布尔闸门，并以逻辑合取重算总结果；得到同样的35人通过及同样的四名失败者和失败机制。工作区完整回归测试168/168通过。

主要机器可读证据：

- `outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.csv`
- `outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.json`
- `outputs/qc/dryad_raw_stream/R2b_validation_summary_v1/participant_technical_gate.csv`
- `outputs/qc/dryad_raw_stream/R2b_validation_summary_v1/failure_mechanism_counts.csv`
- `outputs/qc/dryad_raw_stream/R2b_validation_summary_v1/failed_participants.csv`
- `outputs/qc/dryad_raw_stream/R2b_validation_summary_v1/validation_summary.json`

## 4. 冻结后汇总程序修复

完成39名单人处理后，首次总体汇总暴露一个纯接口错误：单人技术JSON按设计不写入`subject`字段，而总体汇总器错误地把该字段列为JSON必需字段；同一汇总器随后本来就会从`S##_Stim.bdf`文件名和父目录双重推导并核对参与者编号。

修复仅把`subject`从“JSON必需字段”检查中排除，并增加回归测试；参与者编号仍必须通过严格文件名模式和目录一致性核验。修复前冻结版汇总脚本SHA-256为`4650e5fbc91db1f0d0cbb36176fba1868d1718c3e9f573d2ec1cf661ce1249e7`，修复后为`344e8c469eb3b88a571db8454e145091464a852cb51baeb1a72afbfec3847499`。对应测试由`2213d208c56c72d61354d7b18f2733bf4c8a7520e1f362c81cda1d313944dd47`变为`6616f2e04dd1249c65d5308504d8ede007dcb9e921a6b4324b06aacc0c57b0d6`。

修复前后生成的39人明细CSV SHA-256均为`50dcf49353cd7b1fe995eff04092bf4ec3c5bc770842afc0d3833954bc089263`。因此该修复不改变任何参与者技术数值、纳排规则、通过状态或35/39总体结论。原冻结清单不重写；本节作为冻结后偏离记录保留。

## 5. 科学解释边界

技术Go支持的唯一结论是：预先冻结、结果盲的自动流程可以在足够比例的独立参与者上生成可用于下一阶段的EEG测量。它不支持以下说法：

- theta+刺激产生更强神经响应；
- 神经响应具有机制特异性；
- 神经响应与准确率或反应时改善相关；
- 刺激改善认知，更不能外推至认知障碍患者治疗。

下一阶段必须在再次读取原始BDF前冻结结果承载指标和统计模型。主要分析只纳入35名技术通过的验证参与者；开发集和技术失败者不进入确认性效应估计。
