# Dryad公开代码可复现性与指标语义审计

版本：1.0  
日期：2026-09-21  
审计范围：Dryad DOI 10.5061/dryad.t76hdr8dm v5的六个MATLAB代码包与README

## 1. 判定摘要

公开代码足以重建条件、目标、正确性和主要预处理意图，但**不能原样端到端运行并独立生成`Dataset.xlsx`中的entrainment字段**。主要问题分为三类：

1. 人工交互步骤未提供逐被试决策记录；
2. entrainment指标的计算语义与普通“功率/同步强度”表述不完全一致；
3. 最终汇总脚本存在会阻止运行的变量和语法错误。

因此，派生表可以用于透明的二次统计分析，但机制性主张必须由原始BDF的独立、自动化管线复核。

## 2. 可恢复的事件结构

刺激条件码：

- 1：个体化`fθ`；
- 2：固定2 Hz；
- 3：`fθ+`；
- 4：非节律NR。

目标码：11/12/21/22/31/32/41/42，同时编码耳侧、词义、音高与congruency。

反应码：

- 51/52/61/62/71/72/81/82：正确；
- 50/60/70/80：错误；
- 55/65/75/85：无反应。

事件顺序应为刺激开始→目标→反应；response latency减target latency理论上可重建逐trial RT。公开包未提供从事件到`Dataset.xlsx`行为字段的汇总脚本，因此原始数据阶段必须独立核对试次数、RT排除规则和条件均值。

## 3. 预处理可复现性

明确的自动步骤：

- 删除8个外部/EOG通道，保留64 EEG；
- 0.1–45 Hz滤波，下采样至512 Hz；
- ASR `BurstCriterion=20`，不删除burst；
- 平均参考、球面样条插值；
- 1 Hz高通副本上运行20成分runica，再把ICA矩阵转回0.1 Hz数据；
- ICLabel后人工选择并删除成分。

不可直接复现的人工步骤：

- 导入后先目视删除大幅伪迹并另存`*_Pre.set`，未给出区段日志；
- GUI手工删除坏道，虽代码计划保存`Removed_Channs_Stim.mat`，该文件未随代码包提供；
- GUI手工删除ICA成分，虽代码计划保存`Removed_Comps_Stim.mat`，该文件未随代码包提供；
- 个体theta峰有3人由第二研究者人工复核，公开表仅给最终频率，没有盲化自动峰值规则。

独立分析必须用固定自动规则，并把与作者人工流程的差异登记为方法偏离，而不能声称完全复现原管线。

## 4. entrainment指标的实际计算语义

`Stim_ftheta.m`及同类脚本的关键顺序是：

1. 按刺激频率±1 Hz窄带滤波；
2. 只保留正确反应试次；
3. 前中央FC1/FC2/FC3/FC4/FCz和所有epoch先求平均；
4. 对这个单一平均波形做Hilbert变换；
5. 最终脚本在刺激前窗对解析信号虚部平方求均值，再减相应无刺激baseline。

因此，该字段强调跨试次、跨电极相位一致后仍保留的窄带诱发成分。它不是：

- 每trial解析振幅平方后再平均的total power；
- 不依赖相位的一般诱导功率；
- 直接的PLV或coherence；
- 已验证的内源振荡夹带强度。

正文宜称为`author-derived evoked narrow-band response change`，除非独立重算PLV、coherence、ITPC或trial-level power后得到一致结论。

## 5. 会阻止原样运行的代码问题

### 5.1 `Entrainment.m`

- 循环变量定义为`i`，写入和索引却使用未定义的`k`；
- `ti_1`和`ts_1`未定义；
- `eeg_times`、`eeg_times2`和`o`未定义；
- 多处`plot(...abs(...`缺失右括号；
- theta+绘图变量名在`stim_fthetaplus`与`stim_thetaplus`之间不一致。

该脚本不能作为生成Excel entrainment字段的可执行证据。

### 5.2 wavelet脚本

`Wavelet_fthetaplus_con.m`读取`S*_fthetaplus_Con_Correct.set.set`，而epoch脚本保存的是`.set`。若不手工更名或修正脚本，该条件无法加载。

多个wavelet脚本把受试者维初始化为2，再依赖MATLAB动态扩展到44；这通常不会改变数值，但表明脚本未经干净环境完整运行检查。

### 5.3 路径与中间文件

所有`file_path`为空，代码依赖人工整理的工作目录及多个未随包提供的`.set/.mat`中间文件。公开原始BDF和脚本具备重建可能性，但不是一键复现包。

## 6. 对当前科学结论的影响

1. 个体化条件相对2 Hz的行为差异直接来自完整无缺失的被试级汇总表，代码问题不自动否定该配对效应，但必须用原始事件重建验证。
2. 跨人entrainment—RT关联的指标语义更接近evoked narrow-band response，不能直接称“更强神经同步导致更快行为”。
3. 条件内关联阴性可能来自真实机制缺失，也可能受到指标可靠性、跨试次平均和条件级测量误差限制；原始BDF阶段必须报告分半/跨block信度。
4. 原文同时提供PLV、coherence和功率结果，但`Dataset.xlsx`只含一个entrainment字段；本文不能把三类指标当作同一受试者级数值使用。
5. 独立自动化重算本身具有方法学增量：比较evoked response、trial-level power与相位指标是否支持相同的组间/组内结论。

## 7. 原始BDF阶段的冻结规则

主指标建议：

- `ITPC/PLV`：逐trial、逐条件、前中央ROI，在实际刺激频率及邻频控制下计算；
- `evoked narrow-band amplitude`：复现作者先平均再Hilbert的思路，但名称准确；
- `total narrow-band power`：逐trial求功率后平均，与evoked指标分开；
- 每个指标均给奇偶trial和跨block信度。

主要模型：

- post-RT或trial log-RT为结局，条件固定效应；
- entrainment分解为跨人均值与人内条件/区块偏差；
- 被试随机截距，条件随机斜率在收敛允许时加入；
- 控制block、sub-block顺序、congruency、trial序号和刺激持续时间；
- 获益预测以被试为外层验证单位，不能把同一人的trial分到训练和测试两侧。

只有当至少两类神经指标方向一致、可靠性足够且人内效应在顺序/伪迹敏感性下稳定，才能升级为条件级机制证据。
