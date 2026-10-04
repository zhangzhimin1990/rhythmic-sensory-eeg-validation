# Dryad个体化theta听觉刺激数据库审计与Go/No-Go

版本：1.0  
日期：2026-09-21  
数据DOI：10.5061/dryad.t76hdr8dm  
对应论文：Gómez-Lombardi et al., 2026, *Proceedings of the Royal Society B*, DOI 10.1098/rspb.2025.1857

## 1. 当前判定

该队列从优先级B上调为**A+，进入主论文候选核心队列**。

理由不是数据规模，而是它首次为本项目同时提供：

1. 60–75岁老年人；
2. 被试内个体化频率、稍快个体化频率、固定2 Hz与非节律控制；
3. 刺激后的Simon抑制控制行为；
4. 刺激期间entrainment及刺激后N2、诱导theta指标；
5. 原始BDF、行为汇总表和可审计MATLAB代码。

它能够把当前论文从“频率跟随是否具有认知效度”的横断面测量学问题，推进到“个体化参数是否产生主动对照之上的即时行为差异，以及entrainment究竟是条件内机制指标还是跨人反应者表型”的更高层级问题。

## 2. 文件与完整性审计

Dryad v5总量约25.65 GB：

- `Raw_data_base.zip`：5.70 GB，静息/无刺激Simon基线BDF；
- `Raw_data_stim.zip`：19.95 GB，四种刺激条件BDF；
- `Dataset.xlsx`：44人×110字段，无缺失；
- `README.md`；
- 六个MATLAB代码包，覆盖预处理、epoch、wavelet、功率entrainment、Cxy/PLV、诱导功率及N2 ERP。

已下载并核验的小型文件：

- `Dataset.xlsx` SHA-256：`c0c4c21c7a45b775258f333215a56d585c4e5dff0d3934d0a17577cdaaa2910e`；
- `README.md` SHA-256：`0ebe8bfd90606d92713326ba1cf1f3bef5cbcbe69975d47dce3d78975c2491c9`；
- 六个代码包均已保存并计算SHA-256。

代码显示：64通道BioSemi，4096 Hz原始采样，下采样至512 Hz，0.1–45 Hz滤波，ASR、坏道插值、平均参考、20成分ICA；前中央ROI为FC1/FC2/FC3/FC4/FCz。原始刺激数据是单一约20 GB压缩包，不能在网页上按被试选择，因此在全量下载前必须先用现有派生表关闭科学价值闸门。

## 3. 原论文已经回答的问题

原文报告：

- 个体化`fθ`和`fθ+`相较2 Hz及非节律条件有更快RT；
- 44人中刺激主效应、entrainment主效应和基线表现主效应显著；
- 较差基线表现和较高entrainment者获益更大；
- 个体化刺激后诱导theta和N2发生改变；
- DCM提示节律刺激调制听觉—额叶网络。

因此，“个体化theta刺激能否改善Simon RT”本身不是新问题，简单复现原文不足以构成新论文。

## 4. 原分析留下的关键推断缺口

### 4.1 中位数二分掩盖连续信息

原LMM把基线RT分成Good/Poor，把三种节律条件的平均entrainment分成High/Low。二分会损失信息，并把条件内变化与稳定跨人差异混合。

### 4.2 change score与基线表现存在数学耦合

原结局为`(baseline - post)/baseline × 100`，同时又用baseline分组预测该结局。“基线较慢者改善更多”可能部分来自回归均值和结局构造，而不一定代表可迁移的治疗反应异质性。

### 4.3 entrainment主效应不等于条件内机制耦合

若High/Low来自每人的三条件平均entrainment，显著主效应只能说明高entrainment的人总体更快，不能证明“同一个人在entrainment更强的那个条件下也表现得更好”。后者才更接近条件级机制证据。

### 4.4 统计关联不等于个体化选择规则

原文没有证明刺激前的年龄、基线RT和个体theta频率能够在新被试上预测“个体化频率相对固定2 Hz能多获益多少”。若不能预测，个体化刺激的平均优势仍成立，但尚不能形成可部署的个体选择标志物。

## 5. 已完成的独立派生表闸门

分析源：`Dataset.xlsx`，44名被试、3个节律条件、132个被试—条件观测。所有新增分析均为查看原文和派生表后的探索性/验证性再分析，不追溯性称为预注册。

### 5.1 个体化条件相对主动2 Hz对照的平均效应

| 对比 | 估计 | 95% CI | p | 配对dz |
|---|---:|---:|---:|---:|
| `fθ - 2 Hz` RT | −9.92 ms | −15.66, −4.18 | 0.00114 | −0.526 |
| `fθ+ - 2 Hz` RT | −11.20 ms | −17.50, −4.90 | 0.00086 | −0.540 |
| 个体化均值−2 Hz RT | −10.56 ms | −16.25, −4.86 | 0.00054 | −0.563 |
| `fθ+ - fθ` RT | −1.28 ms | −5.20, 2.64 | 0.515 | −0.099 |

个体化频率相对2 Hz的平均RT优势可复现；`fθ+`没有显示超越`fθ`的明确增益。

### 5.2 速度—准确率边界

个体化条件相对2 Hz准确率低0.00428，即约0.43个百分点，95% CI −0.85至−0.003个百分点，p=0.0483。校正准确率的逆效率指标仍改善8.44 ms，95% CI −14.21至−2.68，p=0.00510。

因此可写“主要RT优势在逆效率指标中仍存在”，不可写“完全不存在速度—准确率权衡”。

### 5.3 entrainment的组间与组内分解

以post-RT为结局，调整刺激条件、基线RT、年龄和性别：

- 跨人平均entrainment每增加1 SD，RT快12.25 ms，95% CI −20.16至−4.33，p=0.00242；
- 同一人的条件内entrainment每增加1 SD，RT仅快0.90 ms，95% CI −3.08至1.28，p=0.417；
- 加入被试固定效应后，条件内估计仍为−0.90 ms，95% CI −3.54至1.73，p=0.502；
- 20,000次受试者内置换检验p=0.402；
- 逆效率结局中，组间关联仍存在，条件内关联仍不明确。

当前证据更支持“entrainment可能标记稳定的跨人反应倾向”，不支持“条件级entrainment增量是个体化刺激RT获益的直接机制代理”。

### 5.4 基线差者受益更多的边界

- 基线RT与“相对自身baseline的个体化改善百分比”相关：r=0.349，p=0.020；
- 基线RT与“个体化频率相对主动2 Hz的优势”无明确相关：r=−0.133，p=0.389；
- 基线RT与“个体化频率相对非节律控制的优势”相关：r=−0.475，p=0.00112。

这表明基线表现的异质性主要关系到相对无刺激/非节律参照的改善，而没有证明较慢者从“个体化而非固定节律”中获得更多额外收益。

### 5.5 个体化获益的样本外预测

结局定义为个体化两条件平均RT减2 Hz RT。以刺激前可得的个体theta频率、基线RT、年龄和性别建立嵌套留一被试岭回归：

- 训练集均值基线MAE：14.56 ms；
- 个体化模型MAE：14.64 ms；
- MAE差：+0.08 ms，bootstrap 95% CI −0.32至0.49；
- RMSE由18.95增至19.12 ms；
- 预测值与观察值反向相关r=−0.352，提示小样本选择规则不稳定，而不是可解释的预测成功。

刺激前变量没有提供可用的个体获益预测增量。

## 6. 科学价值与论文创新点

最有价值的新论证不是重复“个体化刺激有效”，而是：

> Personalized rhythmic stimulation shows an average active-control advantage, but neural entrainment separates into a between-person responder phenotype and a weak within-person condition-level correlate; average efficacy, mechanistic target engagement, and individual benefit prediction are distinct validation claims.

这一结果与现有多队列40 Hz证据形成互补：

- 40 Hz队列显示可检测性和高信度不保证行为/临床效度；
- 老年theta队列显示平均行为效应可以存在，但entrainment仍未自动升级为条件级机制中介或可部署预测标志物；
- 两者共同支持“target engagement验证阶梯”，避免论文被误读为单纯阴性结果集合。

## 7. Go/No-Go与下一阶段

### 当前Go

- 派生表完整，无缺失；
- 主动对照之上的RT优势可复现；
- 组间/组内分解产生清晰且科学上重要的边界结论；
- 预测失败本身与主论文验证阶梯一致；
- 速度—准确率细节提供必要的反过度解释证据。

### 原始数据下载后的硬闸门

1. 事件码能否重建四条件、三个block、sub-block顺序、congruency、正确性和逐trial RT；
2. 每人/条件有效trial数、坏道、ICA/ASR负担和刺激伪迹是否可审计；
3. 条件级entrainment能否用独立指标复现，且分半/跨block可靠；
4. 组内entrainment—行为关系在trial/block层级和替代指标下是否仍弱；
5. 个体化优势是否受顺序、疲劳、练习及刺激持续时间混杂；
6. 原文视觉选峰的个体theta频率能否以自动、盲化规则复现。

任一事件或行为映射失败，则该队列只作为派生表敏感性分析，不能承担主论文机制结论。若信号可复现但条件内机制仍为阴性，则论文主线保留并强化验证边界；若条件内机制转为稳定阳性，则升级为“平均效应—条件级靶点结合—预测价值”三层论文。

## 8. 可复现资产

- 分析脚本：`scripts/analyze_dryad_theta_validation_gate.py`；
- 自动测试：`tests/test_analyze_dryad_theta_validation_gate.py`；
- 输出目录：`outputs/models/dryad_theta_validation_gate/`；
- 主要输出：条件长表、聚类稳健模型、配对效应、20,000次置换检验、moderator检查、嵌套留一预测及分析manifest。
- 代码语义审计：`57_Dryad公开代码可复现性与指标语义审计.md`。公开entrainment汇总脚本存在未定义变量和语法错误，且指标先跨trial/电极平均再Hilbert；原始BDF阶段必须独立自动化重算，不能把Excel字段直接称作一般功率或机制性entrainment。
- 低存储原始数据方案：`71_Dryad原始ZIP流式处理方案与访问闸门.md`与`scripts/stream_remote_zip.py`。该方案已通过离线Range安全测试，但尚未完成真实ZIP中央目录和首名受试者BDF验证。

## 9. 结论边界

允许：

- 个体化theta条件在该44人老年样本中具有主动2 Hz对照之上的平均RT优势；
- 跨人平均entrainment与反应速度相关，但条件内entrainment—RT耦合不明确；
- 当前刺激前变量不能稳定预测谁获得更大的个体化优势；
- 平均刺激效应、机制性target engagement和个体获益预测是不同验证层级。

禁止：

- 声称已证明认知障碍治疗效果；
- 把健康老年Simon任务推广为MCI/AD临床获益；
- 把组间entrainment关联写成中介因果；
- 用汇总表结果替代trial级顺序、伪迹和可靠性审计；
- 把预测阴性写成个体差异不存在。
