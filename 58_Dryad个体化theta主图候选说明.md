# Dryad个体化theta主图候选说明

版本：1.0  
日期：2026-09-21  
文件：`outputs/manuscript_figures_v02/Figure7_personalized_theta_validation.png/.pdf`

## 图的中心信息

该图不是为了增加一个数据集面板，而是检验新的中心论证能否被一张图完整表达：

> Average behavioral benefit, condition-level neural coupling, and individual-benefit prediction are distinct validation claims.

## 四个面板

### A：个体化theta相对主动2 Hz的平均行为效应

显示`fθ`、`fθ+`、二者平均RT及逆效率的配对效应和95% CI。RT与逆效率均支持个体化条件，但图下注明准确率低0.43个百分点，防止“完全无速度—准确率权衡”的过度表述。

### B：entrainment组间/组内分解

跨人平均entrainment与更快post-RT相关；同一人的条件内偏差接近0且区间跨0，并报告20,000次置换p=0.402。该面板是全文最直接的机制边界。

### C：基线表现结论依赖参照定义

基线较慢与“相对自身baseline的改善”相关，也与相对非节律控制的优势相关，但不与个体化相对主动2 Hz的额外优势相关。这将回归均值/数学耦合风险和真正的主动对照异质性分开。

### D：刺激前个体获益预测

嵌套留一被试预测几乎收缩到训练集平均值，MAE没有改善。对角线展示理想校准，横线展示均值预测。该面板不用于宣称个体差异不存在，而用于否定当前变量已构成可部署选择规则。

## 当前图级判定

视觉质检通过：

- 全部区间可见，0线清晰；
- 不合并不同数据集的原始幅值；
- 组间与组内估计直接分开；
- 平均效应和预测失败在同图中不相互抵消；
- 预测面板显示全部44人而非只报单一误差摘要。

在原始BDF分析完成前，该图标记为**候选主图**。若trial/block级重算复现平均效应和组间/组内分离，它应进入正文并替代现有较弱的关联/预测整合面板之一，而不是把正文扩展为七张主图。若事件或信号闸门失败，则降为补充图并明确派生表身份。

## 可复现资产

- 生成脚本：`scripts/build_dryad_theta_figure.py`；
- 面板数据：`Figure7A_*`至`Figure7D_*`；
- 输入哈希：`Figure7_manifest.json`；
- 自动测试：`tests/test_build_dryad_theta_figure.py`。
