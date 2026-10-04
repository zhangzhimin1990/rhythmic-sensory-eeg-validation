# Table 2：节律性感觉EEG的四条推断转换与结论边界

版本：0.3  
更新：2026-09-23  
机器可读权威源：`outputs/manuscript_tables_v02/Table2_inferential_transitions.csv`  
明细证据表：`outputs/manuscript_tables_v02/Table2_validation_claims.csv`  
上游主张源：`33_主张-估计量-验证层级主表.csv`

| Transition | Antecedent evidence | Stronger claim tested | Key result | Decision |
|---|---|---|---|---|
| T1 | Detectable or diagnostically sensitive response | State-general endogenous mechanism | Visual result depended jointly on state, ROI, window and denominator; an external temporal-prediction control showed that phase concentration was not mechanism-specific | Not established |
| T2 | High within-recording reliability | Metric- and level-matched cognitive validity | ds005048: reliability ρ=0.838, MMSE ρ=0.085; ds006780: ITPC reliability 0.972, d-prime reliability 0.931, adjusted β=0.058 | Not established in tested settings |
| T3 | Mean advantage over active 2 Hz stimulation | Within-participant neural coupling | RT difference −10.56 ms; within-person coefficient −0.90 ms/SD, 95% CI −3.08 to 1.28; permutation p=0.402 | Not supported for tested estimand |
| T4 | In-sample association or concurrent validity | Held-out incremental prediction | No clear prediction increment in ds007648 or Dryad; ds006780 d-prime prediction worsened and FSIQ prediction showed no clear gain | No consistent increment |

## Central synthesis

The main table is ordered by the claim transition rather than by dataset or whether a p value crosses 0.05. Each row preserves positive evidence at the antecedent level and asks whether the stronger claim passed its own estimand. The nine-row detailed validation table remains available as a supplementary lookup so that compression into four transitions does not erase dataset-specific results or uncertainty.

## Reporting rules

1. `Supported`, `limited`, and `not tested` are not collapsed into a single valid/invalid score.
2. Confidence intervals remain beside the affected claim; `p>0.05` is not translated into proof of absence.
3. Same-data computational reproduction is not called independent replication.
4. Cross-dataset synthesis concerns validation logic rather than a pooled biological effect.
5. Acute behavior, concurrent association, and treatment efficacy remain distinct endpoint categories.
