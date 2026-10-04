#!/usr/bin/env python3
"""Render an auditable manuscript insert from finalized Dryad outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "outputs/models/dryad_confirmatory_group_v1"
AUDIT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
CLAIMS = ROOT / "outputs/status/dryad_confirmatory_claim_release_v1"
OUTPUT = ROOT / "102_Dryad确认性结果与正文回填_v1.md"


def number(value: float, digits: int = 3) -> str:
    value = float(value)
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def effect_sentence(row: pd.Series, label: str, adjusted: str | None = None) -> str:
    p_column = adjusted or "p_value_t_two_sided"
    return (
        f"- {label}: mean theta-plus minus non-rhythmic difference "
        f"{number(row['mean_difference_theta_plus_minus_non_rhythmic'])}; "
        f"95% CI {number(row['ci95_low'])} to {number(row['ci95_high'])}; "
        f"paired d_z={number(row['paired_dz'])}; "
        f"{p_column}={number(row[p_column])}."
    )


def main() -> None:
    required = [
        GROUP / "paired_neural_effects.csv",
        GROUP / "paired_behaviour_effects.csv",
        GROUP / "internal_consistency.csv",
        GROUP / "within_subject_coupling.csv",
        GROUP / "prediction_performance.csv",
        AUDIT / "internal_consistency_bootstrap.csv",
        AUDIT / "prediction_increment_bootstrap.csv",
        AUDIT / "primary_roi_channel_sensitivity.csv",
        AUDIT / "within_condition_coupling_simple_slopes.csv",
        CLAIMS / "claim_release_decision.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"finalized Dryad outputs missing: {missing}")

    neural = pd.read_csv(GROUP / "paired_neural_effects.csv")
    behaviour = pd.read_csv(GROUP / "paired_behaviour_effects.csv")
    reliability = pd.read_csv(AUDIT / "internal_consistency_bootstrap.csv")
    coupling = pd.read_csv(AUDIT / "within_condition_coupling_simple_slopes.csv")
    prediction = pd.read_csv(AUDIT / "prediction_increment_bootstrap.csv")
    sensitivity = pd.read_csv(AUDIT / "primary_roi_channel_sensitivity.csv").iloc[0]
    claims = json.loads((CLAIMS / "claim_release_decision.json").read_text(encoding="utf-8"))
    primary = neural.loc[
        neural["metric"].eq("mean_itpc_in_stimulation_window")
    ].iloc[0]

    lines = [
        "# Dryad确认性结果与正文回填",
        "",
        "状态：35/35冻结确认性队列完成后由机器输出生成；数值来自锁定总体分析和推断审计。",
        "",
        "## 统一主张判定",
        "",
        f"- Claim ceiling: `{claims['claim_ceiling']}`",
        f"- Abstract core sentence: {claims['abstract_core_sentence']}",
        f"- Supporting modifiers: {'; '.join(claims['supporting_modifiers'])}",
        "- Causal mechanism, chronic efficacy, clinical treatment selection, and patient generalisation remain locked.",
        "",
        "## 主要神经终点",
        "",
        effect_sentence(primary, "Frontocentral ITPC"),
        (
            f"- 90% CI {number(primary['ci90_low'])} to {number(primary['ci90_high'])}; "
            f"operational equivalence within ±0.30 d_z: "
            f"{bool(primary['equivalent_within_plus_minus_dz_0_3'])}."
        ),
        (
            f"- ≥4 good ROI-channel sensitivity: estimate "
            f"{number(sensitivity['mean_difference_theta_plus_minus_non_rhythmic'])}; "
            f"95% CI {number(sensitivity['ci95_low'])} to {number(sensitivity['ci95_high'])}."
        ),
        "",
        "## 次要神经终点",
        "",
    ]
    for _, row in neural.loc[~neural["metric"].eq("mean_itpc_in_stimulation_window")].iterrows():
        lines.append(effect_sentence(row, str(row["metric"]), "p_value_holm_key_secondary"))

    lines.extend(["", "## 内部一致性（非跨日重测信度）", ""])
    for _, row in reliability.iterrows():
        lines.append(
            f"- {row['condition']} / {row['metric']} / {row['split']}: "
            f"Spearman–Brown={number(row['spearman_brown'])}; "
            f"95% bootstrap CI {number(row['spearman_brown_bootstrap_ci95_low'])} to "
            f"{number(row['spearman_brown_bootstrap_ci95_high'])}."
        )

    lines.extend(["", "## 行为主动对照效应", ""])
    for _, row in behaviour.iterrows():
        lines.append(effect_sentence(row, str(row["metric"]), "p_value_holm_behaviour"))

    lines.extend(["", "## 条件内神经—行为耦合", ""])
    for _, row in coupling.iterrows():
        lines.append(
            f"- {row['model']} / {row['condition']}: simple slope "
            f"{number(row['simple_slope_phase_alignment_within'])}; "
            f"95% CI {number(row['ci95_low'])} to {number(row['ci95_high'])}; "
            f"Holm p={number(row['p_value_holm_simple_slope_family'])}."
        )

    lines.extend(["", "## 严格样本外预测增量", ""])
    for _, row in prediction.iterrows():
        lines.append(
            f"- Versus {row['comparator']}: MAE improvement "
            f"{number(row['mae_improvement_comparator_minus_neural'])}; "
            f"95% bootstrap CI {number(row['mae_improvement_bootstrap_ci95_low'])} to "
            f"{number(row['mae_improvement_bootstrap_ci95_high'])}; "
            f"RMSE improvement {number(row['rmse_improvement_comparator_minus_neural'])}."
        )

    lines.extend([
        "",
        "## 写作边界",
        "",
        "1. 结果顺序固定为样本流、主要ITPC、次要神经指标、行为、条件内耦合、预测。",
        "2. 不把刺激锁定响应写成内源振荡夹带，不把同期耦合写成中介或因果机制。",
        "3. 健康老年人的急性任务结果不外推为认知障碍患者长期疗效。",
        "4. 所有摘要、正文、图注和投稿信使用同一机器判定的claim ceiling。",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": "manuscript_insert_rendered",
        "output": str(OUTPUT),
        "claim_ceiling": claims["claim_ceiling"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
