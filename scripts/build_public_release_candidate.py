#!/usr/bin/env python3
"""Assemble an allowlisted public-data release candidate.

The builder deliberately excludes raw EEG, source archives, historical private-data
materials, notebook HTML, and local run/file manifests that can contain absolute paths.
It does not decide whether participant-level derived tables may be released; that
remains a manual license and disclosure-control gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release_candidate" / "public_eeg_validation_v0.5"

ROOT_FILES = [
    "README.md",
    ".zenodo.json",
    "LICENSE",
    "CITATION.cff",
    "CITATION.md",
    "RELEASE_NOTES.md",
    "THIRD_PARTY_NOTICES.md",
    "RELEASE_CHECKLIST.md",
    "requirements-qc.txt",
    "27_英文引言与方法初稿_v0.2.md",
    "28_英文结果与讨论初稿_v0.2.md",
    "30_稿件关键数字溯源表.csv",
    "34_正文Table1_队列记录与有效样本.md",
    "35_补充表S1-S5规格与口径.md",
    "36_补充图S1-S4图注及证据说明.md",
    "53_纯公开数据主图主表规格_v0.2.md",
    "54_最小重要效应与阴性结果强度判定.md",
    "55_行为效度与预测主图结果说明.md",
    "56_Dryad个体化theta听觉刺激数据库审计与GoNoGo.md",
    "57_Dryad公开代码可复现性与指标语义审计.md",
    "58_Dryad个体化theta主图候选说明.md",
    "60_正文Table2_验证层级与结论边界.md",
    "61_主图1-4结果说明与视觉核验.md",
    "62_参考文献管理器级核验.csv",
    "63_参考文献管理器级核验说明.md",
    "66_纯公开数据发布与生成式AI披露声明包.md",
    "67_公开发布候选与独立复现审计.md",
    "68_公开数据许可与派生表最小化审计.md",
    "69_行为终点测量信度与效度衰减风险审计.md",
    "70_神经指标信度-效度估计量对齐审计.md",
    "71_Dryad原始ZIP流式处理方案与访问闸门.md",
    "72_节律机制特异性外部反证与候选队列GoNoGo.md",
    "73_Dryad原始刺激包R0与S2单人R1审计.md",
    "74_Dryad_S2自动信号R1与R2冻结方案.md",
    "75_Dryad机制可识别性与R3主张升级规则.md",
    "76_Dryad_R2结果与路线决策.md",
    "77_路线A-B价值成本决策与R2b冻结前提.md",
    "78_Dryad_R2b空间QC开发记录.md",
    "79_研究进度与证据闸门看板.md",
    "80_创新强度提升与主张转换反证框架.md",
    "81_英文标题摘要与核心贡献_v0.3.md",
    "82_正文Figure1与Table2英文图表注_v0.1.md",
    "83_正文Figure2-7英文图注_v0.1.md",
    "84_全文术语与图表引用一致性审计.md",
    "85_Dryad_R2b技术验证结果与冻结后修复审计.md",
    "86_Dryad确认性神经行为分析冻结方案.md",
    "87_Dryad确认性冻结与结果前偏离审计.md",
    "88_Dryad确认性结果判定与推断完整性锁定.md",
    "89_Dryad确认性Figure7与正文回填方案.md",
    "90_Dryad确认性运行器结果盲修正说明.md",
    "91_Dryad确认性远程传输韧性结果盲修正说明.md",
    "92_Dryad确认性断点续传结果盲修正说明.md",
    "93_Dryad确认性IncompleteRead自动重试结果盲修正.md",
    "94_确认性解锁前稿件状态一致性审计.md",
    "95_Dryad确认性总体分析机器解锁闸门.md",
    "96_Dryad确认性主要结果反向分支补全.md",
    "97_Dryad确认性行为非劣效措辞结果盲修正.md",
    "98_Dryad确认性耦合条件特异性结果盲修正.md",
    "99_Dryad确认性预测验证层级结果盲修正.md",
    "100_Dryad确认性统一主张释放规则.md",
    "101_Dryad确认性全冻结链完整性验证.md",
    "102_Dryad确认性结果与正文回填_v1.md",
    "103_Dryad确认性论文整合与投稿就绪审计_v1.md",
    "104_英文全文整合稿_v0.5.md",
    "104_英文全文整合稿_v0.5.manifest.json",
    "105_确认性结果后目标期刊与篇幅适配_v1.md",
    "106_Alzheimers_Dementia投稿适配包_v1.md",
    "107_Alzheimers_Dementia英文全文_v0.1.md",
    "107_Alzheimers_Dementia英文全文_v0.1.manifest.json",
    "112_七队列伦理与许可证据链审计_v1.md",
    "113_Alzheimers_Dementia模拟编辑初筛与同行评审_v1.md",
    "114_英文全文逐句主张与数字溯源审计_v1.md",
]

SCRIPT_FILES = [
    "audit_dryad_bdf_pilot.py",
    "analyze_dryad_theta_validation_gate.py",
    "analyze_behavior_measurement_reliability.py",
    "analyze_estimand_alignment.py",
    "analyze_ds005048_dynamics_network.py",
    "analyze_ds006036_competitor_gate.py",
    "analyze_ds006036_network.py",
    "analyze_ds006780_behavioral_validity.py",
    "analyze_ds007648_behavioral_validity.py",
    "analyze_validation_strength.py",
    "audit_behavioral_validation_datasets.py",
    "audit_bids_events.py",
    "audit_ds006222_metadata.py",
    "audit_ds006780_fsiq_signal.py",
    "audit_ds006780_response_windows.py",
    "audit_public_release_package.py",
    "build_behavioral_validity_figures.py",
    "build_dryad_theta_figure.py",
    "build_dryad_confirmatory_figure7.py",
    "build_dryad_confirmatory_supplement.py",
    "build_consolidated_manuscript.py",
    "build_alzheimers_dementia_manuscript.py",
    "build_foundation_figures_v02.py",
    "build_manuscript_figures.py",
    "build_public_release_candidate.py",
    "build_supplementary_figures.py",
    "derive_ds006780_assr_features.py",
    "derive_ds007648_trial_features.py",
    "develop_dryad_r2b_spatial_qc.py",
    "develop_dryad_r2b_technical_subject.py",
    "extract_remote_zip_member_resumable.py",
    "extract_remote_zip_member_resumable_v2.py",
    "evaluate_dryad_r2_gate.py",
    "evaluate_dryad_r2b_technical_gate.py",
    "download_behavioral_validation_qc.sh",
    "download_ds004504_signal.sh",
    "download_ds005048_signal.sh",
    "download_ds006036_qc_sample.sh",
    "download_ds006222_qc_sample.sh",
    "join_binary_parts.py",
    "model_ds005048_dynamics.py",
    "model_ds006036_competitor_replication.py",
    "model_ds006036_network.py",
    "qc_behavioral_validation_candidates.py",
    "qc_ds004504_resting.py",
    "qc_ds005048.py",
    "qc_ds006036.py",
    "qc_ds006222_signal.py",
    "qc_dryad_signal_pilot.py",
    "refresh_release_manifest.py",
    "reproduce_from_derived.sh",
    "run_ds006780_streaming.sh",
    "run_ds007648_streaming.sh",
    "run_dryad_r2_subject.py",
    "run_dryad_r2b_development_subject.py",
    "run_dryad_r2b_validation_subject.py",
    "summarize_dryad_r2b_development.py",
    "summarize_dryad_r2b_validation.py",
    "freeze_dryad_r2b_protocol.py",
    "analyze_dryad_confirmatory_group.py",
    "assemble_dryad_confirmatory_claim_release.py",
    "audit_dryad_confirmatory_group_inference.py",
    "classify_dryad_confirmatory_behaviour_branch.py",
    "classify_dryad_confirmatory_coupling_branch.py",
    "classify_dryad_confirmatory_prediction_branch.py",
    "classify_dryad_confirmatory_primary_branch.py",
    "derive_dryad_confirmatory_subject.py",
    "execute_dryad_confirmatory_group_locked.py",
    "freeze_dryad_confirmatory_inference_audit.py",
    "freeze_dryad_confirmatory_protocol.py",
    "render_dryad_confirmatory_manuscript_insert.py",
    "run_dryad_confirmatory_postunlock.py",
    "run_dryad_confirmatory_subject.py",
    "run_dryad_confirmatory_subject_amended.py",
    "run_dryad_confirmatory_subject_amended_v2.py",
    "run_dryad_confirmatory_subject_amended_v3.py",
    "run_dryad_confirmatory_subject_amended_v4.py",
    "run_dryad_confirmatory_pending_batch.py",
    "summarize_dryad_confirmatory_completion.py",
    "validate_dryad_confirmatory_protocol.py",
    "verify_dryad_confirmatory_freeze_chain.py",
    "stream_remote_zip.py",
    "select_dryad_r2_pilot.py",
    "__init__.py",
]

EXCLUDED_TEST_NAMES = {
    "test_audit_private_eeg_package.py",
    "test_private_intake_templates.py",
    "test_private_precision.py",
    "test_submission_docx.py",
    "test_submission_companion_docx.py",
    "test_submission_cover_letter_docx.py",
    "test_submission_supplement_docx.py",
    "test_transfer_adaptation_package.py",
    "test_validate_private_freeze_package.py",
}

OUTPUT_DIRS = [
    "outputs/qc/ds004504_full_foundation",
    "outputs/qc/ds005048_full_foundation",
    "outputs/qc/ds006036_full_foundation",
    "outputs/competition/ds006036_exact_preprint",
    "outputs/competition/ds006036_competitor_models",
    "outputs/network/ds005048_full",
    "outputs/network/ds005048_models",
    "outputs/network/ds006036_models",
    "outputs/models/ds007648_behavioral_validity",
    "outputs/models/ds006780_behavioral_validity",
    "outputs/models/dryad_theta_validation_gate",
    "outputs/models/validation_strength",
    "outputs/models/behavior_measurement_reliability",
    "outputs/models/estimand_alignment",
    "outputs/models/dryad_confirmatory_group_v1",
    "outputs/models/dryad_confirmatory_inference_audit_v1",
    "outputs/models/dryad_confirmatory_protocol_freeze_v1",
    "outputs/models/dryad_confirmatory_inference_audit_freeze_v1",
    "outputs/models/dryad_group_execution_gate_freeze_v1",
    "outputs/models/dryad_primary_branch_amendment_freeze_v1",
    "outputs/models/dryad_behaviour_branch_amendment_freeze_v1",
    "outputs/models/dryad_coupling_branch_amendment_freeze_v1",
    "outputs/models/dryad_prediction_branch_amendment_freeze_v1",
    "outputs/models/dryad_claim_release_freeze_v1",
    "outputs/models/dryad_freeze_chain_verifier_freeze_v1",
    "outputs/status/dryad_confirmatory_claim_release_v1",
    "outputs/qc/dryad_raw_stream/S2_pilot_v3",
    "outputs/qc/dryad_raw_stream/S2_signal_qc_frozen_v1",
    "outputs/qc/dryad_raw_stream/R2_selection_frozen_v1",
    "outputs/qc/dryad_raw_stream/R2_subjects",
    "outputs/qc/dryad_raw_stream/R2_gate_frozen_v1",
    "outputs/qc/dryad_raw_stream/R2b_development_audit_v1",
    "outputs/qc/dryad_raw_stream/R2b_protocol_freeze_v1",
    "outputs/qc/dryad_raw_stream/R2b_validation_gate_v1",
    "outputs/qc/dryad_raw_stream/R2b_validation_summary_v1",
    "outputs/manuscript_figures_v02",
    "outputs/manuscript_figures_v03",
    "outputs/manuscript_supplement_v03",
    "outputs/manuscript_tables_v02",
]

ALLOWED_OUTPUT_SUFFIXES = {".csv", ".json", ".png", ".pdf", ".svg"}
EXCLUDED_OUTPUT_PARTS = {"notebook_preview", "file_manifest_sha256.csv", "run_summary.json"}

MINIMIZED_DS006780_COLUMNS = [
    "subject", "age", "sex", "group", "fsiq",
    "n_standard_valid_27", "dprime_27", "itpc_27",
    "local_log_snr_db_mean_27", "morlet_power_percent_change_mean_27",
    "n_standard_valid_40", "dprime_40", "itpc_40",
    "local_log_snr_db_mean_40", "morlet_power_percent_change_mean_40",
    "technical_usable", "technical_exclusion_reason",
]

EXTRA_FILES = [
    "references/manuscript_references_v02.bib",
    "manuscript_ad/introduction_v1.md",
    "manuscript_ad/discussion_v1.md",
    "templates/public_release_manifest_v02.csv",
    "configs/dryad_biosemi64_adjacency_v1.csv",
    "configs/dryad_biosemi64_adjacency_v1.json",
    "configs/dryad_confirmatory_protocol_v1.json",
    "configs/dryad_r2b_protocol_frozen_v1.json",
    "configs/manuscript_canonical_terms_v1.csv",
    "configs/research_progress_v1.csv",
    "data/public/ds004504/v1.0.9_derivatives/participants.tsv",
    "data/public/ds006036/v1.0.6/participants.tsv",
    "outputs/qc/dryad_raw_stream/stim_zip_members.csv",
    "outputs/innovation/claim_transition_falsification_matrix.csv",
]

DEIDENTIFIED_SUBJECT_TABLES = {
    "outputs/models/dryad_confirmatory_group_v1/nested_prediction_rows.csv",
    "outputs/models/dryad_confirmatory_group_v1/participant_behaviour.csv",
    "outputs/manuscript_figures_v03/Figure7B_participant_itpc_contrasts.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_for_release(source: Path, destination: Path, relative: Path) -> None:
    """Copy a file, minimizing unnecessary public clinical fields where defined."""
    if relative.as_posix() in DEIDENTIFIED_SUBJECT_TABLES:
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        if "subject" not in fieldnames:
            raise ValueError(f"Cannot deidentify subject table without subject column: {relative}")
        subjects = sorted(
            {row["subject"] for row in rows},
            key=lambda value: int(value[1:]) if re.fullmatch(r"C\d+", value) else int(value),
        )
        mapping = {
            subject: subject if re.fullmatch(r"C\d+", subject) else f"C{index:02d}"
            for index, subject in enumerate(subjects, start=1)
        }
        for row in rows:
            row["subject"] = mapping[row["subject"]]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    if relative.as_posix() != "outputs/models/ds006780_behavioral_validity/subject_level_features.csv":
        shutil.copy2(source, destination)
        return
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(MINIMIZED_DS006780_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Cannot minimize ds006780 table; missing columns: {sorted(missing)}")
        rows = [
            {column: row[column] for column in MINIMIZED_DS006780_COLUMNS}
            for row in reader
        ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MINIMIZED_DS006780_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def selected_files() -> list[Path]:
    files = [ROOT / relative for relative in ROOT_FILES]
    files.extend(ROOT / "scripts" / name for name in SCRIPT_FILES)
    files.extend(
        path for path in sorted((ROOT / "tests").glob("test_*.py"))
        if path.name not in EXCLUDED_TEST_NAMES
    )
    files.extend(ROOT / relative for relative in EXTRA_FILES)
    files.extend(
        path for path in sorted((ROOT / "data/public/ds006036/v1.0.6").glob(
            "sub-*/eeg/*_events.tsv"
        ))
    )
    for directory in OUTPUT_DIRS:
        base = ROOT / directory
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_OUTPUT_SUFFIXES:
                continue
            relative_parts = set(path.relative_to(base).parts)
            if relative_parts & EXCLUDED_OUTPUT_PARTS or path.name in EXCLUDED_OUTPUT_PARTS:
                continue
            files.append(path)
    unique = sorted(set(files))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing allowlisted files: " + ", ".join(map(str, missing)))
    return unique


def build(output: Path) -> dict[str, object]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty release candidate: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in selected_files():
        relative = source.relative_to(ROOT)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_for_release(source, destination, relative)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "source_class": "author_generated_or_public_derived",
                "manual_license_review": "required",
            }
        )
    manifest = output / "release_candidate_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(output),
        "file_count_excluding_manifest": len(rows),
        "manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
