import tempfile
import unittest
from pathlib import Path

from scripts.build_public_release_candidate import build


class PublicReleaseBuilderTests(unittest.TestCase):
    def test_release_is_public_only_and_has_reproduction_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            summary = build(output)
            self.assertGreater(summary["file_count_excluding_manifest"], 100)
            self.assertTrue((output / "README.md").is_file())
            self.assertTrue((output / "scripts/reproduce_from_derived.sh").is_file())
            reproduce = (output / "scripts/reproduce_from_derived.sh").read_text(encoding="utf-8")
            self.assertIn('[[ -f "$PROJECT_ROOT/release_candidate_manifest.csv" ]]', reproduce)
            self.assertTrue((output / "scripts/qc_dryad_signal_pilot.py").is_file())
            self.assertTrue((output / "scripts/select_dryad_r2_pilot.py").is_file())
            self.assertTrue((output / "scripts/develop_dryad_r2b_spatial_qc.py").is_file())
            self.assertTrue((output / "scripts/develop_dryad_r2b_technical_subject.py").is_file())
            self.assertTrue((output / "scripts/evaluate_dryad_r2b_technical_gate.py").is_file())
            self.assertTrue((output / "74_Dryad_S2自动信号R1与R2冻结方案.md").is_file())
            self.assertTrue((output / "78_Dryad_R2b空间QC开发记录.md").is_file())
            self.assertTrue((output / "81_英文标题摘要与核心贡献_v0.3.md").is_file())
            self.assertTrue((output / "82_正文Figure1与Table2英文图表注_v0.1.md").is_file())
            self.assertTrue((output / "83_正文Figure2-7英文图注_v0.1.md").is_file())
            self.assertTrue((output / "84_全文术语与图表引用一致性审计.md").is_file())
            self.assertTrue((output / "104_英文全文整合稿_v0.5.md").is_file())
            self.assertTrue((output / "107_Alzheimers_Dementia英文全文_v0.1.md").is_file())
            self.assertTrue((output / "THIRD_PARTY_NOTICES.md").is_file())
            self.assertTrue((output / "RELEASE_CHECKLIST.md").is_file())
            self.assertTrue((output / "manuscript_ad/introduction_v1.md").is_file())
            self.assertTrue((output / "manuscript_ad/discussion_v1.md").is_file())
            self.assertTrue((output / "scripts/build_dryad_confirmatory_figure7.py").is_file())
            self.assertTrue((output / "scripts/build_dryad_confirmatory_supplement.py").is_file())
            self.assertFalse((output / "tests/test_submission_docx.py").exists())
            self.assertFalse((output / "tests/test_submission_companion_docx.py").exists())
            self.assertFalse((output / "tests/test_submission_cover_letter_docx.py").exists())
            self.assertFalse((output / "tests/test_submission_supplement_docx.py").exists())
            self.assertFalse((output / "tests/test_transfer_adaptation_package.py").exists())
            self.assertTrue((output / "outputs/manuscript_figures_v03/Figure7_dryad_confirmatory.pdf").is_file())
            self.assertTrue((output / "outputs/manuscript_supplement_v03/TableS7_dryad_confirmatory_estimates.csv").is_file())
            self.assertFalse((output / "32_主图1与4-6图注及证据说明.md").exists())
            self.assertTrue((output / "configs/manuscript_canonical_terms_v1.csv").is_file())
            self.assertTrue((output / "outputs/innovation/claim_transition_falsification_matrix.csv").is_file())
            self.assertTrue((output / "outputs/qc/dryad_raw_stream/S2_signal_qc_frozen_v1/qc_summary.json").is_file())
            self.assertTrue((output / "outputs/qc/dryad_raw_stream/R2_selection_frozen_v1/r2_selected_subjects.csv").is_file())
            self.assertTrue((output / "outputs/manuscript_tables_v02/Table1_public_cohorts.csv").is_file())
            minimized = (output / "outputs/models/ds006780_behavioral_validity/subject_level_features.csv").read_text(encoding="utf-8")
            header = minimized.splitlines()[0]
            self.assertIn("dprime_40", header)
            self.assertNotIn("medication", header.lower())
            self.assertNotIn("ados", header.lower())
            dryad_subjects = (output / "outputs/models/dryad_confirmatory_group_v1/participant_behaviour.csv").read_text(encoding="utf-8")
            self.assertIn("C01", dryad_subjects)
            self.assertNotRegex(dryad_subjects, r"(?m)^12,")
            paths = [path.relative_to(output).as_posix().lower() for path in output.rglob("*")]
            self.assertFalse(any("private" in path for path in paths))
            self.assertFalse(any(Path(path).suffix in {".edf", ".bdf", ".set", ".fdt"} for path in paths))

    def test_nonempty_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            (output / "keep.txt").write_text("user material", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build(output)


if __name__ == "__main__":
    unittest.main()
