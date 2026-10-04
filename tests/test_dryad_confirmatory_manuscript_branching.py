import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "89_Dryad确认性Figure7与正文回填方案.md"


class DryadConfirmatoryManuscriptBranchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_all_primary_result_states_have_prespecified_language(self):
        for heading in (
            "主要ITPC为正且显著",
            "主要ITPC等效",
            "主要ITPC不显著且不等效",
        ):
            self.assertIn(heading, self.text)

    def test_main_figure_preserves_all_inferential_transitions(self):
        for panel in (
            "Panel A：设计、冻结和样本流",
            "Panel B：唯一主要神经估计量",
            "Panel C：行为效应与条件内耦合",
            "Panel D：严格样本外预测增量",
        ):
            self.assertIn(panel, self.text)

    def test_claim_boundaries_are_explicit(self):
        for boundary in (
            "不能推广到认知障碍患者",
            "不允许写成中介、因果链或认知调控机制",
            "不能写成“个体获益不可预测”",
        ):
            self.assertIn(boundary, self.text)


if __name__ == "__main__":
    unittest.main()
