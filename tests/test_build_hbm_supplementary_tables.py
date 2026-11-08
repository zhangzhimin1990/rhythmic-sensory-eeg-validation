from scripts.build_hbm_supplementary_tables import (
    build_ds007648_family,
    build_fsiq_sensitivity,
)


def test_ds007648_declared_family_is_complete_and_unique():
    table = build_ds007648_family()
    assert len(table) == 8
    assert table[["outcome", "term"]].duplicated().sum() == 0
    assert set(table["outcome"]) == {"accuracy", "rt"}
    assert table["holm_q_within_outcome_four_tests"].between(0, 1).all()


def test_fsiq_sensitivity_contains_main_and_restriction_variants():
    table = build_fsiq_sensitivity()
    assert len(table) == 9
    assert table["variant"].is_unique
    assert "main" in set(table["variant"])
    assert "published_age_and_fsiq" in set(table["variant"])
    assert table["n_subjects"].between(1, 109).all()
