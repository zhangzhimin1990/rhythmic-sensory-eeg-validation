#!/usr/bin/env python3
"""Assemble the current manuscript-facing sources into one submission draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "104_英文全文整合稿_v0.5.md"

CITATION_KEYS = {
    "L001": "L001_Duecker2024",
    "L003": "L003_Chan2022",
    "L007": "L007_Duecker2021",
    "L013": "L013_Ang2025",
    "L014": "L014_Lahijanian2024",
    "L015": "L015_Ntetska2025",
    "L016": "L016_vanDeursen2011",
    "L017": "L017_Murty2021",
    "L020": "L020_L027_Hajos2024",
    "L024": "L024_VelezPardo2026",
    "L026": "L026_Agger2023",
    "L027": "L020_L027_Hajos2024",
    "L031": "L031_Yoon2025",
    "L032": "L032_Hirano2020",
    "L035": "L035_Da2024",
    "L036": "L036_HOPE",
    "L037": "L037_Brickwedde2025",
    "L038": "L038_Darrell2026",
    "L039": "L039_GomezLombardi2026",
    "L040": "L040_Park2026",
    "L041": "L041_Breska2017",
    "L042": "L042_GammaAuditoryTrial",
    "L043": "L043_Qi2026",
    "L044": "L044_Attokaren2026",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def between(text: str, start: str, end: str | None = None) -> str:
    content = text.split(start, 1)[1]
    if end is not None:
        content = content.split(end, 1)[0]
    return content.strip()


def replace_citations(text: str) -> str:
    pattern = re.compile(r"\[(L\d{3}(?:, L\d{3})*)\]")

    def repl(match: re.Match[str]) -> str:
        labels = match.group(1).split(", ")
        keys = []
        for label in labels:
            if label not in CITATION_KEYS:
                raise ValueError(f"No bibliography mapping for {label}")
            key = CITATION_KEYS[label]
            if key not in keys:
                keys.append(key)
        return "[" + "; ".join(f"@{key}" for key in keys) + "]"

    converted = pattern.sub(repl, text)
    unresolved = re.findall(r"\[L\d{3}", converted)
    if unresolved:
        raise ValueError(f"Unresolved internal citations: {unresolved}")
    return converted


def main_figure_legends() -> str:
    figure1 = (ROOT / "82_正文Figure1与Table2英文图表注_v0.1.md").read_text(encoding="utf-8")
    figures2_7 = (ROOT / "83_正文Figure2-7英文图注_v0.1.md").read_text(encoding="utf-8")
    f1 = between(figure1, "## Figure 1", "## Table 2")
    f2_7 = between(figures2_7, "## Figure 2", "## Canonical terminology and source alignment")
    return "# Figure legends\n\n## Figure 1\n\n" + f1 + "\n\n## Figure 2\n\n" + f2_7


def table_legends() -> str:
    figure1 = (ROOT / "82_正文Figure1与Table2英文图表注_v0.1.md").read_text(encoding="utf-8")
    table2 = between(figure1, "## Table 2", "## Source alignment")
    return """# Table legends

## Table 1

**Table 1. Public human EEG cohorts, acquisition states, effective sample sizes, and roles in the inferential-transition analysis.** Effective sample sizes are outcome specific and must not be added across datasets or endpoints. Cohorts that provide contextual, data-availability, or mechanism-specificity constraints are distinguished from cohorts contributing direct transition tests.

## Table 2

""" + table2


def supplementary_legends() -> str:
    source = (ROOT / "36_补充图S1-S4图注及证据说明.md").read_text(encoding="utf-8")
    sections = []
    for number in range(1, 5):
        start = f"## Supplementary Figure S{number}."
        end = f"## Supplementary Figure S{number + 1}." if number < 4 else "## Reproducibility package"
        block = start + between(source, start, end)
        heading, body = block.split("\n", 1)
        legend = body.split("**Evidence boundary.**", 1)[0].strip()
        sections.append(heading + "\n\n" + legend)

    sections.extend(
        [
            """## Supplementary Figure S5. Phase-locked and non-phase-locked spectral components

**Legend.** Paired theta-plus-minus-non-rhythmic differences in frontocentral evoked, total-power, and induced local log signal-to-noise ratio in the frozen 35-participant cohort. Points are mean paired differences and bars are two-sided 95% confidence intervals. Holm-adjusted p values control the three key secondary neural endpoints. The selective evoked difference supports a phase-consistent response under the tested decomposition but does not distinguish endogenous oscillatory entrainment from repeated or temporally predictable evoked activity.""",
            """## Supplementary Figure S6. Internal consistency depends on condition and metric

**Legend.** Spearman–Brown-corrected odd–even and time-half coefficients for frontocentral inter-trial phase coherence and evoked local log signal-to-noise ratio in theta-plus and non-rhythmic conditions. Bars are 95% participant-bootstrap confidence intervals from 20,000 resamples. These estimates describe within-session consistency of participant ordering and are not between-day test–retest reliability.""",
        ]
    )
    return "# Supplementary figure legends\n\n" + "\n\n".join(sections)


def supplementary_table_legends() -> str:
    return """# Supplementary table legends

## Supplementary Table S6

**Supplementary Table S6. Participant-level technical and analysis flow for the frozen Dryad confirmation.** Rows use sequential analytic identifiers rather than released subject labels. The table reports individualized and common analysis frequencies, frozen-QC channel counts, reproduced usable epochs, and event-valid and EEG-usable trial counts for theta-plus and non-rhythmic conditions. It contains no source filenames or file checksums.

## Supplementary Table S7

**Supplementary Table S7. Complete locked Dryad confirmatory estimates and multiplicity families.** The table retains the primary ITPC contrast, three Holm-corrected secondary neural contrasts, three Holm-corrected behavioral contrasts, four condition-specific coupling slopes, and two estimation-only prediction comparisons. Blank adjusted-p fields indicate that no multiplicity-adjusted null-hypothesis test was defined for that row; they do not denote zero.
"""


def build(output: Path) -> dict[str, object]:
    front_path = ROOT / "81_英文标题摘要与核心贡献_v0.3.md"
    methods_path = ROOT / "27_英文引言与方法初稿_v0.2.md"
    results_path = ROOT / "28_英文结果与讨论初稿_v0.2.md"
    front = front_path.read_text(encoding="utf-8")
    methods = methods_path.read_text(encoding="utf-8")
    results = results_path.read_text(encoding="utf-8")

    title = between(front, "# Title", "# Abstract")
    abstract = between(front, "# Abstract", "# Keywords")
    keywords = between(front, "# Keywords", "# Core contribution statement")
    introduction_and_methods = "# Introduction\n\n" + between(methods, "# Introduction", "# Internal citation crosswalk")
    results_and_discussion = "# Results\n\n" + between(results, "# Results")

    manuscript = f"""---
title: "{title}"
bibliography: references/manuscript_references_v02.bib
date: 2026-10-01
version: 0.5
---

# Abstract

{abstract}

**Keywords:** {keywords}

{introduction_and_methods}

{results_and_discussion}

{table_legends()}

{main_figure_legends()}

{supplementary_legends()}

{supplementary_table_legends()}

# References

References are rendered from `references/manuscript_references_v02.bib` using the selected journal style.
"""
    manuscript = replace_citations(manuscript)
    manuscript = re.sub(r"\n{3,}", "\n\n", manuscript).rstrip() + "\n"
    output.write_text(manuscript, encoding="utf-8")

    body_before_legends = manuscript.split("# Table legends", 1)[0]
    words = re.findall(r"\b[\w’–-]+\b", body_before_legends, flags=re.UNICODE)
    citations = sorted(set(re.findall(r"@([A-Za-z0-9_]+)", manuscript)))
    manifest = {
        "manuscript": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "body_word_count_including_abstract": len(words),
        "citation_keys": citations,
        "citation_count_unique": len(citations),
        "main_figures": 7,
        "main_tables": 2,
        "supplementary_figures": 6,
        "supplementary_tables_added_here": ["S6", "S7"],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [front_path, methods_path, results_path,
                         ROOT / "82_正文Figure1与Table2英文图表注_v0.1.md",
                         ROOT / "83_正文Figure2-7英文图注_v0.1.md",
                         ROOT / "36_补充图S1-S4图注及证据说明.md",
                         ROOT / "references/manuscript_references_v02.bib"]
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
