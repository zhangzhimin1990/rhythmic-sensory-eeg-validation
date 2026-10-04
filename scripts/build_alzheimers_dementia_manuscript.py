#!/usr/bin/env python3
"""Build the Alzheimer’s & Dementia journal-adapted manuscript."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "107_Alzheimers_Dementia英文全文_v0.1.md"


def between(text: str, start: str, end: str | None = None) -> str:
    content = text.split(start, 1)[1]
    if end is not None:
        content = content.split(end, 1)[0]
    return content.strip()


def words(text: str) -> int:
    return len(re.findall(r"\b[\w’–-]+\b", text, flags=re.UNICODE))


def number_subsections(text: str, section: int) -> str:
    major = 0
    minor = 0
    lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            major += 1
            minor = 0
            line = f"## {section}.{major} {line[3:]}"
        elif line.startswith("### "):
            minor += 1
            line = f"### {section}.{major}.{minor} {line[4:]}"
        lines.append(line)
    return "\n".join(lines)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, object]:
    package_path = ROOT / "106_Alzheimers_Dementia投稿适配包_v1.md"
    generic_path = ROOT / "104_英文全文整合稿_v0.5.md"
    intro_path = ROOT / "manuscript_ad" / "introduction_v1.md"
    discussion_path = ROOT / "manuscript_ad" / "discussion_v1.md"
    legend_path = ROOT / "83_正文Figure2-7英文图注_v0.1.md"
    table_legend_path = ROOT / "82_正文Figure1与Table2英文图表注_v0.1.md"

    package = package_path.read_text(encoding="utf-8")
    generic = generic_path.read_text(encoding="utf-8")
    intro = intro_path.read_text(encoding="utf-8").strip()
    discussion = discussion_path.read_text(encoding="utf-8").strip()

    title = between(package, "## Title", "## Structured abstract")
    abstract = between(package, "## Structured abstract", "## Keywords")
    keywords = between(package, "## Keywords", "## Research in Context")
    research_context = between(package, "## Research in Context", "## Highlights")

    methods = between(generic, "# Methods", "# Results")
    results = between(generic, "# Results", "# Discussion").replace("Table 2", "Table 1")
    methods = number_subsections(methods, 2)
    results = number_subsections(results, 3)

    base_figure1 = table_legend_path.read_text(encoding="utf-8")
    figure1 = between(base_figure1, "## Figure 1", "## Table 2")
    figures2_7 = between(
        legend_path.read_text(encoding="utf-8"),
        "## Figure 2",
        "## Canonical terminology and source alignment",
    )
    table1 = between(base_figure1, "## Table 2", "## Source alignment").replace(
        "**Table 2.", "**Table 1.", 1
    )

    supplemental = between(generic, "# Supplementary figure legends", "# References")
    supplemental += """

## Supplementary Table S8

**Supplementary Table S8. Public human EEG cohorts, acquisition states, effective sample sizes, and roles in the inferential-transition analysis.** Effective sample sizes are outcome specific and must not be added across datasets or endpoints. Cohorts providing contextual, data-availability, or mechanism-specificity constraints are distinguished from cohorts contributing direct transition tests.
"""

    manuscript = f"""---
title: "{title}"
bibliography: references/manuscript_references_v02.bib
journal: Alzheimer's & Dementia
article_type: Research Article
date: 2026-10-01
version: 0.1
---

# Abstract

{abstract}

**Keywords:** {keywords}

# Research in Context

{research_context}

{intro}

# 2. Methods

{methods}

# 3. Results

{results}

{discussion}

# Table legend

## Table 1

{table1}

# Figure legends

## Figure 1

{figure1}

## Figure 2

{figures2_7}

# Supplementary legends

{supplemental}

# References

References are rendered from `references/manuscript_references_v02.bib` using the journal style.
"""
    manuscript = re.sub(r"\n{3,}", "\n\n", manuscript).rstrip() + "\n"
    OUTPUT.write_text(manuscript, encoding="utf-8")

    sections = {
        "abstract": between(manuscript, "# Abstract", "# Research in Context"),
        "research_in_context": between(manuscript, "# Research in Context", "# 1. Introduction"),
        "introduction": between(manuscript, "# 1. Introduction", "# 2. Methods"),
        "methods": between(manuscript, "# 2. Methods", "# 3. Results"),
        "results": between(manuscript, "# 3. Results", "# 4. Discussion"),
        "discussion": between(manuscript, "# 4. Discussion", "# Table legend"),
    }
    manifest = {
        "manuscript": str(OUTPUT.relative_to(ROOT)),
        "sha256": sha256(OUTPUT),
        "title_characters": len(title),
        "section_word_counts": {name: words(value) for name, value in sections.items()},
        "unique_citations": len(set(re.findall(r"@([A-Za-z0-9_]+)", manuscript))),
        "main_display_items": {"figures": 7, "tables": 1, "total": 8},
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [package_path, generic_path, intro_path, discussion_path, legend_path, table_legend_path]
        },
    }
    manifest_path = OUTPUT.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
