# Human rhythmic sensory EEG validation across public cohorts

This repository supports a multi-cohort secondary analysis of human EEG responses to rhythmic visual, auditory, and audiovisual stimulation. The central question is when a measurable frequency-following response can—and cannot—be upgraded to a reliable, cognitively informative, or individually predictive marker.

## Scientific scope

Seven public datasets contribute distinct evidential roles: OpenNeuro ds004504, ds006036, ds005048, ds006222, ds007648, ds006780, and Dryad doi:10.5061/dryad.t76hdr8dm. Raw amplitudes are not pooled across cohorts or modalities. The study separates detectability, within-recording reliability, experimental sensitivity, concurrent validity, the mean active-control behavioral effect, within-person neural coupling, and out-of-sample prediction.

This is not a clinical trial and does not test chronic treatment efficacy or disease modification. The personalized-theta analysis now includes a condition-blind frozen raw-EEG confirmation in 35 healthy older adults. The active-control contrast supports phase-locked target engagement and an acute task-performance advantage, but the prespecified coupling and nested-prediction gates do not support treating the measured phase response as a condition-specific behavioral mechanism or individual treatment-selection marker.

## Reproducibility levels

### Level 1: regenerate manuscript figures from audited derived outputs

The release candidate contains the code, source-backed derived tables, model outputs, tests, and manifests needed to rebuild Figures 1–7 and rerun the claim-strength analysis without redistributing source EEG.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-qc.txt
bash scripts/reproduce_from_derived.sh .venv/bin/python
```

Level 1 verifies the manuscript-facing calculations and figures. It does not independently recreate EEG features from the source recordings.

### Level 2: recreate derived features from public source data

Download the exact dataset versions listed in the manuscript Methods and place them under `data/public/` using the documented directory layout. Raw-to-derived scripts and streaming helpers are provided, but source EEG is deliberately excluded from the release. Large-dataset execution should be performed only after checking storage requirements, dataset licenses, and the frozen inclusion rules.

For the two large Dryad ZIP/ZIP64 objects, `scripts/stream_remote_zip.py` provides a strict HTTP-Range reader that lists the central directory and extracts one member at a time without retaining signed query parameters. The first five result-blind development participants exposed a spatial-QC failure in the original R2 route. The amended BioSemi64 spatial-neighbour protocol was therefore calibrated only on those five participants, frozen, and applied condition-blind to the untouched 39-person validation set. Thirty-five participants passed the prespecified technical rules and formed the confirmatory cohort; four technical failures and all five development participants were excluded from confirmatory estimation.

The audited release candidate records this complete chain: development and validation gates, frozen confirmatory protocol, result-blind transport and runner amendments, 35-person group analysis, inference audit, claim-release branches, Figure 7, supplementary artifacts, journal-adapted manuscript, N001–N057 number traceability, and line-by-line claim-boundary audit. The release does not redistribute BDF files or signed URLs. Participant-level Dryad derived tables included for figure reproduction use sequential analytic identifiers. The frozen chain verifies that condition effects were not present in the pre-results manifests; it does not convert the secondary analysis into a prospectively preregistered trial.

## Key manuscript assets

- `27_英文引言与方法初稿_v0.2.md`
- `28_英文结果与讨论初稿_v0.2.md`
- `30_稿件关键数字溯源表.csv`
- `69_行为终点测量信度与效度衰减风险审计.md`
- `70_神经指标信度-效度估计量对齐审计.md`
- `71_Dryad原始ZIP流式处理方案与访问闸门.md`
- `74_Dryad_S2自动信号R1与R2冻结方案.md`
- `75_Dryad机制可识别性与R3主张升级规则.md`
- `76_Dryad_R2结果与路线决策.md`
- `77_路线A-B价值成本决策与R2b冻结前提.md`
- `78_Dryad_R2b空间QC开发记录.md`
- `79_研究进度与证据闸门看板.md`
- `80_创新强度提升与主张转换反证框架.md`
- `81_英文标题摘要与核心贡献_v0.3.md`
- `82_正文Figure1与Table2英文图表注_v0.1.md`
- `83_正文Figure2-7英文图注_v0.1.md`
- `84_全文术语与图表引用一致性审计.md`
- `86_Dryad确认性神经行为分析冻结方案.md`
- `101_Dryad确认性全冻结链完整性验证.md`
- `102_Dryad确认性结果与正文回填_v1.md`
- `104_英文全文整合稿_v0.5.md`
- `107_Alzheimers_Dementia英文全文_v0.1.md`
- `outputs/manuscript_tables_v02/Table2_inferential_transitions.csv`
- `62_参考文献管理器级核验.csv`
- `references/manuscript_references_v02.bib`
- `outputs/manuscript_figures_v02/`
- `outputs/manuscript_figures_v03/`
- `outputs/manuscript_supplement_v03/`
- `outputs/manuscript_tables_v02/`

## Current evidence boundaries

- A stimulus-frequency peak is described as frequency following or target engagement unless stronger dynamical entrainment criteria are tested.
- Split-half coefficients from one recording are not called between-day test–retest reliability.
- Behavioral reliability is interpreted only when its level matches the validity estimand: aggregate split-half reliability informs between-participant analyses, not within-participant trial coefficients.
- Neural reliability is transferred to a validity claim only when the neural metric and aggregation level are identical; aggregate ITPC reliability does not establish reliability of trial-state phase scores or local log-SNR.
- Non-significant or imprecise estimates are not treated as proof of no association.
- Sensitivity bounds are not clinical minimum important differences.
- Group mean effects, within-person coupling, and held-out individual prediction are separate estimands.
- The ds006036 10-Hz analysis is a same-data computational replication, not an external replication.
- The HOPE pivotal trial is completed with 673 enrolled participants, but the registry had no posted results at the 2026-10-03 literature cutoff; completion status is not efficacy evidence.
- The frozen 35-person Dryad result supports target engagement and acute task-performance advantage without condition-specific neural–behavioral coupling; it does not establish mediation, endogenous resonance, patient benefit, chronic efficacy, or disease modification.

## Software and tests

Python 3.12.14 and exact top-level package versions are recorded in `requirements-qc.txt`. Tests use the standard-library `unittest` runner:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The public release candidate excludes historical private-data planning code and documents. Automated checks do not replace manual review of dataset licenses, ethics statements, and third-party copyright.

Participant-level derived tables are limited to fields needed for manuscript-facing reproducibility. In particular, the release builder removes unused medication and extended phenotyping fields from the ds006780 analysis table even though the source dataset is public and CC0.

## Data and code availability

Dataset DOIs, versions, code-release language, and AI-use disclosure are maintained in `66_纯公开数据发布与生成式AI披露声明包.md`. A repository DOI and immutable release identifier will be added only after the final release candidate passes the machine and manual publication gates.

The authors approved public release of the original analysis code under the MIT License on 4 October 2026. The audited release candidate contains the license and citation metadata and has passed derived-output reproduction and machine privacy/portability screening. It remains local until a clean public repository and immutable archive are created. `THIRD_PARTY_NOTICES.md` and `RELEASE_CHECKLIST.md` distinguish original project code from source datasets, manuscript text, and third-party materials that are not relicensed.
