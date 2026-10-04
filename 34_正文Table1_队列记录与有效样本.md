# Table 1：公开队列、证据角色与结局特异有效样本

版本：0.2  
更新：2026-09-21  
机器可读权威源：`outputs/manuscript_tables_v02/Table1_public_cohorts.csv`

| Dataset | Population and public N | Primary effective sample | Evidence role | Principal limitation |
|---|---|---|---|---|
| ds004504 | AD 36; FTD 23; CN 29; N=88 | Resting-feature/MMSE N=88 | Disease-related resting spectral axis; linked predictors for ds006036 | Control MMSE ceiling; no stimulation |
| ds006036 | Same 88 participants as ds004504 | Eyes-open N=80/76/78/77 by frequency; four-frequency N=68; selectivity N=76 | Visual detectability; state/frequency/metric dependence; resting-to-visual correspondence | Short reconstructed exposure; state and metric change together |
| ds005048 | Older normal/MCI/AD; N=35 | Target engagement N=35; MMSE N=33 | Auditory 40-Hz detectability, within-recording stability, and MMSE boundary | Small heterogeneous groups; no task outcome |
| ds006222 | Healthy young adults; 69 participants/70 records | Six-record signal gate; behavior unavailable | Signal and public-release completeness boundary | Trial outcome and occlusion files unavailable; full cohort not modeled |
| ds007648 | Healthy adults; public N=22 | Neural N=20; 8,579 accuracy trials; 7,456 correct RT trials | Within-participant 36/40-Hz behavioral validity and LOSO prediction | Two corrupted/truncated releases; small healthy-young cohort |
| ds006780 | ASD/TD/unaffected sibling children; ASSR N=123 | Technical N=111; FSIQ N=109 | Reliability, developmental sensitivity, adjusted d-prime/FSIQ validity, repeated CV | Age/diagnosis/ability-range coupling; no older intervention cohort |
| Dryad t76hdr8dm | Healthy older adults aged 60–75; N=44 | Released table N=44; 132 rhythmic participant-condition rows | Active-control effect, between/within neural decomposition, pretreatment benefit prediction | Raw BDF gate pending; author pipeline has manual steps and code defects |

## Locked reporting rules

1. ds004504 and ds006036 contain the same 88 participants and are not summed as 176 independent participants.
2. Records available, structural quality-control N, neural-analysis N, and outcome-specific N remain separate.
3. Trials, runs, and blocks are repeated observations rather than additional participants.
4. Split-half coefficients are described as within-recording reliability or stability, not between-day test–retest reliability.
5. ds006222 contributes a public-release/data-availability boundary and is not used for behavioral efficacy.
6. Dryad findings are labeled released-table analyses until raw event and signal reconstruction is complete.
7. Population, paradigm, acquisition, reference, and metric differences preclude pooling raw EEG amplitudes across datasets.

## Table note

AD indicates Alzheimer disease; ASD, autism spectrum disorder; CN, cognitively normal; CV, cross-validation; FSIQ, full-scale intelligence quotient; FTD, frontotemporal dementia; LOSO, leave-one-participant-out; MCI, mild cognitive impairment; MMSE, Mini-Mental State Examination; RT, response time; TD, typically developing. Values and effective sample sizes are outcome specific. Full acquisition details, stimulus definitions, licenses, and DOIs are retained in the machine-readable table.
