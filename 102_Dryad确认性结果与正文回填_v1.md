# Dryad确认性结果与正文回填

状态：35/35冻结确认性队列完成后由机器输出生成；数值来自锁定总体分析和推断审计。

## 统一主张判定

- Claim ceiling: `target_engagement_and_behaviour_without_specific_coupling`
- Abstract core sentence: The active-control contrast supported target engagement and an acute task-performance advantage, but did not establish condition-specific neural-behaviour coupling.
- Supporting modifiers: no consistent internal out-of-sample prediction increment was established
- Causal mechanism, chronic efficacy, clinical treatment selection, and patient generalisation remain locked.

## 主要神经终点

- Frontocentral ITPC: mean theta-plus minus non-rhythmic difference 0.088; 95% CI 0.051 to 0.124; paired d_z=0.830; p_value_t_two_sided=2.23e-05.
- 90% CI 0.057 to 0.118; operational equivalence within ±0.30 d_z: False.
- ≥4 good ROI-channel sensitivity: estimate 0.088; 95% CI 0.051 to 0.124.

## 次要神经终点

- evoked_local_log_snr_db: mean theta-plus minus non-rhythmic difference 7.518; 95% CI 4.522 to 10.514; paired d_z=0.862; p_value_holm_key_secondary=3.83e-05.
- total_power_local_log_snr_db: mean theta-plus minus non-rhythmic difference 0.306; 95% CI -0.097 to 0.709; paired d_z=0.261; p_value_holm_key_secondary=0.264.
- induced_local_log_snr_db: mean theta-plus minus non-rhythmic difference -0.097; 95% CI -0.393 to 0.200; paired d_z=-0.112; p_value_holm_key_secondary=0.512.

## 内部一致性（非跨日重测信度）

- f_theta_plus / itpc / odd_even: Spearman–Brown=0.956; 95% bootstrap CI 0.893 to 0.982.
- f_theta_plus / itpc / time_half: Spearman–Brown=0.961; 95% bootstrap CI 0.903 to 0.984.
- f_theta_plus / evoked_local_log_snr_db / odd_even: Spearman–Brown=0.855; 95% bootstrap CI 0.663 to 0.942.
- f_theta_plus / evoked_local_log_snr_db / time_half: Spearman–Brown=0.717; 95% bootstrap CI 0.384 to 0.881.
- non_rhythmic / itpc / odd_even: Spearman–Brown=0.964; 95% bootstrap CI 0.896 to 0.990.
- non_rhythmic / itpc / time_half: Spearman–Brown=0.946; 95% bootstrap CI 0.867 to 0.978.
- non_rhythmic / evoked_local_log_snr_db / odd_even: Spearman–Brown=0.135; 95% bootstrap CI -0.735 to 0.571.
- non_rhythmic / evoked_local_log_snr_db / time_half: Spearman–Brown=0.236; 95% bootstrap CI -0.561 to 0.622.

## 行为主动对照效应

- accuracy: mean theta-plus minus non-rhythmic difference -0.006; 95% CI -0.014 to 0.002; paired d_z=-0.270; p_value_holm_behaviour=0.119.
- median_correct_rt_ms: mean theta-plus minus non-rhythmic difference -27.727; 95% CI -36.171 to -19.284; paired d_z=-1.128; p_value_holm_behaviour=3.49e-07.
- inverse_efficiency_ms: mean theta-plus minus non-rhythmic difference -25.856; 95% CI -35.455 to -16.257; paired d_z=-0.925; p_value_holm_behaviour=8.31e-06.

## 条件内神经—行为耦合

- correctness_binomial_gee / non_rhythmic: simple slope -0.098; 95% CI -1.187 to 0.991; Holm p=1.000.
- correctness_binomial_gee / f_theta_plus: simple slope -0.072; 95% CI -0.879 to 0.735; Holm p=1.000.
- correct_rt_gaussian_gee / non_rhythmic: simple slope 0.006; 95% CI -0.029 to 0.042; Holm p=1.000.
- correct_rt_gaussian_gee / f_theta_plus: simple slope -0.004; 95% CI -0.026 to 0.019; Holm p=1.000.

## 严格样本外预测增量

- Versus training_fold_mean: MAE improvement 3.874; 95% bootstrap CI -0.431 to 8.032; RMSE improvement 2.793.
- Versus baseline: MAE improvement -1.024; 95% bootstrap CI -3.207 to 1.013; RMSE improvement 0.225.

## 写作边界

1. 结果顺序固定为样本流、主要ITPC、次要神经指标、行为、条件内耦合、预测。
2. 不把刺激锁定响应写成内源振荡夹带，不把同期耦合写成中介或因果机制。
3. 健康老年人的急性任务结果不外推为认知障碍患者长期疗效。
4. 所有摘要、正文、图注和投稿信使用同一机器判定的claim ceiling。
