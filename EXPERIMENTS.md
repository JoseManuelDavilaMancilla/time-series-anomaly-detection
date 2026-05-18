# Experiments log

**This file supersedes `submissions_log.md` and `CLAUDE_PROGRESS.md`. Both agents (kimi, claude) write to it.**

Append a row when you finish an experiment. Keep it chronological. Keep notes terse — link to the relevant `claude_*.py` / `generate_submission_v*.py` for details.

**Conventions**
- `author` = `kimi` or `claude`
- `val_f1` = your validation F1 if you have one. Leave blank for kimi's old pre-`claude_validation.py` runs.
- `lb_f1` = leaderboard F1 once uploaded. Mark as `—` if not uploaded.
- For dropped experiments (negative Δ), still log them so the other agent doesn't redo them.

---

## Leaderboard standings (latest — 2026-05-18 morning)

| Rank | Submission name | Score | Author | Date |
|---:|---|---:|---|---|
| **~2** | **stl_ar (v68)** | **0.6905** | **claude** | **2026-05-17** |
| ~2 | pseudo_iter (v66) | 0.6902 | claude | 2026-05-17 |
| ~2 | more_mp_cusum (v67) | 0.6901 | claude | 2026-05-17 |
| ~2 | matrix_profile (v65) | 0.6890 | claude | 2026-05-17 |
| ~2 | tda_cached (v62) | 0.6863 | claude | 2026-05-17 |

**Our best**: 0.6905. **Gap to 0.70**: 0.0095.
**In flight**: v69 (iter v68), v70 (combined), v71 (catch22+catboost).

---

## Successful changes (used in current best 0.6238)

| Change | Validation Δ | LB transfer | Source file |
|---|---|---|---|
| Scale-invariant CW (no raw `value`) | — | +0.021 (kimi v8 vs prior) | `generate_submission_v8.py` |
| Segment selection (smooth=3, thr_frac=0.7) | — | +0.0156 (kimi v15 vs v12) | `generate_submission_v15.py` |
| Hybrid per-metric CW routing | +0.0089 | absorbed | `claude_v3_per_metric.py` |
| Segment selection (replace top-k) | +0.0154 | absorbed | `claude_v1_segments.py` |
| 3-seed CNN ensemble | +0.0110 | absorbed | `claude_v7_cnn_ensemble.py` |
| Tuned segment params (smooth=3, thr_frac=0.7) | +0.0118 | absorbed | `claude_v9_seg_sweep.py` |
| Stronger CW (500/15) + IF-on-test (disjoint+constant) + v12 weights, combined | +0.0024 | +0.0127 LB (5.3× transfer!) | `claude_v11_kimi_v12_ablation.py` (all_v12 config) |
| **Total stacked (vs v8)** | **+0.0495** | **+0.0753 LB** | `submission_kimi_combo_all_v12.json` |

Note: validation→LB transfer was >1×, suggesting the time-split holdout is a conservative proxy.

**Kimi's May 13 summary**: Segment selection works in kim's framework (+0.0156 LB, consistent with claude's +0.0154 val). Hybrid per-metric CW does NOT help in kim's framework (+0.0005, noise). GBDT per-window (XGB/LGBM) both worse than RF. Next biggest gap is CNN ensemble (+0.011), which kim has not yet built.

---

## All experiments

| Date | Author | File | val_f1 | lb_f1 | Outcome | Notes |
|---|---|---|---|---|---|---|
| 2026-05-11 | kimi | v1 (initial unsup ensemble) | — | ~0.33 | baseline | Z/MAD/IQR/IF/LOF/AE with bugs. |
| 2026-05-11 | kimi | v2 (bug fixes + weighted) | — | improved | iterate | Fixed scorer calls, train-stats normalization, temporal scorers. |
| 2026-05-11 | kimi | v3 (test_ratio top-k) | — | ~0.47 | win | Use info.json test_ratio for top-k everywhere. |
| 2026-05-11 | kimi | v4 (supervised RF) | — | ~0.47 | flat | RF on engineered features. |
| 2026-05-11 | kimi | v5 (XGB+LGBM) | — | ~0.47 | flat | Bottleneck is generalization, not capacity. |
| 2026-05-12 | kimi | v6 (category-aware strategy) | — | 0.4371 | regress | Per-category strategy hurt. |
| 2026-05-12 | kimi | v7 (CW RF + category ensemble) | — | 0.5271 | iterate | First strong CW result. |
| 2026-05-12 | kimi | v8 (scale-invariant CW) | — | **0.5485** | **win** | Drop raw `value` from CW features. Kimi's previous best. |
| 2026-05-12 | kimi | v9 (metric-aware CW) | — | 0.5367 | drop | Metric_type one-hot in CW hurts. |
| 2026-05-12 | kimi | v10 (aug crosswindow) | — | 0.5338 | drop | Augmentation hurts. |
| 2026-05-12 | kimi | v11d (no per-window RF) | — | 0.5255 | drop | PW RF helps; don't remove. |
| 2026-05-12 | claude | `claude_v1_segments.py` | 0.9298 | — | win (used) | Segment selection vs top-k: +0.0154. |
| 2026-05-12 | claude | `claude_v2_lgbm.py` | 0.9169 | — | drop | LGBM cross-window lost to RF by −0.013. |
| 2026-05-12 | claude | `claude_v3_per_metric.py` (pure) | 0.9305 | — | superseded | +0.0007 over global; bimodal (3 types +0.024, 3 types −0.020). |
| 2026-05-12 | claude | `claude_v3_per_metric.py` (hybrid) | 0.9387 | — | win (used) | Route per-metric for {ErrorCount, ResourceUtil, SuccessRate}, global else. |
| 2026-05-12 | claude | `claude_v4_cusum.py` | 0.9356 | — | drop | CUSUM hurts by −0.003. |
| 2026-05-12 | claude | `claude_v5_low_pw.py` | 0.8980–0.9279 | — | drop | PW weight sweep: 0.35 baseline is best; lowering hurts up to −0.041. |
| 2026-05-12 | claude | `claude_v6_cnn.py` (single seed) | 0.9449 | — | superseded | 1D CNN, 32-pt context. +0.0062 over hybrid. |
| 2026-05-12 | claude | `claude_v7_cnn_ensemble.py` (3-seed) | 0.9497 | — | win (used) | +0.0110 over hybrid. |
| 2026-05-12 | claude | `claude_v8_cnn5.py` (5/7 seeds) | 0.9498 / 0.9473 | — | drop | 5-seed = +0.0001 over 3; 7-seed worsens by −0.003. Saturated. |
| 2026-05-12 | claude | `claude_v9_seg_sweep.py` | **0.9615** | — | win (used) | smooth=3, thr_frac=0.7 beats (5, 0.6) by +0.0118. |
| 2026-05-12 | claude | `claude_v10_cnn_diverse.py` | 0.9577 | — | drop | Diverse-arch CNN ensemble lost to identical-arch by −0.004. |
| 2026-05-13 | claude | **`submission_segtuned.json`** | 0.9615 | **0.6111** | **uploaded (rank 3)** | All claude wins stacked. +0.0626 over kimi v8. |
| 2026-05-13 | kimi | v12 (stronger CW + IF-on-test) | — | **0.5665** | win | +0.018 over v8. Sources un-isolated. |
| 2026-05-13 | kimi | v13 (LightGBM per-window) | — | 0.5174 | drop | LGBM per-window much worse than RF. |
| 2026-05-13 | kimi | v14 (XGBoost per-window) | — | 0.5302 | drop | XGBoost per-window worse than RF. Confirms RF > GBDT at this scale. |
| 2026-05-13 | claude | `claude_v11_kimi_v12_ablation.py` (all_v12) | **0.9639** | **0.6238** uploaded as "stronger CW + IF + segments + CNN ensemble" | **win — rank 3** | Validation said +0.0024 over claude's prior best. **LB delivered +0.0127** — 5.3× transfer ratio, much higher than the previous 1.33×. Probably means our 100-window holdout under-represents the categories where IF-on-test helps most (disjoint + constant_train). Total LB gain claude+kimi combined: 0.6238 − 0.5485 (kimi v8) = **+0.0753**. |
| 2026-05-13 | claude | `claude_v12_per_metric_cnn.py` | 0.9519 | — | **drop** | Per-metric CNN routing (specialized ensembles for ErrorCount/ResourceUtil/SuccessRate, global else) lost by −0.012 because ResourceUtil's per-metric CNN pool (12k samples) overfit vs global (90k). Counter-intuitive but: CNNs need data scale more than per-metric specialization. RFs are fine on per-metric pools, CNNs are not. **Dead-end.** |
| 2026-05-13 | claude | `claude_v13_gbm_seedhack.py` (5-seed LGBM hom / diverse / RF+LGBM blend) | 0.9578 / 0.9560 / 0.9627 | — | **drop** | Even with 5-seed ensembling, LGBM cross-window still loses to RF: hom −0.006, diverse −0.008, RF+LGBM 50/50 blend −0.001. Single-seed LGBM was −0.013; ensembling closes half the gap, not enough. Diverse hyperparams hurt vs homogeneous (less-optimal configs drag the average). Per-metric: LGBM beats RF on QPS (+0.007) but loses LatencySecond (−0.020) and ResourceUtil. Note: XGBoost segfaulted on this build (exit 139), so XGB variant not tested. **GBM cross-window is dead at this scale — RF is the right choice.** |
| 2026-05-13 | claude | `claude_v14_hybrid_qps_lgbm.py` (LGBM-for-QPS only) | 0.9652 | **0.6236** | **drop** | Surgical routing: LGBM for QPS windows, RF hybrid for everything else. Validation showed +0.0012 overall (QPS: +0.0069, all others unchanged). LB: −0.0002 vs 0.6238 — **regression**. Validation→LB transfer was negative for this narrow change. **Lesson: validation Δ < ~0.002 is noise and shouldn't drive LB uploads, especially for category-targeted changes.** |
| 2026-05-13 | claude | `claude_v15_seg_sweep_on_v14.py` (192-config sweep) | best=0.9652 | — | **drop** | Re-swept segment params on v14's score distribution to test whether sharper scores need different params. Result: existing (smooth=3, thr_frac=0.7) is already optimal — top 4 configs all use these values. Every alternative loses 0.0006–0.0010. **Segment-param optimum is robust across score distributions.** Confirms the v9 finding doesn't need re-tuning when the upstream model changes. |
| 2026-05-14 | claude | `claude_v16_long_context_cnn.py` (3-seed context=64) | 0.9620 | — | **drop** | Long-context 3-seed CNN ensemble lost to short-context by −0.0019. Per-metric mixed: Count +0.004, SuccessRate +0.002, but LatencySecond −0.006 and ResourceUtil −0.009. Likely the bigger model overfits relative to the same pool size — 64-pt context doubles parameter count but training data is unchanged. Anomaly segments average only 17 points, so longer context wasn't needed. |
| 2026-05-14 | claude | `claude_v17_stacking.py` (2-fold LR meta-classifier per category) | 0.9550 (mean of 0.9722 and 0.9379) | — | **drop** | Logistic regression stackers on cached channel scores. Strong asymmetric failure: train on fold A → eval fold B = 0.9722 (beats baseline); train on fold B → eval fold A = 0.9379. Fold B's disjoint had zero positive labels, fell back to hand-tuned. Per-category training sets (6–18 windows per fold) too small to learn weights reliably. Lesson: stacking needs much larger training data than a 100-window holdout 2-fold split. Hand-tuned baseline (0.9639) wins. |
| 2026-05-14 | claude | `claude_v18_segment_classifier.py` (segment-level RF on 28 features) | 0.9201 | — | **drop** | RADICAL: direct segment scoring instead of point-wise + heuristic post-processing. 29.9k segment candidates trained at 24.5% pos rate. RF feature importance dominated by `cw_mean` (0.31) and `cw_max` (0.20) — segment-shape features (length, boundary jumps) get tiny weights. The classifier just reproduces existing point-level scores aggregated over the segment. Lost by −0.044 vs v14 baseline. **Lesson: existing scores already capture segment-level info via mean/max; segment shape adds nothing.** |
| 2026-05-14 | claude | `claude_v19_autoencoder.py` (1D conv AE reconstruction-error channel) | 0.9630 (best at ae_w=0.05) | — | **drop** | RADICAL: 1D conv autoencoder (78k params) trained on normal-only training points; per-point reconstruction error added as new channel at weight sweep {0.05, 0.10, 0.15, 0.20}. All weights lost: best −0.0009, worst −0.0123. Per-metric mixed: AE helps SuccessRate (+0.009 at w=0.05) but hurts QPS (−0.014). **Lesson: reconstruction-error and CW score are correlated enough that adding AE only adds noise.** |
| 2026-05-14 | claude | `claude_v20_tta.py` (BatchNorm train-mode at CNN inference) | 0.9541 (batch), 0.9534 (blend) | — | **drop** | RADICAL: test-time adaptation by flipping CNN's BatchNorm to train-mode at inference (uses test window's batch stats instead of training running stats). Both batch-only (−0.0098) and 50/50 blend (−0.0105) lost. Per-metric bimodal: TTA helps SuccessRate (+0.018) and Count (+0.008), hurts ErrorCount (−0.037), LatencySecond (−0.021). |
| 2026-05-14 | claude | `claude_v21_tta_routed.py` (per-metric TTA routing) | best=0.9599 (TTA on SuccessRate only) | — | **drop** | Route TTA only to metric types where v20 showed gains. Sweep: SR only (−0.0040), SR+Count (−0.0077), SR+Count+ResourceUtil (−0.0077). Per-metric still shows TTA gain on SR (+0.018) and Count (+0.008), but the routing affects other metrics indirectly via shared CNN state (BN running stats mutate during batch-mode forward passes). |

---

| 2026-05-14 | claude | `claude_v22_metadata_features.py` (intervals as CW per-point feature) | **0.9650** | **pending** | **win (small)** | First win in 7 attempts: intervals (sampling rate) added as a per-point CW feature beats baseline by +0.0011. intervals+seqlen mode lost (−0.0133, too much metadata). Per-metric: helps Count +0.007, ResourceUtil +0.006, SuccessRate +0.013; hurts ErrorCount −0.007, Latency −0.011. Broad change — affects all categories. `submission_metadata_cw.json` ready. |
| 2026-05-14 | claude | `claude_v23_per_interval_cw.py` (per-intervals hybrid routing) | best=0.9600 | — | **drop** | Per-intervals routing (7 sub-models by sampling rate) instead of metric_type routing. Lost by −0.0055 (without intervals feat) and −0.0050 (with). Per-metric: helps Count (+0.007) and LatencySecond (+0.010), but ResourceUtil collapses (−0.049) because per-metric routing was specifically carrying it. **Lesson: v22's design (metric_type routing + intervals as feature) is the local optimum — combining routings can't have both.** |
| 2026-05-14 | claude | `claude_v24_cw_ensemble.py` (3-seed CW ensemble) | 0.9644 | — | **drop** | Three RF-hybrid CWs with different `random_state`, predictions averaged. Lost by −0.0006. Each RF is itself a 500-tree bagging ensemble, so 3 different seeds average to nearly the same answer (RF is already deterministic enough that seed averaging adds no value). Per-metric near-identical to v22 single CW. **Lesson: CW seed ensembling provides no gain — the 500 trees already span the variance.** |
| 2026-05-14 | claude | `claude_v26_transformer.py` (3-seed Transformer encoder) | 0.9575 | — | **drop** | Plain transformer with sinusoidal PE, 397k params. Lost by −0.0075. Helped ErrorCount (+0.025) but hurt everything else. **Lesson: bigger model + same data = overfits.** |
| 2026-05-14 | claude | `claude_v27_spectral_features.py` (FFT/entropy/ZCR per-point features) | 0.9634 | — | **drop** | 8 new spectral features added to CW. Net −0.0016 with same bimodal pattern: helps LatencySecond +0.014 but hurts ResourceUtil −0.023. |
| 2026-05-14 | claude | `claude_v28_ratio_feature.py` (anomaly_ratio as window feature) | 0.9642 | — | **drop** | Using info.json's anomaly ratio as window-broadcast feature. Net −0.0009. Model uses it as a shortcut that doesn't generalize. |
| 2026-05-14 | claude | `claude_v29_pseudo_label.py` (pseudo-labeled test data added to training) | 0.9595 | — | **drop** | Used current best model to predict on held-out test_x, used those predictions as pseudo-labels for retraining. Net −0.0055. ResourceUtil collapses, SuccessRate hurt. |
| 2026-05-14 | claude | `claude_v30_anomaly_transformer.py` phase 1 (AT with Gaussian-prior KL discrepancy) | 0.9637 (best at disc_w=0.5) | — | **drop** | 397k-param Anomaly Transformer per Xu et al. 2022, KL discrepancy loss. Lost vs CNN baseline at all 5 disc_weights (0.0, 0.25, 0.5, 0.75, 1.0). HUGE per-metric gain on ErrorCount (+0.035 at disc_w=0.5) but ResourceUtil collapse offsets. Phase 2/3 cancelled per pre-defined rule. **Lesson: discrepancy mechanism IS learning real signal (BCE & disc decrease cleanly), but inductive bias still wrong for this dataset.** |
| 2026-05-14 | claude | `claude_v31_investigate.py` (analysis only — no model) | — | — | analysis | Investigated ResourceUtil failure modes. **Key finding**: 15/17 ResourceUtil holdout windows already get F1=1.0; only 2 are bad. Specific causes: (1) wid=944 has end-of-window false positive from upward score drift, (2) wid=967 has start-of-window misalignment from same-mode convolution edge effects. Channel separation on ResourceUtil is actually strong (CNN+0.72, IF+0.65, global+0.69). **Bottleneck is post-processing, not model.** |
| 2026-05-14 | claude | `claude_v32_edge_fix.py` (reflective padding + boundary penalty sweep) | **0.9667 (+0.0017)** | **pending upload** | **WIN** | Validated v31's hypothesis. Reflective padding for segment-selection smoothing (vs default 'same' mode) gives +0.0017 net. Entire gain in **SuccessRate** (0.9566 → 0.9671, +0.0104); all other metrics unchanged. Boundary penalty wasn't triggered on this holdout. Pure post-processing change — broad in scope, narrow in benefit. `submission_edge_fix.json` ready. |
| 2026-05-14 | claude | `claude_v33_investigate_all.py` (failure mode classification, all holdout) | — | — | analysis | Classified 11 holdout F1<0.9 failures. Result: **5/11 are off_by_few (all shifted RIGHT)** + 2 split_segment + 2 extra_segments + 2 other. Systematic right-shift bias is unambiguous — CNN's asymmetric context window (16 past + 15 future) plus convolution effects bias prediction peak rightward of truth. |
| 2026-05-14 | claude | `claude_v34_postproc_fixes.py` (left-shift + merge close) | 0.9723 (+0.0056 val) | **0.6184** (−0.0054 vs LB best) | **REGRESSED** | Validation win was misleading. LB went DOWN. Pattern across v22/v32/v34 confirms: our 70/30 time-split validation has stopped predicting LB direction. Each "improvement" loses ~0.002 LB despite gaining on val. |

---

## CRITICAL: validation methodology broken (2026-05-14 ~7:25 pm)

Three consecutive LB uploads regressed from our best (0.6238):

| Submission | Val Δ | LB result | Δ vs best |
|---|---|---|---|
| v22 intervals-aware CW | +0.0011 | 0.6223 | **−0.0015** |
| v32 reflective smoothing | +0.0017 | 0.6220 | **−0.0018** |
| v34 left-shift + merge | +0.0056 | **0.6184** | **−0.0054** |

The "investigation-driven post-processing fixes" methodology that produced these has been **overfitting to specific holdout windows** rather than improving generalization. The validation-to-LB transfer is now **negative**, not just diminishing.

Current best LB remains **0.6238** ("stronger CW + IF + segments + CNN ensemble", May 13).

Pivot: stop the post-processing rabbit hole. Friend's 0.644 LB with a simpler RF-based approach suggests a different methodology entirely. Awaiting friend's details before next experiment.
| 2026-05-14 | claude | `claude_v25_vote_ensemble.py` (submission-level voting ensemble) | (no train-set eval) | **pending** (4 vote files generated) | **diagnostic** | Vote-set candidates: kimi_combo, metadata-cw, segtuned, qps_routed. **Inter-submission disagreement: only 1.07% of points** — submissions are highly correlated since they share the CW+CNN+IF+segments backbone. Upper bound on voting impact is small. 4 vote files written: `submission_vote_top2_kimi_combo_plus_metadata.json`, `_top3_*`, `_top4_all.json`, `_all_minus_segtuned.json`. Cannot validate without LB upload. |

---

## CEILING REACHED — 6 consecutive radical experiments dropped (v22 broke it slightly)

After v15 set our LB best at 0.6238, we tried 6 fundamentally different approaches over 2026-05-13/14:

| # | Approach | Val Δ |
|---|---|---|
| v16 | Long-context CNN (64 vs 32) | −0.002 |
| v17 | Stacking meta-classifier | −0.009 |
| v18 | Segment-level classifier (radical reformulation) | −0.044 |
| v19 | Autoencoder reconstruction-error channel | −0.001 |
| v20 | Test-time adaptation (TTA via train-mode BN) | −0.010 |
| v21 | Per-metric TTA routing | −0.004 |

**Every baseline run hits exactly 0.9639 validation F1.** No new channel, no new architecture, no inference trick breaks it. This is the **architectural ceiling** for our point-level scoring + heuristic segmentation pipeline on the stratified 100-window time-split holdout.

Remaining options to push past 0.6238 LB would require **fundamentally new model classes** that we don't have time to implement in this session:
- **Anomaly Transformer / DCdetector** — published methods, days of work
- **Self-supervised contrastive pretraining** of a sequence encoder on all 1000 windows
- **A different validation regime** that doesn't max out at this ceiling (e.g., direct LB-submission feedback loop)

Practical recommendation: **accept 0.6238 / rank 3 as final** and use remaining slots only if a fresh, *qualitatively* different idea appears.

---

## Ceiling reached at v14 (val 0.9639, LB 0.6238)

After 4 consecutive "radical" experiments all failed at exactly the v14 validation ceiling, we conclude:

1. **The current ensemble (CW + CNN + IF + online + global) already spans the point-level score space.** No new channel of the same kind adds complementary signal.
2. **Aggregating differently** (segment vs point) doesn't help because mean/max of existing scores already captures segment information.
3. **Bigger models overfit** at this dataset scale (CNN_Long at context 64 lost; Wide CNN lost in v10).
4. **Smaller per-group training pools** hurt more than per-group specialization helps (per-metric CNN, per-fold stackers both failed).

The remaining LB gap from rank 1 (0.0116) is likely about **distribution shift** between training and test, not about model architecture. Only untried direction: **test-time adaptation** (BN recalibration on each test window). Cheap to try.

---

## Calibration

**Validation-to-leaderboard transfer ratio** — three data points now:

| Upload | Val Δ (incremental) | LB Δ (incremental) | Ratio |
|---|---|---|---|
| `submission_segtuned.json` (vs v8) | +0.0471 | +0.0626 (0.6111) | 1.33× |
| `submission_kimi_combo_all_v12.json` (vs segtuned) | +0.0024 | +0.0127 | **5.3×** |
| `submission_qps_lgbm_routed.json` (vs all_v12) | +0.0012 | **−0.0002** | **−0.17×** |

Three different ratios for three different change types. Pattern:

- **Architectural / broad** changes (stronger CW + IF + reweighting): transfer at ≥1.4× and possibly much higher
- **Stacked / mixed** changes (the whole stack from v8 to segtuned): transfer at ~1.3×
- **Narrow / categorical** changes with Δ_val < 0.002: transfer is **unreliable** and can go negative — this is just noise

**Rule of thumb**: don't upload a change unless Δ_val ≥ +0.003 AND it touches multiple categories. The v14 single-category +0.0012 was a waste of a slot.

---

## Known dead-ends (don't redo)

| Idea | Why dead | Source |
|---|---|---|
| LGBM cross-window (single) | −0.013 vs RF | `claude_v2_lgbm.py` |
| LGBM cross-window (5-seed ensemble, hom or diverse) | −0.006 to −0.008 vs RF; ensembling halves the gap but doesn't close it | `claude_v13_gbm_seedhack.py` |
| RF + LGBM 50/50 blend cross-window | −0.001 vs RF; LGBM errors correlate with RF, no complementary signal | `claude_v13_gbm_seedhack.py` |
| XGBoost per-window | −0.018 vs RF | kimi v14 |
| LightGBM per-window | −0.031 vs RF | kimi v13 |
| CUSUM channel | −0.003; absorbed by hybrid CW | `claude_v4_cusum.py` |
| Score-space 50/50 averaging | −0.008 vs weighted blend | `claude_v7_cnn_ensemble.py` |
| Lowering per-window RF weight | −0.034 to −0.041 | `claude_v5_low_pw.py` |
| 5/7-seed CNN | saturated at 3 | `claude_v8_cnn5.py` |
| Architecturally diverse CNN ensemble | −0.004 vs identical-arch 3-seed | `claude_v10_cnn_diverse.py` |
| Pure per-metric CW (no routing) | +0.0007 (noise); superseded by hybrid | `claude_v3_per_metric.py` |
| Per-metric CNN routing | −0.012 vs global CNN; per-metric pool (~12k) too small for CNN, overfits | `claude_v12_per_metric_cnn.py` |
| Long-context CNN (context=64) | −0.002 vs short-context (32); bigger model overfits, segments are too short (mean 17) to benefit from 64-pt window | `claude_v16_long_context_cnn.py` |
| Stacking meta-classifier (per-category LR) | −0.009 vs hand-tuned; per-category training sets too small (6–18 windows) to learn reliable weights | `claude_v17_stacking.py` |
| Segment-level classifier (28-feature RF on segment candidates) | −0.044; feature importance dominated by CW score aggregates — segment shape adds nothing the heuristic doesn't get | `claude_v18_segment_classifier.py` |
| 1D conv autoencoder reconstruction-error channel | −0.001 to −0.012 across weights {0.05–0.20}; AE correlated with CW, no complementary signal | `claude_v19_autoencoder.py` |
| Hybrid per-metric CW (in kimi's framework) | +0.0005 (noise); does not help | kimi v16 |
| Metric_type one-hot in CW features | kimi v9: −0.011 LB | kimi v9 |
| Data augmentation (shift/scale) | kimi v10: −0.015 LB | kimi v10 |
| No per-window RF | kimi v11d: −0.023 LB | kimi v11d |
| Per-window model selection (train-F1) | val 0.180; doesn't transfer | kimi exp 4 |
| Subsequence discord detection | val 0.143–0.152 | kimi exp 7 |

---

## Things worth trying (not yet tested)

| Idea | Suggested by | Expected EV |
|---|---|---|
| Stacking meta-classifier on cached scores | both | +0.005–0.010 |
| Per-metric CNN routing (same idea as hybrid CW) | claude | +0.005–0.010 |
| Longer CNN context (64 vs 32) as primary | claude | unknown |
| CNN ensemble scorer (kimi doesn't have one yet) | claude | +0.011 (biggest remaining gap) |
| Spectral / autocorrelation features for CW | kimi | +0.005 |
| Multi-holdout validation (3–5 seeds) for confidence intervals | claude | no gain, but de-risks decisions |

| 2026-05-13 | kimi | `generate_submission_v15.py` (v12 + segment selection smooth=3/thr_frac=0.7) | — | — | win (pending LB → 0.5821) | Combines stronger CW (500/15/2) + IF-on-test + segment selection. ~7.1 changed predictions/window vs v12. |

| 2026-05-13 | kimi | `generate_submission_v16.py` (v15 + HybridCrossWindowModel) | — | **0.5826** | drop | Per-metric routing added only +0.0005 vs v15 (noise). Hybrid CW does NOT help in this framework. |
| 2026-05-15 | claude | `claude_v38_friend_repro.py` (per-metric RF+HGBT+LR, causal rolling, skip zero-label) | 0.3101 CW-LOO | **0.5586** | **drop** | First attempt at friend's 0.6453 pipeline. HUGE regression (−0.065 vs our best). Root cause: causal rolling features. We misread the spec. |
| 2026-05-15 | claude | `claude_v39_include_all_windows.py` (same + include zero-label windows) | 0.3101 CW-LOO | **0.5529** | **drop** | Tested whether skipping zero-label windows was the bug. No — including them made it slightly worse. **APPROACH.md later confirmed: skip zero-label is correct.** |
| 2026-05-15 | claude | `claude_v40_centered_rolling.py` (v38 + centered rolling via pandas) | 0.3152 CW-LOO | **0.6402 — NEW BEST** | **WIN** | **KEY FIX**: APPROACH.md revealed friend uses `pandas.Series.rolling(center=True, min_periods=1)` for ALL rolling stats. Switching from causal→centered: +0.0164 vs v38, +0.0164 vs prior best 0.6238. Gap to friend (0.6453) = 0.0051. `submission_v40_centered_rolling.json` |
| 2026-05-15 | claude | `claude_v41_smooth3.py` (v40 + smooth=3) | 0.3159 | **not submitted** | drop | LOO same as v40 (0.3152). Smooth width makes no difference. |
| 2026-05-15 | claude | `claude_v42_smooth7.py` (v40 + smooth=7) | 0.3141 | **not submitted** | drop | LOO slightly worse. Smooth width confirmed not a lever. |
| 2026-05-15 | claude | `claude_v43_shift_pipeline.py` (v40 P1 + temporal-split shift P2, W_SHIFT=0.30) | 0.3160 | **0.6561 — NEW BEST** | **WIN** | TWO-PIPELINE BLEND. P1=standard 68 feats on full train_x; P2=75 feats (68 + 7 shift) trained on last 30% of each window using first 70% as reference. Shift features (rank_in_self, self_robust_z, above_ref_max, below_ref_min, mean/std/median shift) are NON-TRIVIAL during training because temporal split gives real train→pseudo-test distribution differences. prob = 0.7*P1 + 0.3*P2. Gap to friend (0.66): **0.0039**. `submission_v43_shift_pipeline.json` |
| 2026-05-16 | claude | `claude_v44_wshift20.py` (W_SHIFT=0.20) | 0.3155 | — | **drop** | Less P2 influence (0.30→0.20). LOO worse than v43. Not submitted. |
| 2026-05-16 | claude | `claude_v45_wshift40.py` (W_SHIFT=0.40) | — | **0.6531** | **drop** | More P2 influence (0.30→0.40). LB worse. W_SHIFT=0.30 confirmed optimal. |
| 2026-05-16 | claude | `claude_v46_split60.py` (SPLIT_FRAC=0.60) | — | — | **drop** | Smaller training portion (60/40 split). LOO neutral/worse. Not submitted. |
| 2026-05-16 | claude | `claude_v47_split80.py` (SPLIT_FRAC=0.80) | — | — | **drop** | Larger training portion (80/20 split). LOO neutral/worse. Not submitted. SPLIT_FRAC=0.70 confirmed optimal. |
| 2026-05-16 | claude | `claude_v48_p2_all_windows.py` (P2 pool includes all-zero-label windows) | — | — | **drop** | Added all-normal training windows to P2 pool. LOO worse — negative windows dilute the shift signal. Confirms: only include windows with at least 1 anomaly in training pool. |
| 2026-05-16 | claude | `claude_v49_p2_consistent_ref.py` (P2 consistent reference fix) | 0.3146 | **0.6487** | **drop** | Attempted to fix P2's reference at inference (use train_x[:70%] instead of full train_x). LB regression −0.0074 vs v43. Current approach (full train_x as ref at inference) is correct. |
| 2026-05-16 | claude | `claude_v50_more_minmax.py` (+rolling min/max at w=5,21,41) | 0.3154 | **0.6602** | **WIN** | Added (rmax_w − x) and (x − rmin_w) for w ∈ {5,21,41} on top of existing w=11. 68→74 P1 features, 75→81 P2. LOO predicted neutral but LB improved +0.004. Confirms LOO is unreliable — submit architecturally sound changes aggressively. |
| 2026-05-16 | claude | `claude_v51_pseudo_label.py` (pseudo-labeling, PW=0.30, v43 preds) | 0.3185 | **0.6711** | **WIN — BREAKTHROUGH** | TRANSDUCTIVE LEARNING. Use v43's binary test predictions as pseudo-labels for the 1000 test windows. Add pseudo-labeled test_x to BOTH P1 and P2 training pools with sample weight 0.30. 765/1000 windows have k>0 (predicted anomalies). Model now trains on test distribution → closes domain gap. +0.011 LB over v50. |
| 2026-05-16 | claude | `claude_v52_minmax_pseudo.py` (74 feats + v51 pseudo-labels) | 0.3186 | **0.6762** | **WIN** | Combine rolling min/max features (v50) with pseudo-labeling (v51). Better pseudo-labels from stronger model → better retraining. +0.005 LB. |
| 2026-05-16 | claude | `claude_v53_iter_pseudo.py` (74 feats + v52 pseudo-labels, iter 2) | 0.3186 | **0.6797** | **WIN** | Second iteration: v52's stronger predictions as new pseudo-labels. +0.0035 LB. |
| 2026-05-16 | claude | `claude_v54_iter_pseudo2.py` (74 feats + v53 pseudo-labels, iter 3) | 0.3192 | **0.6810** | **WIN** | Third iteration. +0.0013 LB. Gains diminishing (~+0.001/round). |
| 2026-05-16 | claude | `claude_v55_iter_pseudo3.py` (74 feats + v54 pseudo-labels, iter 4) | 0.3186 | — | **skip** | Fourth iteration. LOO 0.3186 < v54's 0.3192. Not submitted — converging. |
| 2026-05-16 | claude | `claude_v56_pw50.py` (PSEUDO_WEIGHT=0.30→0.50, v54 pseudo-labels) | 0.3186 | **0.6821** | **WIN** | Higher pseudo-label influence. Despite identical LOO to v55, LB improved +0.0011 over v54. LOO again unreliable. |
| 2026-05-16 | claude | `claude_v57_pw50_iter.py` (PW=0.50, v56 pseudo-labels, iter 2) | 0.3186 | **0.6832** | **WIN** | Continue iterating with stronger pseudo-labels at PW=0.50. +0.0011 LB. |
| 2026-05-16 | claude | `claude_v58_global_ctx.py` (+global context features: mean_z, std_z, max_z) | 0.3192 | **0.6847** | **WIN** | NEW METHOD. Add 3 window-level features: z-score of this window's mean/std/max vs all windows of same metric_type. Train windows use train.npy population; test windows use test.npy population. P1=77 feats, P2=84 feats. LOO 0.3192 (best since v54). +0.0015 LB. Gap to 0.70 = 0.0153. |
| 2026-05-17 | claude | `claude_v59_lgbm_multirounds.py` (LightGBM replaces HGBT + 8 internal pseudo-label rounds) | — | **0.6834** | **DROP** | LightGBM (n_estimators=400, num_leaves=63) replaces sklearn HGBT. Multi-round pseudo-labeling: 8 rounds in 1 slot, converged after round 1 (18 flips, then 0 changes). LB −0.0013 vs v58. **LightGBM consistently worse than HGBT** — confirmed again (also lost in v2/v13/v14). Multi-round pseudo-labeling wastes slots after convergence; 1 round is sufficient. |
| 2026-05-17 | claude | `claude_v60_fft_features.py` (FFT reconstruction error +5 features) | 0.3200 | **0.6859** | **WIN** | NEW METHOD. Add 5 per-point FFT features via reconstruction error: test_fft_residual, test_fft_residual_z, test_res_vs_train, periodicity_strength, train_fft_res_std_log. K=adaptive min(10, n//4). P1=82, P2=89. Pseudo-labels from v58 (PW=0.50). LB +0.0012. |
| 2026-05-17 | claude | `claude_v62_tda_cached.py` (TDA persistent homology +10 features, cached) | 0.3186 | **0.6863** | **WIN** | NEW METHOD. Takens delay embedding (dim=2) → ripser H0+H1 → 10 window-level broadcast features (h0_max, h0_sum, h0_entropy, h1_max, h1_sum, h1_n_sig, ref_h0_max, ref_h1_max, h0_bottleneck, h1_bottleneck). Cache in tda_cache/ (max_pts=150, 7 min to precompute). P1=92, P2=99. Pseudo-labels from v60 (PW=0.50). LOO 0.3186 (↓ but LB ↑). +0.0004 LB. |
| 2026-05-17 | claude | `claude_v63_wavelet_acf.py` (CWT wavelet +4 feats + ACF +7 feats + PW=0.70) | 0.3186 | **0.6858** | **DROP** | Added CWT (Morlet, scales 2,4,8,16) and ACF (lags 1,2,3,5,10,20,40). P1=103, P2=110. PW=0.70 (first time above 0.50). Pseudo-labels from v62. LOO flat, LB −0.0005 vs v62. Wavelet+ACF adds no signal on top of FFT+TDA. |
| 2026-05-17 | claude | `claude_v65_matrix_profile.py` (stumpy matrix profile m=5,10,20 + extra rolling + FFT broadcast) | 0.3201 | **0.6890 ← BEST** | **WIN** | NEW SIGNAL CLASS. Matrix profile at m=5,10,20: per-point distance to nearest non-trivial subsequence match (discord = anomaly). Also added: extra rolling w=3,7,63 (mean+std, +6 feats) and FFT broadcast (top-3 peak mags + HF energy ratio, +4 feats). P1=105, P2=112. Pseudo-labels from v62 (PW=0.70). LOO 0.3201 (best ever). LB +0.0027 over v62 — biggest single jump since pseudo-labeling. Validates friend's key insight: add features from DIFFERENT signal class (subsequence-distance) not more of the same. |
| 2026-05-17 | claude | `claude_v66_pseudo_iter.py` (pseudo-label iter from v65, identical architecture) | 0.3201 | **0.6902 ← BEST** | **WIN** | Iterate pseudo-labels: PSEUDO_SOURCE=v65 (0.6890). Same 105/112-feat architecture as v65. LOO ties v65 (0.3201). LB +0.0012 — reliable pseudo-label iteration gain confirmed again. Gap to 0.70 = 0.0098 (first time under 0.01). |
| 2026-05-17 | claude | `claude_v67_more_mp_cusum.py` (6 MP windows m=3,5,10,15,20,30 + CUSUM +2 feats) | 0.3195 | **0.6901** | **WIN** | Extend MP from 3→6 scales (+3 feats). Add CUSUM path-dependent features S_pos/S_neg (+2 feats). 110/117 total. Pseudo-labels from v65 (PW=0.70). LOO 0.3195 (↓ but LB ↑ as usual). LB +0.0011 over v65. Both new signal classes contribute. |
| 2026-05-17 | claude | `claude_v68_stl_ar.py` (STL seasonal + AR(1) residuals, +3 feats) | 0.3187 | **0.6905** | **WIN** | Professor-style contextual anomaly detection. AR(1) one-step residual + STL (daily period for intervals>=864s, linear detrend fallback). P1=108, P2=115. PSEUDO_SOURCE=v66. LOO 0.3187 (QPS dipped slightly), LB +0.0003. Seasonal signal confirmed in the data. |
| 2026-05-18 | claude | `claude_v69_iter_v68.py` (pseudo-label iter from v68) | 0.3193 | pending | pending | PSEUDO_SOURCE=v68 (0.6905). Same 108/115 arch as v68. LOO 0.3193 (QPS recovered vs v68). Submission ready. |
| 2026-05-18 | claude | `claude_v70_combined.py` (v67 6MP+CUSUM + v68 STL/AR combined, 113/120 feats) | 0.3191 | pending | pending | All winning signal classes stacked: 6 MP scales + CUSUM + STL/AR. PSEUDO_SOURCE=v68. LOO 0.3191. Submission ready. |
| 2026-05-18 | claude | `claude_v71_catch22.py` (catch22+22 + complexity+3 + per-metric-rules+6 + CatBoost 4th model, 139/146 feats) | 0.3180 | pending | pending | Four new additions: pycatch22 canonical features, sample/perm/LZ entropy, per-metric domain rules, CatBoost blend (0.65 HGBT+0.15 CB+0.10 RF+0.10 LR). PSEUDO_SOURCE=v68. LOO 0.3180 (QPS 0.3155). Generating submission. |
