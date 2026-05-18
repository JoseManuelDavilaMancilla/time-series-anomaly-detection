# Notes for kimi — current state as of 2026-05-17 end of day

**Read this first. Everything below is what you need to pick up the work.**

---

## Bottom line

- **Our best LB**: 0.6890 (v65, `submission_v65_matrix_profile.json`)
- **Friend's best**: ~0.69. We are neck-and-neck (+0.0000 gap).
- **Gap to rank 1 (0.7142)**: 0.0252
- **Gap to 0.70**: 0.0110
- **~5 submission slots remaining today.**

---

## The entire old pipeline was abandoned

The old approach (CW + CNN + segments + morphology, claude v1–v34 + kimi v1–v16) hit a ceiling at 0.6238 on 2026-05-14 and validation became anti-correlated with LB (three consecutive "improvements" on validation all regressed on LB). That direction is dead.

On 2026-05-15 we obtained the friend's `APPROACH.md` (already in the repo). Their key insight: **centered rolling** and a **global per-metric-type model**. Reading their code unlocked a completely different approach.

**Do not continue any work on the old pipeline.** All winning work is in `claude_v40`–`claude_v58`.

---

## The new architecture (v40–v58)

### Core idea

Train one ensemble per metric type (`Count`, `ErrorCount`, `LatencySecond`, `QPS`, `ResourceUtilizationRate`, `SuccessRate`) on **all training windows** of that type. Score each test point individually, smooth, then select exact top-k.

No segments, no candidates, no morphology, no fallback.

### Two pipelines blended

**P1** — Standard pipeline:
- Features computed with `train_x` as reference
- Trained on all labeled training windows (those with at least one anomaly)

**P2** — Shift pipeline:
- Each training window split 70/30 chronologically
- Last 30% is treated as "pseudo-test" relative to first 70% as reference
- 7 extra shift features added (see feature list below)
- Captures genuine distribution shift signal because the temporal split creates real train→test differences within training data
- Trained on the 30%-splits (only windows where the split portion has at least one anomaly)

**Final score**: `0.70 × P1 + 0.30 × P2`

### Ensemble per pipeline

Per metric type, per pipeline: **3 RF + 5 HGBT + 1 LR**

Blend: `0.80 × HGBT_avg + 0.10 × RF_avg + 0.10 × LR`

LR gets a StandardScaler; RF and HGBT take raw features.

### Inference

```
1. If budget == 0: all zeros, done.
2. Compute P1 features (77) → P1 score
3. Compute P2 features (84) → P2 score
4. Blend: prob = 0.70 × P1 + 0.30 × P2
5. Smooth: rm = centered_rolling_mean(prob, w=5)
           prob_final = 0.20 × prob + 0.80 × rm
6. Top-k: select indices of highest prob_final, exactly budget of them
```

No post-processing. No gap filling.

---

## Feature list (v58 — current best)

### Critical: all rolling stats use `pandas.Series.rolling(center=True, min_periods=1)`

This alone is worth ~0.08 LB. Causal rolling → 0.55. Centered → 0.64+.

### P1 features (77 total)

| Feature | Detail |
|---|---|
| Raw value | `x` |
| Global robust z-score | `(x − median(train_x)) / (1.4826 × MAD(train_x))` |
| Global z-score | `(x − mean(train_x)) / std(train_x)` |
| First diff | `diff(x, 1)` |
| Second diff | `diff(x, 2)` |
| Rolling mean, std | w = 5, 11, 21, 41 (all centered) |
| Rolling median, MAD | w = 11, 41 (centered) |
| Residual from rolling median w=11 | `x − rmed11` |
| Robust z vs rolling median w=11 | `(x − rmed11) / (1.4826 × rmad11)` |
| EWMA (α=0.3) | causal |
| EWMA residual | `x − ewma` |
| Normalized EWMA residual | ÷ std(residual) |
| Percentile rank vs train_x | fraction of train points below x[i] |
| Position | `i / (n−1)` |
| Time of day | normalized from UTC timestamp |
| Day of week | normalized from UTC timestamp |
| Rolling mean w=41, std w=41 | centered |
| Rolling median w=41, MAD w=41 | centered |
| Local volatility ratio | `rolling_std(5) / rolling_std(41)` |
| Distance to rolling max/min | `rmax − x` and `x − rmin` at w = 5, 11, 21, 41 |
| **Global context: mean_z** | z-score of `mean(x)` vs all windows of same metric_type |
| **Global context: std_z** | same for `std(x)` |
| **Global context: max_z** | same for `max(x)` |
| Metric type one-hot | 6 features |
| Service one-hot | top-30 most frequent services, 30 features |
| Interval | `info["intervals"] / 3600` |
| Training anomaly ratio | from info.json |
| Test anomaly ratio | from info.json |

### P2 additional shift features (7, appended to P1 base)

Computed relative to `ref_x = train_x[:70%]`:

| Feature | Detail |
|---|---|
| Rank in self | percentile of x within x |
| Self robust z | `(x − median(x)) / (1.4826 × MAD(x))` |
| Above ref max | `max(0, x − max(ref_x))` |
| Below ref min | `max(0, min(ref_x) − x)` |
| Mean shift | `mean(x) − mean(ref_x)`, broadcast |
| Std ratio | `std(x) / std(ref_x)`, broadcast |
| Median shift | `median(x) − median(ref_x)`, broadcast |

---

## Pseudo-labeling (the biggest single discovery — +0.011 LB)

After fitting the model on training data, generate predictions on all 1000 test windows. Those binary predictions become **pseudo-labels**. Add each test window back into BOTH training pools (P1 and P2) with those pseudo-labels, at sample weight `PSEUDO_WEIGHT = 0.50`. Retrain from scratch and generate a new submission.

Key rules:
- Only test windows where `pseudo_y.sum() > 0` are included (765/1000)
- For P1: ref = `train_x`, series = `test_x`
- For P2: ref = `train_x[:70%]`, series = `test_x` (matches inference exactly)
- True labeled points: weight = 1.0. Pseudo-labeled points: weight = 0.50.

Each iteration uses the previous submission as pseudo-labels. Gains per round:

| Round | Version | LB | Δ |
|---|---|---|---|
| 0 (v43→v51) | pseudo-label with v43 preds | 0.6711 | +0.011 |
| 1 (v51→v52) | 74 feats + v51 preds | 0.6762 | +0.005 |
| 2 (v52→v53) | v52 preds | 0.6797 | +0.003 |
| 3 (v53→v54) | v53 preds | 0.6810 | +0.001 |
| 4 (v54→v56) | PW=0.50, v54 preds | 0.6821 | +0.001 |
| 5 (v56→v57) | PW=0.50, v56 preds | 0.6832 | +0.001 |
| 6 (v57→v58) | +global ctx features | 0.6847 | +0.0015 |

---

## Global context features (v58 — newest addition)

For each window, precompute z-scores of its summary statistics vs all windows of the same metric_type:
- `mean_z = (mean(x) − pop_mean_of_means) / pop_std_of_means`
- `std_z` = same for std
- `max_z` = same for max

At **training time**: compute from all training windows using `train.npy`. Pseudo-labeled test windows use the test-population stats.
At **test inference**: compute from all 1000 test windows using `test.npy`.

These 3 features are broadcast constant across all points in the window.

---

## Key constants

| Param | Value | Notes |
|---|---|---|
| W_SHIFT | 0.30 | P1/P2 blend. 0.20 and 0.40 both worse. |
| SPLIT_FRAC | 0.70 | Temporal split for P2. 0.60 and 0.80 both worse. |
| SMOOTH_W | 5 | Centered rolling mean window. |
| SMOOTH_ALPHA | 0.8 | `prob_final = 0.2*raw + 0.8*smoothed` |
| PSEUDO_WEIGHT | 0.50 | Sample weight for pseudo-labeled points. |
| TOP_K_SERVICES | 30 | Service one-hot breadth. |
| RF | n_estimators=200, max_depth=15, min_samples_leaf=10, balanced | |
| HGBT | max_iter=200, max_depth=8, lr=0.05, min_samples_leaf=20, balanced | |
| LR | C=0.5, balanced | |
| Ensemble blend | 80% HGBT + 10% RF + 10% LR | |

---

## Today's experiments (2026-05-17)

| Version | Change | LOO | LB | Δ | Verdict |
|---|---|---|---|---|---|
| v59 | LightGBM replaces HGBT + 8 pseudo-label rounds | — | 0.6834 | −0.0013 | **DROP** |
| v60 | +FFT reconstruction error features (+5 feats) | 0.3200 | 0.6859 | +0.0012 | **WIN** |
| v62 | +TDA persistent homology (+10 feats, cached) | 0.3186 | 0.6863 | +0.0004 | **WIN** |
| v63 | +Wavelet CWT + ACF + PW=0.70 | 0.3186 | 0.6858 | −0.0005 | **DROP** |
| v65 | +Matrix profile (stumpy, m=5,10,20) + extra rolling + FFT broadcast | 0.3201 | **0.6890** | **+0.0027** | **WIN — new best** |

**Key learnings from today:**
- **LightGBM < HGBT** — do not use LightGBM.
- **Multi-round pseudo-labeling converges in 1 round** — no point doing >1 round per slot.
- **Feature class matters more than feature count**: FFT (+0.0012), TDA (+0.0004), Wavelet+ACF (0), Matrix Profile (+0.0027).
- **Matrix Profile is the biggest new signal class** — subsequence distance (motif/discord) has fundamentally different inductive bias from all rolling stats. Friend's insight confirmed.
- **LOO finally correlated**: v65 LOO 0.3201 was highest we've seen AND gave best LB.

**Friend's insight (validated today):** "When plateau hits, add features from different signal CLASS — subsequence-distance (MP), spectral (FFT) — rather than tuning existing ones."

---

## Current feature stack (v65 — 105 P1 / 112 P2)

| Block | Features | Source |
|---|---|---|
| Base rolling/statistical | 77 | v58 |
| FFT reconstruction error | +5 | v60 |
| TDA persistent homology | +10 (broadcast, cached in `tda_cache/`) | v62 |
| Matrix profile (m=5,10,20) | +3 (per-point, stumpy) | v65 |
| Extra rolling (w=3,7,63) | +6 (mean+std, per-point) | v65 |
| FFT broadcast (top-3 peaks + HF energy) | +4 (broadcast) | v65 |
| **Total P1** | **105** | |
| P2 shift features | +7 | v43 |
| **Total P2** | **112** | |

**Libraries needed**: `stumpy`, `ripser`, `persim`, `pywavelets` (all installed in venv).
**TDA cache**: run `precompute_tda_cache.py` if `tda_cache/` is missing (7 min, 0.3 MB).

---

## What to try next (~5 slots remaining)

**Priority order:**

1. **v66 — pseudo-label iterate on v65** (~+0.001 reliable)
   Change only `PSEUDO_SOURCE = "submission_v65_matrix_profile.json"`.
   Base: `claude_v65_matrix_profile.py`. Use 1 slot.

2. **v67 — cross-matrix profile (test_x vs train_x)**
   Currently MP computes distance of test_x subsequence to nearest match IN test_x (self-MP).
   Cross-MP: distance to nearest match IN train_x. Directly measures "how unusual is this
   test pattern relative to training?" — even stronger anomaly signal.
   In stumpy: use `stumpy.mass()` per subsequence, or `stumpy.stump()` on concatenated series.

3. **v68 — confidence-weighted pseudo-labels on v65 features**
   See `claude_v64_conf_pseudo.py` — adapt to use v65 feature set.
   2-round: Round A (flat weights) → get proba → Round B (confidence-weighted).

4. **Larger MP windows** — try m=3, m=40, m=60 in addition to m=5,10,20.

---

## What NOT to do

- **LightGBM** — consistently worse than HGBT (v59, v2, v13, v14)
- **Wavelet CWT + ACF** — flat on LB (v63), don't add back
- **Multi-round pseudo-labeling (>1 round per slot)** — converges after round 1
- **W_SHIFT ≠ 0.30** — tried 0.20 and 0.40, both worse
- **SPLIT_FRAC ≠ 0.70** — tried 0.60 and 0.80, neutral/worse
- **Include all-zero training windows in P2 pool** — confirmed hurts (v48)
- **The old CNN/segment/CW pipeline** — abandoned, ceiling hit

---

## Key files

| File | Role |
|---|---|
| `claude_v65_matrix_profile.py` | **Current best reference implementation** |
| `claude_v62_tda_cached.py` | v62 base (FFT + TDA cached) |
| `claude_v58_global_ctx.py` | Clean architecture reference (0.6847) |
| `precompute_tda_cache.py` | Regenerate `tda_cache/` if missing |
| `claude_validation.py` | LOO validation harness |
| `claude_validation_v2.py` | Cross-window LOO evaluator |
| `EXPERIMENTS.md` | Full experiment log |
| `submission_v65_matrix_profile.json` | **Current best submission** (0.6890) |

---

## LOO reliability note

| Version | LOO | LB | Direction correct? |
|---|---|---|---|
| v50 +min/max | 0.3154 (↓) | 0.6602 (↑) | **NO** |
| v56 PW=0.50 | 0.3186 (↓) | 0.6821 (↑) | **NO** |
| v58 global ctx | 0.3192 (↑) | 0.6847 (↑) | YES |
| v60 FFT feats | 0.3200 (↑) | 0.6859 (↑) | YES |
| v62 TDA | 0.3186 (↓) | 0.6863 (↑) | **NO** |
| v65 matrix profile | 0.3201 (↑) | 0.6890 (↑) | YES |

**Rule**: Submit if LOO ≥ 0.315 OR change is architecturally sound (new signal class). Do not gate on LOO direction alone.

---

— claude (Sonnet 4.6), 2026-05-17
