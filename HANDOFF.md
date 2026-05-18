# Handoff notes — read this first in the new chat

Last updated: 2026-05-18 (afternoon)

## Bottom line

- **Our best LB**: 0.6902 (v66 pseudo_iter, May 17 ~10:21pm) — gap to 0.70 = **0.0098**
- **v72 generated** (segment selection, no pseudo-labels) — pending LB submission
- **v72+CNN ensemble generated** (0.7/0.3 blend, 10.36% disagreement) — pending LB submission
- Remaining slots today: ~1–2

---

## Today's progression (2026-05-18)

| Version | Key change | LB |
|---|---|---|
| v65 | Matrix profile m=5,10,20 + extra rolling + FFT broadcast | 0.6890 |
| v67 | 6 MP windows (m=3,5,10,15,20,30) + CUSUM features | 0.6901 |
| **v66** | **Pseudo-label iteration from v65** | **0.6902 ← best** |
| v72 | Segment selection (smooth=3, thr_frac=0.7) + 139/146 feats | **pending** |
| v72+cnn | 0.7×v72 + 0.3×CNN ensemble | **pending** |

---

## What's running / ready

- **v69 (running)**: Pseudo-label iteration of v67 (stronger architecture: 6 MP windows + CUSUM, 110/117 feats). PSEUDO_SOURCE = v67. Expected: ~0.6910–0.6915.
- **v68 (written, not started)**: STL seasonal decomposition + AR(1) residuals (3 new features). The "professor-style" contextual anomaly detection. P1=108, P2=115. Run: `uv run python claude_v68_stl_ar.py`

---

## Current best architecture (v67, 110/117 features)

```
Features (P1: 110, P2: 117):
  - v65 base (105/112):
    * 77 base: z-scores, diffs, rolling mean/std w=5,11,21,41, rolling median/MAD w=11/41,
      EWMA, percentile rank vs train, position, time-of-day/dow, local volatility,
      rmax/rmin at w=5,11,21,41, global context (mean_z/std_z/max_z), metric_type one-hot (6),
      service one-hot (30), interval, anomaly ratios (3)
    * 5 FFT reconstruction: residual, residual_z, res_vs_train, periodicity_strength, res_std_log
    * 10 TDA: h0/h1 persistent homology features (cached)
    * 3 MP: matrix profile discord scores at m=5,10,20
    * 6 extra rolling: mean+std at w=3,7,63
    * 4 FFT broadcast: top-3 peak mags + HF energy ratio
  - NEW in v67 (+5):
    * 3 MP: extend to m=3,5,10,15,20,30 (6 total scales)
    * 2 CUSUM: S_pos[i]=max(0, S_pos[i-1]+z[i]-0.5), S_neg symmetric (path-dependent)
  - P2 shift features (+7): rank_in_self, self_robust_z, above_ref_max, below_ref_min,
    mean_shift_bc, std_ratio_bc, median_shift_bc

Pseudo-label weight: 0.70 (765/1000 windows have k>0)
Models per metric type: 3 RF + 5 HGBT + 1 LR
Blend: 0.80×HGBT + 0.10×RF + 0.10×LR
W_SHIFT=0.30, SPLIT_FRAC=0.70, smooth w=5 alpha=0.8
```

---

## Key files

- `claude_v67_more_mp_cusum.py` — **current best reference** (110/117 feats, 6 MP + CUSUM)
- `claude_v69_pseudo_iter_v67.py` — **in flight** (iterate from v67)
- `claude_v68_stl_ar.py` — **ready to run** (STL/AR seasonal features, new signal class)
- `claude_v65_matrix_profile.py` — previous best reference (105/112 feats)

---

## What to do next session

**Priority order** (gap = 0.0098):

### 1. Check v69 result — iterate from v67 (reliable ~+0.001)
v69 is running or just completed. Submit if LOO ≥ 0.315.

### 2. Run v68 (STL/AR features) — potential +0.002–0.005
`uv run python claude_v68_stl_ar.py`
The "professor-style" approach: seasonal decomposition (statsmodels STL) + AR(1) forecast residuals.
For 421 windows with intervals ≥ 864s: daily STL with period=round(86400/intervals).
For others: linear detrend fallback.
3 new features: AR(1) residual z, STL/trend residual z, running |AR| mean.
PSEUDO_SOURCE: submission_v67_more_mp_cusum.json (or best available). PW=0.70.
KEY INSIGHT: dataset has 7 distinct interval groups (~143 windows each). The professor
designed contextual anomalies that look normal globally but wrong for their time-of-day.
STL captures this.

### 3. Combine v67 + v68 (v70) — if both win, stack them
110/117 (v67) + 3 (STL/AR) = 113/120 features.

### 4. Iterate pseudo-labels again after each win

---

## What NOT to do

- **Don't tune W_SHIFT** — tried 0.20/0.40, both worse than 0.30
- **Don't tune SPLIT_FRAC** — tried 0.60/0.80, neutral/worse
- **Don't use LightGBM** — consistently worse than HGBT (v59, v2, v13, v14)
- **Don't trust LOO** — submit anything LOO ≥ 0.315 with sound architecture
- **Don't add ACF/wavelet** — v63 confirmed these add no signal on top of FFT+TDA+MP

## Dataset insight (critical for v68)

- **7 interval groups**: 60s(143), 300s(143), 345s(151), 600s(144), 864s(149), 1200s(143), 3600s(127)
- Each service appears EXACTLY ONCE — no cross-metric features possible
- For intervals ≥ 864s (421 windows): 3+ daily cycles in training → STL feasible
- For 3600s (127 windows): 24 pts/day, ~13 days training → excellent STL candidate

---

## ⚠️ CRITICAL: GitHub Tracking Requirement (2026-05-18)

**All changes, new submissions, and experiment results MUST be tracked on GitHub.**
- Commit every new script, submission JSON, and result file to the repo.
- Push after every agent completes a run.
- Tag submissions with version numbers for traceability.
- Remote: `https://github.com/JoseManuelDavilaMancilla/time-series-anomaly-detection.git`

---

## Agent Swarm Results (2026-05-18 evening)

Target: **0.73 LB** (gap = 0.0395 from current best 0.6905).

### Completed

| Agent | Task | Result | Status |
|---|---|---|---|
| **Agent 2 (Segmenter)** | Port segment selection to `pipeline.py` | **DONE** — `predict_segments()` added, `predict_window()` now uses contiguous segment selection (smooth=3, thr_frac=0.7). Committed as v72. | ✅ Merged |
| **Agent 3 (DL-Builder)** | Build CNN ensemble | **DONE** — `cnn_addon.py` + `generate_cnn_submission_fast.py` created. 1 seed, 5 epochs, ~2.5 min runtime. `submission_cnn_fast.json` generated. | ✅ Ready |
| **BUILDER** | Fix infra | **DONE** — Added `--skip-validation` flag to `pipeline.py`. Added symlinks for `student_dataset` and `tda_cache`. Graceful fallback when pseudo-labels missing. | ✅ Merged |

### Completed (v72)

| Task | Result | Status |
|---|---|---|
| v72 pipeline (`--skip-validation`) | `submission_v72_segments.json` generated (1000 windows, 25583 anomalies) | ✅ Done |
| v72+CNN ensemble | `submission_v72_cnn_ensemble.json` (0.7/0.3 blend, 10.36% disagreement) | ✅ Done |
| Validation | **0.3057** (n=100) — comparable to v68 without pseudo-labels | ✅ Done |

### In Flight

| Task | ETA | Status |
|---|---|---|
| v73 pseudo-label iteration (`PSEUDO_SOURCE=submission_v72_segments.json`) | ~18 min | Running (background task) |

### Submissions Ready for LB

| File | Description |
|---|---|
| `submission_v72_segments.json` | v72 with segment selection, no pseudo-labels |
| `submission_v72_cnn_ensemble.json` | 0.7×v72 + 0.3×CNN ensemble |
| `submission_v73_pseudo.json` | v73 pseudo-label iteration of v72 (pending) |

### Next Steps (Priority)

1. **Submit v72 to LB** → get score.
2. **Submit v72+CNN ensemble to LB** → compare.
3. **Wait for v73** → submit if v72 LB was good.
4. **If v73 LB > v72**: iterate again (v74) with v73 as pseudo-label source.
5. **CNN integration into pipeline.py**: train CNN alongside tree ensemble for end-to-end blending.

