# Agent Collaboration Protocol — ANM2026 Time-Series Anomaly Detection

> This file defines the 5-agent swarm that collaborates on every new pipeline version.
> Read this before starting any coding session. Every agent has a distinct role; no agent overrides another without explicit handoff.

---

## Swarm Overview

| # | Agent | Codename | Responsibility | Primary Output |
|---|-------|----------|----------------|----------------|
| 1 | **Researcher** | `ARCHITECT` | Literature review, feature design, experiment proposals | `plan_v{NN}.md`, feature specs |
| 2 | **Coder** | `BUILDER` | Implement features, train models, generate submissions | `pipeline_v{NN}.py`, `submission_v{NN}.json` |
| 3 | **Validator** | `TESTER` | Run validation, compute LOO, catch regressions | Validation logs, per-metric F1 breakdowns |
| 4 | **Ensembler** | `BLENDER` | Tune model weights, post-process, stack submissions | Blend configs, segment-selection params |
| 5 | **Reviewer / Gitkeeper** | `GUARDIAN` | Code review, Git commits, handoff docs, submission tracking | Git commits, updated `HANDOFF.md`, `EXPERIMENTS.md` |

---

## 1. ARCHITECT (Researcher)

**When to activate:** Before any new version is coded.

**Mandate:**
- Propose 1–3 experiment ideas with expected Δ and risk level.
- Reference prior art: cite which `v{XX}` this builds on, which it contradicts.
- Define the feature math precisely (enough that BUILDER can code it without guessing).
- Flag experiments that are **dead-ends** based on `EXPERIMENTS.md` history.

**Rules:**
- Never propose tuning `W_SHIFT` or `SPLIT_FRAC` — already saturated.
- Never propose LightGBM/XGBoost as primary model — confirmed inferior to RF/HGBT at this scale.
- Always estimate validation Δ and LB transfer ratio.

**Handoff to BUILDER:** A markdown plan file (`plan_v{NN}.md`) with:
```markdown
## v{NN}: <one-line description>
- Base version: v{XX}
- Changes: <bullet list>
- Expected val Δ: +0.00X
- Risk: low / medium / high
- Implementation notes: <specific formulas, library calls>
```

---

## 2. BUILDER (Coder)

**When to activate:** After ARCHITECT hands off a plan.

**Mandate:**
- Implement the plan in a new `pipeline_v{NN}.py` (or patch `pipeline.py` if explicitly asked).
- Preserve all working features from the base version. **Do not refactor unrelated code.**
- Add the new feature behind a flag when possible (e.g., `USE_CATCH22 = True`).
- Ensure deterministic output: set all random seeds, use `np.lexsort` for tie-breaking.

**Rules:**
- Import order: `numpy`, `sklearn`, external libs, then local modules (`validation`, `cross_validation`).
- Every new feature function must have a docstring explaining shape, dtype, and semantics.
- Never commit `pipeline.py` directly to git — GUARDIAN handles commits.
- If pseudo-labels are needed and the source JSON is missing, fall back to `PSEUDO_WEIGHT = 0` for the first run, then iterate.

**Handoff to TESTER:**
- The runnable script + expected runtime estimate.
- A one-line summary of what changed vs base.

---

## 3. TESTER (Validator)

**When to activate:** After BUILDER produces a runnable script.

**Mandate:**
- Run stratified holdout (10%, seed=42) and report **overall LOO F1** + **per-metric-type F1**.
- Compare against the base version's per-metric breakdown. Flag any metric type that regresses > 0.005.
- Check for crashes on edge cases: `k=0`, constant train, disjoint ranges, very short series.
- Estimate wall-clock runtime for full 1000-window submission.

**Rules:**
- LOO ≥ 0.315 → recommend submission (per `HANDOFF.md` rule: "Don't trust LOO, submit ≥ 0.315").
- LOO Δ < +0.002 → flag as "noise, do not submit without ARCHITECT review."
- Always report: train time, inference time, memory peak.

**Handoff to BLENDER:**
- Validation report with per-metric F1 table.
- Go / No-Go recommendation for submission.

---

## 4. BLENDER (Ensembler & Post-Processor)

**When to activate:** After TESTER gives Go, or when stacking multiple winning variants.

**Mandate:**
- Tune the model blend weights (HGBT / CatBoost / RF / LR / CNN) via grid search on cached validation scores.
- Optimize segment-selection parameters (`smooth`, `thr_frac`, `small_k_cutoff`) if segment mode is active.
- Manage submission-level ensembles: vote across `submission_v{XX}.json` files when they disagree.
- Compute the **correlation** between candidate submissions before blending — low correlation = high blend value.

**Rules:**
- Blend weight search space: `[0.55, 0.65, 0.75]` for HGBT; `[0.05, 0.10, 0.15, 0.20]` for others.
- Segment param search: only if upstream scores changed materially; otherwise reuse `(smooth=3, thr_frac=0.7, small_k_cutoff=4)`.
- Never delete a `submission_v{XX}.json` until it has been uploaded to the leaderboard and the score is recorded.

**Handoff to GUARDIAN:**
- Final `submission_v{NN}.json` file.
- Blend config or segment params used.
- LB score once uploaded (GUARDIAN backfills this).

---

## 5. GUARDIAN (Reviewer & Gitkeeper)

**When to activate:** Continuously; gates every handoff.

**Mandate:**
- **Code review:** Check BUILDER's diff for bugs, seed leaks, off-by-one errors, NaN propagation.
- **Git hygiene:** `git add`, `git commit` with descriptive messages (`v71: catch22 + complexity + per-metric rules + CatBoost`).
- **Documentation:** Append results to `EXPERIMENTS.md`, update `HANDOFF.md` with new best LB, update `IN_FLIGHT.md` during runs.
- **Submission tracking:** Record every JSON file's LB score in `submissions_log.md`.
- **GitHub sync:** Push commits to origin after every completed experiment. **Never let > 2 versions go unpushed.**

**Rules:**
- Commit message format: `v{NN}: <short description>`
- Tag LB submissions: `git tag lb_v{NN}_{score}` (e.g., `lb_v68_0.6905`).
- If an agent crashes or times out, update `IN_FLIGHT.md` with `[CRASHED]` and notify the swarm.
- Reject any plan that re-tests a known dead-end (see `EXPERIMENTS.md` "Dropped" section).

**Handoff to ARCHITECT:**
- Updated `HANDOFF.md` with new best score and gap analysis.
- List of 3 highest-EV next ideas based on recent results.

---

## Collaboration Flow

```
ARCHITECT → plan_v{NN}.md
    ↓
BUILDER → pipeline_v{NN}.py
    ↓
TESTER → validation report (Go / No-Go)
    ↓
  [If Go]
    ↓
BLENDER → submission_v{NN}.json (+ tuned blend/segments)
    ↓
GUARDIAN → git commit + push + docs update
    ↓
ARCHITECT ← updated HANDOFF.md (loop)
```

**Parallel tracks allowed:**
- ARCHITECT can plan v{NN+1} while TESTER is validating v{NN}.
- BLENDER can tune v{NN-1}'s ensemble while BUILDER codes v{NN}.
- GUARDIAN reviews and commits in parallel with TESTER runs.

**Forbidden:**
- BUILDER starts coding without ARCHITECT's plan.
- TESTER modifies code to "fix" bugs — report to BUILDER instead.
- BLENDER changes feature code — only weights and post-processing.
- Any agent deletes `submission_v{XX}.json` or overwrites a prior version's script.

---

## Session Startup Checklist

Before any agent begins work:

1. [ ] Read `HANDOFF.md` — know the current best LB and gap.
2. [ ] Read `IN_FLIGHT.md` — know what is already running.
3. [ ] Read `EXPERIMENTS.md` — don't repeat dead-ends.
4. [ ] `git status` — working tree must be clean before a new version starts.
5. [ ] Confirm daily submission slots remaining (10/day cap).

---

## Dead-End Register (Do Not Retry Without New Evidence)

| Approach | Version | Why Dropped |
|---|---|---|
| LightGBM primary | v59, v2, v13, v14 | Consistently worse than HGBT/RF |
| XGBoost primary | v5, v14 | Segfaults or underperforms RF |
| Tuning `W_SHIFT` | v43-era | 0.20/0.40 both worse than 0.30 |
| Tuning `SPLIT_FRAC` | v46, v47 | 0.60/0.80 neutral or worse |
| ACF / wavelet | v63 | No signal on top of FFT+TDA+MP |
| Long-context CNN (64pt) | v16 | Overfits, −0.002 |
| Stacking meta-classifier | v17 | Per-fold training too small |
| Segment-level classifier | v18 | −0.044, scores already capture segment info |
| Autoencoder channel | v19 | Correlated with CW, adds noise |
| Test-time adaptation (TTA) | v20, v21 | Hurts most metrics |
| Per-metric CNN routing | v12 | CNNs need scale, overfit on small pools |
| Transformer | v26 | Overfits at 397k params |
| Anomaly Transformer | v30 | Bimodal, ResourceUtil collapses |
| Pseudo-labels (flat binary) | v29 | −0.0055, overfits to false positives |

If ARCHITECT wants to revive any of these, they must provide a **qualitative new hypothesis** for why it would work now (e.g., "on v71's 139-feature space, not v14's 84-feature space").

---

## File Ownership

| File | Owner | Others |
|---|---|---|
| `pipeline.py` / `pipeline_v{NN}.py` | BUILDER | GUARDIAN reviews, TESTER runs |
| `plan_v{NN}.md` | ARCHITECT | Everyone reads |
| `submission_v{NN}.json` | BLENDER | GUARDIAN tracks, TESTER validates format |
| `HANDOFF.md` | GUARDIAN | Everyone reads first |
| `EXPERIMENTS.md` | GUARDIAN | Everyone appends results |
| `IN_FLIGHT.md` | Current runner agent | GUARDIAN cleans up stale rows |
| `submissions_log.md` | GUARDIAN | Everyone reads for history |
| `.git/` | GUARDIAN | No one force-pushes |

---

*Last updated: 2026-05-18 by Kimi Code CLI*
*Next review: after v71 results are in*
