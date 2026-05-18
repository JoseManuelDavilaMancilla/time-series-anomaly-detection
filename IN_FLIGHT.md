# In-flight experiments

**Rule**: before starting a long run (anything over ~2 minutes), append a row here. Delete the row when done (results go to `EXPERIMENTS.md`). Read this first when you start a session — don't duplicate what the other agent is already running.

| Started | Author | Experiment | ETA | File / output |
|---|---|---|---|---|
| 2026-05-18 morning | claude | v69 — pseudo-label iteration from v68 (0.6905). Same 108/115-feat arch, PSEUDO_SOURCE=v68. | generating submission | `claude_v69_iter_v68.py` → `submission_v69_iter_v68.json` |
| 2026-05-18 morning | claude | v70 — v67 (6MP+CUSUM) + v68 (STL/AR) combined. 113/120 feats. PSEUDO_SOURCE=v68. | generating submission | `claude_v70_combined.py` → `submission_v70_combined.json` |
| 2026-05-18 morning | claude | v71 — catch22(22) + complexity(3) + per-metric-rules(6) + CatBoost 4th model. 139/146 feats. PSEUDO_SOURCE=v68. | ~45 min remaining | `claude_v71_catch22.py` → `submission_v71_catch22.json` |

---

## How to use this file

1. **Before a run**: add a row with start time, your name, what you're testing, ETA, and the file it'll produce.
2. **During**: leave it alone.
3. **After**: delete your row. Append the result to `EXPERIMENTS.md`.

If a row is older than its ETA + 30 min, assume the agent crashed and feel free to clear it.
