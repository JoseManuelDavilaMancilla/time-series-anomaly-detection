"""
Validate submission.json format and lengths.
"""

import json
from pathlib import Path
import numpy as np

DATASET_ROOT = Path("student_dataset")
SUBMISSION = Path("submission.json")


def main():
    with open(SUBMISSION, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "predictions" in data, "Missing 'predictions' key"
    preds = data["predictions"]
    assert len(preds) == 1000, f"Expected 1000 windows, got {len(preds)}"

    window_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()])
    assert len(window_dirs) == 1000, f"Expected 1000 dataset windows, got {len(window_dirs)}"

    errors = []
    for wdir in window_dirs:
        wid = wdir.name.split("_", 1)[0]
        test_len = len(np.load(wdir / "test.npy"))
        if wid not in preds:
            errors.append(f"Window {wid}: missing prediction")
            continue
        pred = preds[wid]
        pred_len = len(pred)
        if pred_len != test_len:
            errors.append(f"Window {wid}: pred={pred_len}, test={test_len}")
        if not all(isinstance(v, int) and v in (0, 1) for v in pred):
            errors.append(f"Window {wid}: predictions must be 0 or 1 integers")

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("All validations passed!")


if __name__ == "__main__":
    main()
