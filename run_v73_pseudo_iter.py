"""
v73: Pseudo-label iteration from v72 submission.

Usage:
    uv run python run_v73_pseudo_iter.py [--skip-validation]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    pseudo_source = Path("submission_v72_segments.json")
    if not pseudo_source.exists():
        print(f"ERROR: {pseudo_source} not found. Run v72 first.")
        sys.exit(1)

    # Read pipeline.py, patch PSEUDO_SOURCE, run it
    pipeline_src = Path("pipeline.py").read_text()
    # Replace the PSEUDO_SOURCE line
    patched = pipeline_src.replace(
        'PSEUDO_SOURCE   = Path("submission_v68_stl_ar.json")',
        f'PSEUDO_SOURCE   = Path("{pseudo_source.name}")'
    )
    # Also change output filename
    patched = patched.replace(
        'output=Path("submission_v72_segments.json")',
        'output=Path("submission_v73_pseudo.json")'
    )

    temp_script = Path("pipeline_v73_temp.py")
    temp_script.write_text(patched)

    cmd = [sys.executable, str(temp_script)]
    if args.skip_validation:
        cmd.append("--skip-validation")

    print(f"Running v73 with pseudo-labels from {pseudo_source}")
    result = subprocess.run(cmd)
    temp_script.unlink(missing_ok=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
