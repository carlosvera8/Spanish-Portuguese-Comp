"""Run the whole pipeline end to end."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [
    "00_calibration_report.py",
    "01_build_dataset.py",
    "02_score_cognates.py",
    "03_fetch_etymology.py",
    "04_analyze.py",
]

if __name__ == "__main__":
    for stage in STAGES:
        print(f"\n{'#' * 78}\n# {stage}\n{'#' * 78}")
        result = subprocess.run([sys.executable, str(HERE / stage)], check=False)
        if result.returncode != 0:
            sys.exit(f"stage failed: {stage}")
    print("\npipeline complete -- see results/")
