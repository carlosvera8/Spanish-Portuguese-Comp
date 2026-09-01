"""Stage 0: quantify how much the sound-correspondence layer actually buys.

Written as its own stage because the claim "normalisation matters" is an
empirical one and should be checkable, not asserted in a README.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

from spcomp.cognate import calibrate, score_pair
from spcomp.goldset import NON_COGNATES, TRUE_COGNATES
from spcomp.util import use_utf8_stdout

use_utf8_stdout()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "calibration.txt"

METRICS = {
    "raw": lambda s: s.raw_similarity,
    "normalised": lambda s: s.normalised_similarity,
    "aggressive": lambda s: s.aggressive_similarity,
    "best": lambda s: s.best_similarity,
}

if __name__ == "__main__":
    scored = ([(1, score_pair(a, b)) for a, b, _ in TRUE_COGNATES]
              + [(0, score_pair(a, b)) for a, b, _ in NON_COGNATES])
    labels = np.array([label for label, _ in scored])

    lines = [
        "CALIBRATION REPORT",
        f"gold set: {int((labels == 1).sum())} known cognates / "
        f"{int((labels == 0).sum())} known non-cognates",
        "",
        "Threshold-independent separability:",
        f"  {'metric':<12}{'AUC':>9}{'mean(cog)':>12}{'mean(non)':>12}{'gap':>8}",
    ]
    for name, getter in METRICS.items():
        values = np.array([getter(s) for _, s in scored])
        lines.append(
            f"  {name:<12}{roc_auc_score(labels, values):>9.4f}"
            f"{values[labels == 1].mean():>12.3f}{values[labels == 0].mean():>12.3f}"
            f"{values[labels == 1].mean() - values[labels == 0].mean():>8.3f}"
        )

    lines += ["", "Best-F1 operating point:",
              f"  {'metric':<12}{'thr':>6}{'F1':>8}{'prec':>8}{'recall':>8}"]
    for name in METRICS:
        result = calibrate([(a, b) for a, b, _ in TRUE_COGNATES],
                           [(a, b) for a, b, _ in NON_COGNATES], name)
        lines.append(f"  {name:<12}{result['threshold']:>6.2f}{result['f1']:>8.3f}"
                     f"{result['precision']:>8.3f}{result['recall']:>8.3f}")

    lines += ["", "Cost of skipping sound normalisation",
              "(true cognates misclassified as divergent):"]
    positives = [s for label, s in scored if label == 1]
    for threshold in (0.40, 0.50, 0.60, 0.70):
        raw_miss = sum(1 for s in positives if s.raw_similarity < threshold)
        norm_miss = sum(1 for s in positives if s.normalised_similarity < threshold)
        lines.append(
            f"  threshold {threshold:.2f}:  raw misses {raw_miss:>2}/{len(positives)} "
            f"({raw_miss / len(positives):>5.1%})   "
            f"normalised misses {norm_miss:>2}/{len(positives)} "
            f"({norm_miss / len(positives):>5.1%})"
        )

    lines += ["", "Largest rescues (raw -> normalised):"]
    deltas = sorted(
        ((s.normalised_similarity - s.raw_similarity, s) for s in positives),
        key=lambda item: -item[0],
    )
    for delta, s in deltas[:10]:
        lines.append(f"  {s.spanish:<12}{s.portuguese:<12}"
                     f"{s.raw_similarity:.2f} -> {s.normalised_similarity:.2f}  (+{delta:.2f})")

    report = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(report)
