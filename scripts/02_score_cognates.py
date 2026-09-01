"""Stage 2: calibrate the cognate threshold, then score every aligned pair."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from spcomp.cognate import calibrate, score_pair
from spcomp.goldset import NON_COGNATES, TRUE_COGNATES
from spcomp.util import use_utf8_stdout

use_utf8_stdout()

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "processed" / "pairs.csv"
OUT = ROOT / "data" / "processed" / "scored.csv"
METRIC = "normalised"

if __name__ == "__main__":
    cognates = [(a, b) for a, b, _ in TRUE_COGNATES]
    non_cognates = [(a, b) for a, b, _ in NON_COGNATES]

    calibration = calibrate(cognates, non_cognates, METRIC)
    print("calibration on hand-curated gold set "
          f"({len(cognates)} cognates / {len(non_cognates)} non-cognates)")
    for key in ("metric", "threshold", "f1", "precision", "recall"):
        print(f"  {key:<10} {calibration[key]}")
    print(f"  confusion  TP={calibration['true_positives']} "
          f"FN={calibration['false_negatives']} "
          f"FP={calibration['false_positives']} "
          f"TN={calibration['true_negatives']}")

    threshold = calibration["threshold"]

    pairs = pd.read_csv(PAIRS)
    usable = pairs[~pairs["is_multiword"]].copy()
    print(f"\nscoring {len(usable)} single-token pairs "
          f"({int(pairs['is_multiword'].sum())} multiword pairs excluded)")

    scores = [score_pair(row.spanish, row.portuguese) for row in usable.itertuples()]
    usable["raw_similarity"] = [s.raw_similarity for s in scores]
    usable["normalised_similarity"] = [s.normalised_similarity for s in scores]
    usable["archi_spanish"] = [
        __import__("spcomp.phonology", fromlist=["normalise"]).normalise(s.spanish, "es")
        for s in scores
    ]
    usable["archi_portuguese"] = [
        __import__("spcomp.phonology", fromlist=["normalise"]).normalise(s.portuguese, "pt")
        for s in scores
    ]
    usable["is_cognate"] = usable["normalised_similarity"] >= threshold
    usable["threshold"] = threshold

    OUT.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(OUT, index=False, encoding="utf-8")

    cognate_rate = usable["is_cognate"].mean()
    print(f"\ncognate rate overall: {cognate_rate:.1%}")
    print(f"non-cognate pairs to explain: {int((~usable['is_cognate']).sum())}")
    naive = (usable["raw_similarity"] >= threshold).mean()
    print(f"(cognate rate WITHOUT sound normalisation: {naive:.1%} "
          f"-> {cognate_rate - naive:+.1%} correction)")
