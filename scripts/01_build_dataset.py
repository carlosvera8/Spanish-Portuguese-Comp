"""Stage 1: download IDS and build the concept-aligned Spanish/Portuguese table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spcomp.ids_data import build_pairs, download
from spcomp.util import use_utf8_stdout

use_utf8_stdout()

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ids"
OUT = ROOT / "data" / "processed" / "pairs.csv"

if __name__ == "__main__":
    print("downloading IDS CLDF tables ...")
    download(RAW)
    pairs = build_pairs(RAW)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT, index=False, encoding="utf-8")
    print(f"aligned pairs: {len(pairs)}")
    print(f"semantic fields: {pairs['semantic_field'].nunique()}")
    print(pairs.head(8).to_string(index=False))
