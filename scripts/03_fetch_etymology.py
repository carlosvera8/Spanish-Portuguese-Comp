"""Stage 3: resolve the donor language for every word in the aligned set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from spcomp.buckets import bucket, ultimate_donor
from spcomp.etymology import WiktionaryClient
from spcomp.util import use_utf8_stdout

use_utf8_stdout()

ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / "data" / "processed" / "scored.csv"
CACHE = ROOT / "data" / "raw" / "wiktionary"
OUT = ROOT / "data" / "processed" / "etymology.csv"

if __name__ == "__main__":
    scored = pd.read_csv(SCORED)
    client = WiktionaryClient(CACHE, delay=0.15)

    records = []
    for row in tqdm(list(scored.itertuples()), desc="wiktionary"):
        spanish = client.etymology_deep(str(row.spanish), "es")
        portuguese = client.etymology_deep(str(row.portuguese), "pt")
        es_code, es_cat = ultimate_donor(spanish.chain)
        pt_code, pt_cat = ultimate_donor(portuguese.chain)
        records.append({
            "Parameter_ID": row.Parameter_ID,
            "concept": row.concept,
            "semantic_field": row.semantic_field,
            "spanish": row.spanish,
            "portuguese": row.portuguese,
            "normalised_similarity": row.normalised_similarity,
            "is_cognate": row.is_cognate,
            "es_donor_code": es_code, "es_donor_category": es_cat,
            "pt_donor_code": pt_code, "pt_donor_category": pt_cat,
            "es_chain": " > ".join(f"{r}:{c}" for r, c, _ in spanish.chain),
            "pt_chain": " > ".join(f"{r}:{c}" for r, c, _ in portuguese.chain),
            "bucket": bucket(es_cat, pt_cat),
        })

    frame = pd.DataFrame(records)
    frame.to_csv(OUT, index=False, encoding="utf-8")
    resolved = (frame["es_donor_category"] != "Unknown") & (frame["pt_donor_category"] != "Unknown")
    print(f"\nrows: {len(frame)}   both-sides resolved: {resolved.sum()} ({resolved.mean():.1%})")
