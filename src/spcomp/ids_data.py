"""Loader for the Intercontinental Dictionary Series (IDS) CLDF dataset.

IDS gives us two things that would otherwise have to be invented:

1. A *concept-aligned* Spanish/Portuguese word list. Cognacy can only be
   assessed between words that mean the same thing, so alignment has to come
   before any form comparison.
2. Hand-curated semantic fields (22 chapters, 1310 concepts). This removes the
   need for topic modelling: LDA requires document-word co-occurrence, which
   isolated dictionary entries do not have.

Source: https://github.com/intercontinental-dictionary-series/ids
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import requests

CLDF_BASE = "https://raw.githubusercontent.com/intercontinental-dictionary-series/ids/master/cldf"
SPANISH_ID = "176"
PORTUGUESE_ID = "178"

_FILES = ["languages.csv", "parameters.csv", "chapters.csv", "forms.csv"]


def download(raw_dir: Path) -> None:
    """Fetch the CLDF tables we need into `raw_dir` (idempotent)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        target = raw_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        response = requests.get(f"{CLDF_BASE}/{name}", timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)


# IDS records Spanish in a phonemic representation while Portuguese uses
# standard orthography. Only the palatal nasal actually differs, so fold it
# back to the orthographic form to keep the two comparable.
_PHONEMIC_FIXES = str.maketrans({"ɲ": "ñ"})


def _clean_form(value: str) -> str:
    """Take the first listed variant and drop parenthetical glosses."""
    if not isinstance(value, str):
        return ""
    value = value.translate(_PHONEMIC_FIXES)
    token = value.split(",")[0].split(";")[0].split("/")[0]
    token = token.split("(")[0]
    # IDS ships decomposed (NFD) text; compose so accents are single codepoints.
    return unicodedata.normalize("NFC", token.strip())


def build_pairs(raw_dir: Path) -> pd.DataFrame:
    """Return one row per concept with aligned Spanish and Portuguese forms."""
    forms = pd.read_csv(raw_dir / "forms.csv", dtype=str, low_memory=False)
    parameters = pd.read_csv(raw_dir / "parameters.csv", dtype=str)
    chapters = pd.read_csv(raw_dir / "chapters.csv", dtype=str)

    subset = forms[forms["Language_ID"].isin([SPANISH_ID, PORTUGUESE_ID])].copy()
    subset["clean"] = subset["Form"].fillna(subset["Value"]).map(_clean_form)
    subset = subset[subset["clean"].str.len() > 0]

    # Keep the first attested form per (language, concept); IDS lists synonyms
    # as separate rows and the first is the primary translation.
    primary = subset.groupby(["Language_ID", "Parameter_ID"], as_index=False).first()

    spanish = primary[primary["Language_ID"] == SPANISH_ID][["Parameter_ID", "clean"]]
    spanish = spanish.rename(columns={"clean": "spanish"})
    portuguese = primary[primary["Language_ID"] == PORTUGUESE_ID][["Parameter_ID", "clean"]]
    portuguese = portuguese.rename(columns={"clean": "portuguese"})

    pairs = spanish.merge(portuguese, on="Parameter_ID", how="inner")
    pairs = pairs.merge(
        parameters[["ID", "Name", "Concepticon_Gloss"]],
        left_on="Parameter_ID", right_on="ID", how="left",
    ).drop(columns=["ID"])

    pairs["chapter_id"] = pairs["Parameter_ID"].str.split("-").str[0]
    chapters = chapters.rename(columns={"ID": "chapter_id", "Description": "semantic_field"})
    pairs = pairs.merge(chapters[["chapter_id", "semantic_field"]], on="chapter_id", how="left")

    pairs = pairs.rename(columns={"Name": "concept"})
    # Multiword expressions ("pasado manana" / "depois de amanha") cannot be
    # scored as single tokens. Flag rather than drop, so the exclusion is
    # visible and countable downstream.
    pairs["is_multiword"] = (
        pairs["spanish"].str.contains(r"[\s-]", regex=True)
        | pairs["portuguese"].str.contains(r"[\s-]", regex=True)
    )
    return pairs[
        ["Parameter_ID", "concept", "Concepticon_Gloss", "chapter_id",
         "semantic_field", "spanish", "portuguese", "is_multiword"]
    ]
