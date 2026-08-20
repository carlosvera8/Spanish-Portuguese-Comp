# Spanish–Portuguese Lexical Divergence

Where Spanish and Portuguese use *unrelated* words for the same concept — why?
This measures the divergence across 1,214 concept-aligned pairs and attributes
each divergent pair to a historical mechanism, using Wiktionary etymology as
ground truth rather than inference from clusters.

## Headline results

| | |
|---|---|
| Concept pairs analysed | **1,214** (IDS core vocabulary) |
| Cognate | **886 (73.0%)** |
| Non-cognate | **328 (27.0%)** |
| Etymology resolved both sides | **86.1%** |

### 1. Divergence is mostly *internal*, not contact-driven

| Mechanism | Pairs | Share |
|---|---|---|
| **A — different Latin etymon** | 167 | **50.9%** |
| C — one-sided borrowing | 65 | 19.8% |
| B — divergent loan route | 15 | 4.6% |
| Unresolved | 81 | 24.7% |

Over half of all divergence is two languages inheriting from Latin and simply
picking **different Latin words** — `ventana` (< VENTUM, "wind") vs `janela`
(< IANUAM, "door"); `arder` vs `queimar`; `subir` vs `ascender`. No foreign
contact involved. Contact of any kind explains under a quarter.

### 2. No semantic field is significantly enriched for divergence

Testing all 22 fields with Fisher exact + Benjamini–Hochberg FDR, **exactly one
field is significant — and in the opposite direction**:

| Field | n | non-cognate rate | lift | p | p-adj |
|---|---|---|---|---|---|
| **Quantity** | 34 | **2.9%** | 0.11 | 0.0006 | **0.0123** ✓ |
| The house | 39 | 38.5% | 1.42 | 0.14 | 0.58 |
| Basic actions | 69 | 37.7% | 1.39 | 0.050 | 0.36 |
| Spatial relations | 67 | 37.3% | 1.38 | 0.065 | 0.36 |
| Cognition | 48 | 14.6% | 0.54 | 0.048 | 0.36 |

Numerals are almost perfectly conserved (1 divergent pair in 34). Nothing is
significantly *more* divergent than baseline.

**This is why the FDR correction mattered.** Raw p-values would have declared
"Basic actions" (p = 0.0499) and "Cognition" (p = 0.0475) significant. With 22
simultaneous tests, those are exactly the false positives the correction is
designed to kill.

### 3. The Arabic layer is largely *shared*, not divergent

A common intuition is that Moorish influence explains Spanish–Portuguese
divergence. In core vocabulary it mostly does the opposite — both languages
borrowed the *same* Arabic words, because al-Andalus covered the whole
peninsula:

```
azúcar / açúcar      aceite / azeite      aceituna / azeitona
almohada / almofada  ojalá / oxalá
```

These are **cognate pairs**. Measured Arabic loans in the core list: Spanish
31, Portuguese 20 — a real but modest asymmetry, not a wholesale difference.
The largest "foreign" donor for *both* languages is Greek (55 / 52), almost
all learned and scientific vocabulary borrowed long after the two split.

Genuine one-sided Arabic cases exist and are visible in the output —
`albañil` (< Andalusi Arabic *bannāʔ*) vs `pedreiro`; `alfombra` vs `tapete`.

### 4. Where contact *does* show up, it is trade geography

The clearest divergent-route pairs cluster in animals, food, and household
goods — reflecting that Spain and Portugal met the world through different
doors:

| Concept | Spanish | Portuguese | Donors |
|---|---|---|---|
| tea | té | chá | Min Nan (via Dutch) / **Cantonese (via Macau)** |
| potato | papa | batata | Quechua / **Taíno** |
| crocodile | cocodrilo | jacaré | Greek / **Tupi** |
| cup | taza | xícara | Andalusi Arabic / **Nahuatl** |
| monkey | mono | macaco | Arabic / **Bantu** |
| pineapple | piña | abacaxi | Latin / **Tupi–Cariban** |

## What this rejects

**Semantic embeddings do not detect cognates.** Vectors encode meaning;
cognacy is form plus descent. `perro` and `cão` are neighbours in any embedding
space and etymologically unrelated.

**LDA topic modelling does not apply.** It needs document–word co-occurrence;
isolated dictionary entries have none. IDS ships 22 hand-curated semantic
fields, which is a better label set than anything inferable from a word list.

**Raw edit distance is not enough.** Regular sound change makes true cognates
look unrelated. Measured on the calibration set, sound normalisation raises AUC
from **0.9855 → 0.9973**, and at a 0.6 threshold cuts misclassified cognates
from **25% → 7%**:

| Spanish | Portuguese | raw | normalised |
|---|---|---|---|
| hijo | filho | 0.40 | **1.00** |
| hecho | feito | 0.40 | **1.00** |
| bueno | bom | 0.20 | **0.75** |

## Pipeline

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.txt
python scripts/run_all.py
```

| Stage | Does |
|---|---|
| `00_calibration_report.py` | Measures what sound normalisation buys |
| `01_build_dataset.py` | Downloads IDS CLDF, aligns es/pt by concept |
| `02_score_cognates.py` | Calibrates threshold on gold set, scores all pairs |
| `03_fetch_etymology.py` | Resolves donor language per word (cached) |
| `04_analyze.py` | Fisher + BH-FDR enrichment, mechanism attribution |

`pytest tests/ -q` → 44 tests. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Limitations

- **Homographs.** 22.7% of words have multiple numbered Wiktionary etymologies;
  the first block is used, which mis-assigns some (`papa` "potato" picks up the
  "pope" etymology). This pushes pairs *out of* bucket A into B/C, so the
  finding that A dominates is **conservative**.
- **24.7% of divergent pairs unresolved** — Wiktionary coverage, not silent
  dropping. Reported, not hidden.
- **IDS gives one primary translation per concept**, so synonyms that would
  reveal a shared cognate are missed; divergence is somewhat over-counted.
- **96 multiword pairs excluded** from scoring, flagged in `pairs.csv`.
- Core vocabulary only. A frequency-weighted or full-dictionary tier would
  likely show *more* contact-driven divergence, since borrowings concentrate in
  culturally specific, lower-frequency vocabulary.

## Data

- [Intercontinental Dictionary Series](https://github.com/intercontinental-dictionary-series/ids) (CLDF) — CC BY 4.0
- [English Wiktionary](https://en.wiktionary.org) — CC BY-SA 4.0
