# Methodology

## The question

Spanish and Portuguese share most of their vocabulary. Where they *don't*,
why not? This project measures the divergence and attributes each divergent
pair to a historical mechanism.

## Design decisions, and what was rejected

### Alignment before comparison

Cognacy can only be assessed between words that mean the same thing. The
pipeline therefore starts from a concept-aligned word list
([IDS](https://github.com/intercontinental-dictionary-series/ids), 1310
concepts, Spanish = language 176, Portuguese = 178) rather than from two raw
dictionaries.

### Rejected: semantic embeddings as the cognate detector

Word vectors encode **meaning**; cognacy is a fact about **form and descent**.
`perro` and `cão` occupy nearly the same point in any embedding space — both
mean "dog" — while being etymologically unrelated. Conversely `hijo` and
`filho` are cognate but that fact is invisible to a semantic model. Cognate
detection here is edit distance over sound-normalised forms.

Embeddings remain useful for *exploratory* semantic grouping beyond the 22
hand-curated fields, which is a separate question from cognacy.

### Rejected: LDA topic modelling

LDA infers topics from document–word co-occurrence. Isolated dictionary
entries have no documents, so there is nothing for it to condition on. IDS
already ships 22 hand-curated semantic fields, which is a better label set
than anything inferable from a bare word list.

### The load-bearing component: sound-correspondence normalisation

Spanish and Portuguese descend from Latin by *different but regular* sound
changes. Raw string distance therefore mistakes many true cognates for
divergences:

| Spanish | Portuguese | Latin | raw similarity |
|---|---|---|---|
| hijo | filho | FILIUM | 0.40 |
| llave | chave | CLAVEM | 0.40 |
| hecho | feito | FACTUM | 0.40 |
| bueno | bom | BONUM | 0.20 |

`src/spcomp/phonology.py` projects both languages onto a shared archi-form
alphabet, encoding regular correspondences: Latin F- → Spanish *h-*;
PL-/CL-/FL- → Spanish *ll-* / Portuguese *ch-*; -CT- → Spanish *ch* /
Portuguese *it*; -LI-/-C'L- → Spanish *j* / Portuguese *lh*; Latin short
E/O diphthongisation in Spanish only; Portuguese nasal *-ão* ↔ Spanish
*-ón*.

Measured effect (see `results/calibration.txt`): AUC rises from **0.9855** to
**0.9973**, and at a naive 0.6 threshold the share of true cognates
misclassified drops from **25% to 7%**.

### Threshold calibration, not guessing

The decision threshold is chosen by sweeping for best F1 against a
hand-curated gold set of 76 known cognates and 24 known non-cognates
(`src/spcomp/goldset.py`). The gold set deliberately over-samples
orthographically distant cognates, since those are the hard cases. Reported
precision/recall are measured, not assumed.

### Etymology as ground truth, not inference

The "why" is **read**, not guessed. English Wiktionary encodes etymology in
machine-readable templates. Two dialects exist and both are parsed:

```
classic:  {{inh+|pt|roa-opt|janella}}, from {{inh|pt|la-vul|*ianuella}}
compact:  {{etymon|pt|:bor|yue:茶<t:tea>|tree=+}}
```

Two traps handled explicitly:

1. `{{cog}}` / `{{ncog}}` are *cross-references*, not donors, and must not be
   counted as sources.
2. Wiktionary usually records only one hop. `azeitona` says merely
   "< Old Galician-Portuguese *azeitona*"; the Arabic origin lives on the
   ancestor's page. Without recursive resolution, foreign donors are
   systematically undercounted and Latin inheritance is inflated.

### Four mechanisms, not one

| Bucket | Meaning | Example |
|---|---|---|
| A | Different Latin etymon | `ventana` < VENTUM vs `janela` < IANUAM |
| B | Divergent loan route | `té` < Min Nan vs `chá` < Cantonese |
| C | One-sided borrowing | `alfombra` < Andalusi Arabic vs `tapete` < Latin |
| D | Sound-change artefact | filtered upstream by the normaliser |

### Multiple comparisons

Twenty-two semantic fields tested against the base rate is twenty-two
simultaneous hypotheses; at α = 0.05 roughly one spurious hit is expected by
chance. Fields are tested with a two-sided Fisher exact test and corrected
with Benjamini–Hochberg FDR. Both raw and adjusted p-values are reported.

## Known limitations

- **IDS records one primary translation per concept.** Synonyms that would
  reveal a shared cognate (Portuguese has both `janela` and the rarer
  Latin-derived alternatives) are not considered, so divergence is somewhat
  over-counted.
- **96 multiword pairs are excluded**, not scored. They are flagged in
  `pairs.csv`, never silently dropped.
- **Wiktionary is crowd-sourced** and uneven in depth. Resolution rate is
  reported rather than assumed.
- **Recursion depth is capped at 3 hops**, so a few very deep chains stop at
  an intermediate ancestor.
- IDS Spanish uses a phonemic representation; only `ɲ`→`ñ` actually differs
  and is folded back.
