"""Stage 4: test whether non-cognacy concentrates in particular semantic fields,
and attribute each divergent pair to a mechanism."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spcomp.stats import field_enrichment
from spcomp.util import use_utf8_stdout

use_utf8_stdout()

ROOT = Path(__file__).resolve().parents[1]
ETYM = ROOT / "data" / "processed" / "etymology.csv"
RESULTS = ROOT / "results"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


if __name__ == "__main__":
    frame = pd.read_csv(ETYM)
    frame["is_non_cognate"] = ~frame["is_cognate"]
    RESULTS.mkdir(parents=True, exist_ok=True)

    banner("1. OVERALL")
    print(f"concept pairs analysed        {len(frame)}")
    print(f"cognate                       {int(frame['is_cognate'].sum())} "
          f"({frame['is_cognate'].mean():.1%})")
    print(f"non-cognate                   {int(frame['is_non_cognate'].sum())} "
          f"({frame['is_non_cognate'].mean():.1%})")
    resolved = (frame["es_donor_category"] != "Unknown") & (frame["pt_donor_category"] != "Unknown")
    print(f"etymology resolved both sides {int(resolved.sum())} ({resolved.mean():.1%})")

    banner("2. WHY DO PAIRS DIVERGE? (mechanism for non-cognate pairs)")
    divergent = frame[frame["is_non_cognate"]]
    counts = divergent["bucket"].value_counts()
    for name, count in counts.items():
        print(f"  {name:<28} {count:>4}  ({count / len(divergent):>5.1%})")
    counts.to_csv(RESULTS / "divergence_mechanisms.csv")

    banner("3. NON-COGNATE RATE BY SEMANTIC FIELD (Fisher exact, BH-FDR corrected)")
    enrichment = field_enrichment(frame)
    enrichment.to_csv(RESULTS / "field_enrichment.csv", index=False)
    print(f"{'semantic field':<34}{'n':>5}{'non-cog':>9}{'rate':>8}{'lift':>7}"
          f"{'p':>9}{'p_adj':>9}  sig")
    for row in enrichment.itertuples():
        print(f"{row.semantic_field[:33]:<34}{row.n:>5}{row.n_flagged:>9}"
              f"{row.rate:>8.1%}{row.lift:>7.2f}{row.p_value:>9.4f}"
              f"{row.p_adjusted:>9.4f}  {'YES' if row.significant else ''}")

    banner("4. FOREIGN DONORS, BY LANGUAGE (non-inherited sources only)")
    native = {"Latin/Romance-inherited", "Ibero-Romance-ancestor", "Unknown"}
    for language, column in (("SPANISH", "es_donor_category"), ("PORTUGUESE", "pt_donor_category")):
        foreign = frame[~frame[column].isin(native)]
        print(f"\n  {language}: {len(foreign)} borrowed of {len(frame)} "
              f"({len(foreign) / len(frame):.1%})")
        for name, count in foreign[column].value_counts().head(8).items():
            print(f"     {name:<48} {count:>4}")

    banner("5. EXEMPLAR DIVERGENT PAIRS BY MECHANISM")
    for mechanism in ["B_DIVERGENT_LOAN_ROUTE", "C_ONE_SIDED_BORROWING", "A_DIFFERENT_LATIN_ETYMON"]:
        subset = divergent[divergent["bucket"] == mechanism].head(12)
        if subset.empty:
            continue
        print(f"\n  --- {mechanism} ---")
        for row in subset.itertuples():
            print(f"    {str(row.concept)[:26]:<27} {str(row.spanish):<16}"
                  f"{str(row.portuguese):<16} {row.es_donor_code:>8} / {row.pt_donor_code:<8}"
                  f" [{row.semantic_field[:22]}]")

    # --- figure -----------------------------------------------------------
    plot = enrichment.sort_values("rate")
    colours = ["#c0392b" if s else "#95a5a6" for s in plot["significant"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(plot["semantic_field"], plot["rate"], color=colours)
    ax.axvline(frame["is_non_cognate"].mean(), color="#2c3e50", linestyle="--",
               label=f"overall {frame['is_non_cognate'].mean():.1%}")
    ax.set_xlabel("share of concept pairs that are NOT cognate")
    ax.set_title("Spanish/Portuguese lexical divergence by semantic field\n"
                 "(red = significant after Benjamini-Hochberg FDR correction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "divergence_by_field.png", dpi=150)
    print(f"\nwrote {RESULTS / 'figures' / 'divergence_by_field.png'}")
