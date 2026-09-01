"""Field-level enrichment testing.

The substantive question is whether non-cognacy concentrates in particular
semantic domains. With 22 semantic fields, testing each against the base rate
means 22 simultaneous hypotheses -- at alpha = 0.05 roughly one spurious
"significant" field is expected by chance alone. Benjamini-Hochberg FDR
control is therefore applied, and both raw and adjusted p-values are reported.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def field_enrichment(
    frame: pd.DataFrame,
    field_column: str = "semantic_field",
    flag_column: str = "is_non_cognate",
    alpha: float = 0.05,
    min_count: int = 5,
) -> pd.DataFrame:
    """Test each semantic field for enrichment of `flag_column` against the rest.

    Uses a two-sided Fisher exact test per field (exact, so small cells are
    handled correctly), then Benjamini-Hochberg across fields.
    """
    overall_rate = frame[flag_column].mean()
    rows = []

    for field_name, group in frame.groupby(field_column):
        in_flag = int(group[flag_column].sum())
        in_total = len(group)
        if in_total < min_count:
            continue
        out = frame[frame[field_column] != field_name]
        out_flag = int(out[flag_column].sum())
        out_total = len(out)

        table = [[in_flag, in_total - in_flag], [out_flag, out_total - out_flag]]
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

        rows.append({
            "semantic_field": field_name,
            "n": in_total,
            "n_flagged": in_flag,
            "rate": in_flag / in_total,
            "baseline_rate": overall_rate,
            "lift": (in_flag / in_total) / overall_rate if overall_rate else float("nan"),
            "odds_ratio": odds_ratio,
            "p_value": p_value,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    reject, adjusted, _, _ = multipletests(result["p_value"], alpha=alpha, method="fdr_bh")
    result["p_adjusted"] = adjusted
    result["significant"] = reject
    return result.sort_values("rate", ascending=False).reset_index(drop=True)
