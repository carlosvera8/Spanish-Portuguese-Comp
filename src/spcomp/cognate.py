"""Cognate scoring for aligned Spanish/Portuguese word pairs.

Cognacy is a claim about *form and descent*, not about meaning, so the
measurement here is edit distance over sound-normalised forms -- not cosine
similarity over semantic embeddings. Two words meaning "dog" (perro / cao)
sit on top of each other in any embedding space while being etymologically
unrelated, which is precisely the distinction we need to preserve.
"""

from __future__ import annotations

from dataclasses import dataclass

from .phonology import normalise, normalise_aggressive


def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance, iterative two-row implementation."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost, # substitution
            ))
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """Length-normalised similarity in [0, 1]; 1.0 means identical."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


@dataclass(frozen=True)
class PairScore:
    spanish: str
    portuguese: str
    raw_similarity: float
    normalised_similarity: float
    aggressive_similarity: float

    @property
    def best_similarity(self) -> float:
        return max(self.normalised_similarity, self.aggressive_similarity)


def score_pair(spanish: str, portuguese: str) -> PairScore:
    """Score one aligned pair under three progressively stronger normalisations."""
    return PairScore(
        spanish=spanish,
        portuguese=portuguese,
        raw_similarity=similarity(spanish.lower(), portuguese.lower()),
        normalised_similarity=similarity(
            normalise(spanish, "es"), normalise(portuguese, "pt")
        ),
        aggressive_similarity=similarity(
            normalise_aggressive(spanish, "es"), normalise_aggressive(portuguese, "pt")
        ),
    )


def calibrate(
    cognate_pairs: list[tuple[str, str]],
    non_cognate_pairs: list[tuple[str, str]],
    metric: str = "best",
) -> dict:
    """Sweep the decision threshold and return the best-F1 operating point.

    Returns the chosen threshold together with the confusion matrix, so the
    downstream error rate is a measured quantity rather than an assumption.
    """
    def value(score: PairScore) -> float:
        return {
            "raw": score.raw_similarity,
            "normalised": score.normalised_similarity,
            "aggressive": score.aggressive_similarity,
            "best": score.best_similarity,
        }[metric]

    positives = [value(score_pair(a, b)) for a, b in cognate_pairs]
    negatives = [value(score_pair(a, b)) for a, b in non_cognate_pairs]

    best = {"f1": -1.0}
    for step in range(0, 101):
        threshold = step / 100
        true_pos = sum(1 for v in positives if v >= threshold)
        false_neg = len(positives) - true_pos
        false_pos = sum(1 for v in negatives if v >= threshold)
        true_neg = len(negatives) - false_pos

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if f1 > best["f1"]:
            best = {
                "metric": metric,
                "threshold": threshold,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "false_negatives": false_neg,
                "false_positives": false_pos,
                "true_negatives": true_neg,
            }
    return best
