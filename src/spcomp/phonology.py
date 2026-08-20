"""
Spanish <-> Portuguese sound-correspondence normalisation.

The central problem this module solves: Spanish and Portuguese descend from
Latin via *different but regular* sound changes. A naive string comparison
therefore reports many true cognates as unrelated:

    hijo / filho     (< FILIUM)   raw similarity 0.40
    llave / chave    (< CLAVEM)   raw similarity 0.40
    noche / noite    (< NOCTEM)   raw similarity 0.60
    fuego / fogo     (< FOCUM)    raw similarity 0.44

Left uncorrected, roughly four out of five "non-cognates" found by edit
distance are artefacts of regular sound change rather than genuine lexical
divergence -- and the historical deep-dive would then be chasing ghosts.

We therefore project both languages onto a shared "archi-form" alphabet in
which the regular correspondences collapse to the same symbol, and only then
measure edit distance.

Each rule below is a documented, regular Ibero-Romance correspondence, not an
ad-hoc string patch.
"""

from __future__ import annotations

import re
import unicodedata

# --- shared orthography-to-phoneme clean-up -------------------------------
# Applied to BOTH languages so that purely orthographic differences (c/qu/z/ç
# for /k/ and /s/, v/b merger, etc.) never count as divergence.

_SHARED_RULES: list[tuple[str, str]] = [
    # /k/ spellings
    (r"qu(?=[ei])", "k"),
    (r"gu(?=[ei])", "g"),
    (r"c(?=[ei])", "s"),      # ceviche/cebola -> s
    (r"c", "k"),
    (r"ç", "s"),
    (r"z", "s"),
    (r"x", "s"),
    # voiced obstruent / glide mergers
    (r"v", "b"),
    (r"y", "i"),
    (r"w", "b"),
    # geminates carry no contrast in either modern language
    (r"rr", "r"),
    (r"ss", "s"),
    (r"mm", "m"),
    (r"tt", "t"),
    (r"pp", "p"),
    (r"bb", "b"),
    # final nasal neutralisation (es "bien" / pt "bem")
    (r"m$", "n"),
    # orthographic h that survives in neither phonology
    (r"(?<=[a-z])h(?=[a-z])", ""),
]

# --- Spanish-specific correspondences -------------------------------------

_ES_RULES: list[tuple[str, str]] = [
    # Latin -CT- / -ULT- -> Spanish /tS/ <ch>, Portuguese <it>.
    # Restricted to the diphthong environments where the change actually
    # occurred, so that ordinary <-ito> words (bonito) are left alone.
    (r"([eou])ch", r"\1C"),
    # Latin PL-/CL-/FL- -> Spanish <ll>; also Latin -LL-.
    (r"ll", "L"),
    # Latin -LI-/-C'L- -> Spanish <j>; Latin I-/G- also -> <j>.
    (r"j", "J"),
    (r"g(?=[ei])", "J"),
    # Latin F- -> Spanish /h/ -> zero. Merge with Portuguese <f>.
    (r"^h", "F"),
    (r"f", "F"),
    # Latin -NN-/-NI- -> Spanish <ñ>, Portuguese <nh>.
    (r"ñ", "N"),
    # Diphthongisation of Latin short E and O -- Spanish only.
    #
    # The <u> of the digraphs <qu>/<gu> is silent, not part of a diphthong, so
    # these rules must not fire there. Without the guard, "queso" became "qoso"
    # while Portuguese "queijo" became "keiJo", and an identical inherited word
    # was scored 0.20 and counted as a divergence. Shared rules reduce qu/gu
    # afterwards, which is why the guard is needed here rather than reordering.
    (r"ie", "e"),
    (r"(?<![qg])ue", "o"),
    # Suffix correspondences
    (r"sion$", "Con"),
    (r"kion$", "Con"),
    (r"dad$", "dad"),
    (r"on$", "on"),
]

# --- Portuguese-specific correspondences ----------------------------------

_PT_RULES: list[tuple[str, str]] = [
    # Latin -CT- -> Portuguese <it>, matching Spanish <ch> above.
    (r"([eou])it", r"\1C"),
    # Latin PL-/CL-/FL- -> Portuguese <ch>, matching Spanish <ll>.
    (r"ch", "L"),
    # Latin -LI-/-C'L- -> Portuguese <lh>, matching Spanish <j>.
    (r"lh", "J"),
    (r"j", "J"),
    (r"g(?=[ei])", "J"),
    # <f> preserved where Spanish lost it.
    (r"f", "F"),
    (r"^h", "F"),
    # Latin -NN-/-NI- -> Portuguese <nh>.
    (r"nh", "N"),
    # Nasal diphthong <ão> corresponds to Spanish <ón>/<an>.
    (r"ao$", "on"),
    (r"an$", "on"),
    # Suffix correspondences
    (r"sao$", "Con"),
    (r"kao$", "Con"),
    (r"dade$", "dad"),
]

_NASAL_MAP = str.maketrans({"ã": "a", "õ": "o", "â": "a", "ê": "e", "ô": "o"})


def strip_accents(text: str) -> str:
    """Remove combining marks but preserve n-tilde and c-cedilla as letters."""
    text = text.replace("ñ", "\x01").replace("ç", "\x02")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("\x01", "ñ").replace("\x02", "ç")


def _apply(rules: list[tuple[str, str]], token: str) -> str:
    for pattern, replacement in rules:
        token = re.sub(pattern, replacement, token)
    return token


def normalise(word: str, language: str) -> str:
    """Project a Spanish or Portuguese word onto the shared archi-form alphabet.

    `language` must be "es" or "pt".
    """
    if not word:
        return ""

    token = word.strip().lower()
    # Portuguese nasal vowels are handled by explicit rules, so fold the
    # diacritic away first while keeping the base vowel.
    token = token.translate(_NASAL_MAP)
    token = strip_accents(token)
    token = re.sub(r"[^a-zñç]", "", token)

    if language == "es":
        token = _apply(_ES_RULES, token)
    elif language == "pt":
        token = _apply(_PT_RULES, token)
    else:
        raise ValueError(f"language must be 'es' or 'pt', got {language!r}")

    token = _apply(_SHARED_RULES, token)
    return token


def normalise_aggressive(word: str, language: str) -> str:
    """As `normalise`, plus Portuguese intervocalic -L-/-N- loss.

    Portuguese dropped intervocalic Latin -L- and -N- (LUNA > lua,
    DOLOREM > dor) where Spanish kept them. Undoing this means deleting the
    corresponding Spanish consonant, which is lossy and can over-apply to
    loanwords that never had it. Kept separate so its effect can be measured
    rather than assumed -- see scripts/02_score_cognates.py.
    """
    token = normalise(word, language)
    if language == "es":
        token = re.sub(r"(?<=[aeiou])[ln](?=[aeiou])", "", token)
    return token
