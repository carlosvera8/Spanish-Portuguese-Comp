"""Classify why a Spanish/Portuguese pair diverged.

Non-cognacy is not one phenomenon. Lumping it together is what makes the
"why" question feel unanswerable. Four mechanisms are separable here:

  A  DIFFERENT_LATIN_ETYMON  both words inherited from Latin, but from
                             *different* Latin words (ventana < VENTUM vs
                             janela < IANUAM). No external contact at all.
  B  DIVERGENT_LOAN_ROUTE    both borrowed, same referent, different donor
                             (te < Min Nan via Dutch vs cha < Cantonese
                             via Macau).
  C  ONE_SIDED_BORROWING     one language borrowed, the other kept Romance
                             (alfombra < Andalusi Arabic vs tapete < Latin).
  D  SOUND_CHANGE_ARTEFACT   not really divergent; handled upstream by the
                             phonological normaliser.
"""

from __future__ import annotations

# Wiktionary language codes grouped into donor categories.
DONOR_CATEGORIES: dict[str, str] = {}


def _register(category: str, codes: str) -> None:
    for code in codes.split():
        DONOR_CATEGORIES[code] = category


_register("Latin/Romance-inherited",
          "la la-vul la-med la-lat la-ecc la-cla la-imp la-lit itc-pro itc-ola ine-pro")
_register("Ibero-Romance-ancestor", "roa-opt osp roa-oan gl ast pt es")
_register("Arabic", "ar xaa ary arz acw ar-and")
_register("Germanic",
          "gem-pro goh got gmw-pro gml gmh non nl de en ang frk odt sv da nb enm dum")
_register("Other-Romance (French/Occitan/Italian/Catalan)",
          "fr fro frm oc pro ca it roa-oit vec nap scn rm co")
_register("Greek", "grc el gkm")
_register("Indigenous American (Tupi-Guarani)", "tpw tpn tup tup-gua gn tpr")
_register("Indigenous American (Caribbean/Arawak)", "tnq awd arw sai-car cai crb")
_register("Indigenous American (Nahuatl)", "nah nci nhn azc-pro")
_register("Indigenous American (Quechua/Aymara)", "qu qwh quz ay")
_register("African (Bantu)", "kmb umb bnt-pro ln sw zu kg lua yo wo ff bnt")
_register("Asian (Sinitic)", "zh yue nan cmn ltc och nan-hbl hak wuu")
_register("Asian (Japanese/Korean/SE-Asian)", "ja ko ms id jv tl vi km th")
_register("Asian (Indic/Iranian/Turkic)",
          "sa hi ur ta te kn ml mr si pa bn ne fa tr ota az uz ps ku")
_register("Celtic", "cel-pro xtg ga cy gd bre xcel")
_register("Basque", "eu")
_register("Hebrew/Aramaic", "he arc yi")

# Codes that mean "inherited within Ibero-Romance" for bucketing purposes.
_INHERITED = {"Latin/Romance-inherited", "Ibero-Romance-ancestor"}


def categorise(language_code: str) -> str:
    return DONOR_CATEGORIES.get(language_code, f"Other/unclassified ({language_code})")


def ultimate_donor(chain: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Return (language_code, category) of the *proximate* foreign donor.

    The chain runs shallow-to-deep (pt < Old Galician-Portuguese < Arabic <
    Persian < Sanskrit). Walking FORWARD and taking the first non-Romance
    source names the contact event that actually happened in Iberia -- Arabic
    for `acucar`, not Sanskrit. Walking backward instead would report the
    deepest reconstructable ancestor, which is a different question.

    If every step stays inside Latin/Ibero-Romance, the word is inherited.
    """
    if not chain:
        return ("unknown", "Unknown")

    for _, code, _ in chain:
        category = categorise(code)
        if category not in _INHERITED and not category.startswith("Other/unclassified"):
            return (code, category)

    for _, code, _ in reversed(chain):
        if categorise(code) == "Latin/Romance-inherited":
            return (code, "Latin/Romance-inherited")

    last_code = chain[-1][1]
    return (last_code, categorise(last_code))


def bucket(spanish_category: str, portuguese_category: str) -> str:
    """Assign a divergence mechanism to a non-cognate pair."""
    spanish_native = spanish_category in _INHERITED
    portuguese_native = portuguese_category in _INHERITED

    if spanish_category == "Unknown" or portuguese_category == "Unknown":
        return "UNKNOWN"
    if spanish_native and portuguese_native:
        return "A_DIFFERENT_LATIN_ETYMON"
    if not spanish_native and not portuguese_native:
        # Both borrowed. Same donor family means parallel borrowing of the same
        # cultural item; different families mean genuinely divergent contact.
        if spanish_category == portuguese_category:
            return "B_SAME_DONOR_FAMILY"
        return "B_DIVERGENT_LOAN_ROUTE"
    return "C_ONE_SIDED_BORROWING"
