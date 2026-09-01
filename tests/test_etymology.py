"""Parser tests for both Wiktionary etymology template dialects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spcomp.buckets import bucket, ultimate_donor
from spcomp.etymology import parse_etymology

CLASSIC = """
==Portuguese==

===Etymology===
{{inh+|pt|roa-opt|janella}}, from {{inh|pt|la-vul|*ianuella}}, diminutive of
{{der|pt|la|ianua||door}}.

===Noun===
"""

COMPACT = """
==Portuguese==

===Etymology===
{{etymon|pt|:bor|yue:茶<t:tea><tr:caa4>|tree=+|text=+}} Portuguese dictionaries
state {{ncog|cmn|-}} as the source.
"""

OTHER_LANGUAGE_ONLY = """
==Spanish==

===Etymology===
{{inh|es|osp|ventana}}.

==Portuguese==

===Etymology===
{{inh|pt|roa-opt|tapete}}.
"""


def test_parses_classic_chain():
    chain = parse_etymology(CLASSIC, "pt")
    assert [c for _, c, _ in chain] == ["roa-opt", "la-vul", "la"]


def test_parses_compact_etymon_template():
    chain = parse_etymology(COMPACT, "pt")
    assert chain and chain[0][1] == "yue"


def test_cog_and_ncog_are_not_treated_as_sources():
    """`cog`/`ncog` are cross-references, not donors."""
    chain = parse_etymology(COMPACT, "pt")
    assert "cmn" not in [c for _, c, _ in chain]


def test_section_isolation():
    """A Spanish section must not leak into the Portuguese chain."""
    assert [c for _, c, _ in parse_etymology(OTHER_LANGUAGE_ONLY, "pt")] == ["roa-opt"]
    assert [c for _, c, _ in parse_etymology(OTHER_LANGUAGE_ONLY, "es")] == ["osp"]


def test_ultimate_donor_skips_romance_intermediates():
    chain = [("inh", "roa-opt", "azeitona"), ("bor", "ar", "az-zaytuna")]
    assert ultimate_donor(chain) == ("ar", "Arabic")


def test_ultimate_donor_reports_latin_when_fully_inherited():
    chain = [("inh", "roa-opt", "tapete"), ("der", "la", "tapete")]
    assert ultimate_donor(chain)[1] == "Latin/Romance-inherited"


def test_buckets():
    assert bucket("Latin/Romance-inherited", "Latin/Romance-inherited") == "A_DIFFERENT_LATIN_ETYMON"
    assert bucket("Arabic", "Latin/Romance-inherited") == "C_ONE_SIDED_BORROWING"
    assert bucket("Asian (Sinitic)", "Asian (Sinitic)") == "B_SAME_DONOR_FAMILY"
    assert bucket("Indigenous American (Nahuatl)", "Indigenous American (Tupi-Guarani)") == "B_DIVERGENT_LOAN_ROUTE"


HOMOGRAPH = """
==Portuguese==

===Etymology 1===
{{inh+|pt|roa-opt|can}}, from {{inh|pt|la|canis}}.

====Noun====
# dog

===Etymology 2===
{{inh+|pt|roa-opt|cão}}, from {{inh|pt|la|cānus}}.

===Etymology 3===
From {{der|pt|fa|خان}}.

====Noun====
# khan
"""


def test_homograph_pages_use_only_the_first_etymology():
    """Regression: `cao` merged dog + grey-haired + khan, yielding Persian."""
    chain = parse_etymology(HOMOGRAPH, "pt")
    codes = [c for _, c, _ in chain]
    assert codes == ["roa-opt", "la"]
    assert "fa" not in codes


def test_donor_is_proximate_not_deepest():
    """Regression: `acucar` should report Arabic contact, not Sanskrit depth."""
    chain = [("inh", "roa-opt", "acucar"), ("bor", "ar", "sukkar"),
             ("der", "fa", "shakar"), ("der", "sa", "sarkara")]
    assert ultimate_donor(chain) == ("ar", "Arabic")


AFFIX_COMPACT = (
    "==Spanish==\n\n===Etymology===\n"
    "{{ety|es|id=to kneel|:af|a-<pos:state-entering prefix><id:verb prefix>"
    "|rodilla<t:knee><id:knee>|-ar<pos:infinitive suffix>|text=+|tree=1}}\n"
)

AFFIX_CLASSIC = "==Spanish==\n\n===Etymology===\n{{af|es|a-|cerca|-ar}}\n"


def test_word_formation_extracts_base_and_drops_affixes():
    """Regression: inline <pos:...> annotations contain colons."""
    from spcomp.etymology import parse_word_formation

    bases = parse_word_formation(AFFIX_COMPACT, "es")
    assert "rodilla" in bases
    assert not any(b.startswith("-") or b.endswith("-") for b in bases)


def test_word_formation_classic_template():
    from spcomp.etymology import parse_word_formation

    assert parse_word_formation(AFFIX_CLASSIC, "es") == ["cerca"]


def test_word_formation_is_not_mistaken_for_a_donor():
    """Affixation is not borrowing; it must not appear in the donor chain."""
    assert parse_etymology(AFFIX_COMPACT, "es") == []
