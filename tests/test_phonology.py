"""The normaliser is the load-bearing component; these pin its behaviour."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from spcomp.cognate import similarity
from spcomp.phonology import normalise

# Pairs that MUST collapse to identical archi-forms: each is a textbook
# regular Ibero-Romance correspondence.
IDENTICAL = [
    ("hijo", "filho"), ("hija", "filha"), ("hoja", "folha"),
    ("llave", "chave"), ("llamar", "chamar"), ("llorar", "chorar"),
    ("noche", "noite"), ("ocho", "oito"), ("leche", "leite"),
    ("hecho", "feito"), ("pecho", "peito"), ("mucho", "muito"),
    ("ojo", "olho"), ("viejo", "velho"), ("oreja", "orelha"),
    ("fuego", "fogo"), ("nuevo", "novo"), ("puerta", "porta"),
    ("tierra", "terra"), ("piedra", "pedra"), ("fiesta", "festa"),
    ("corazón", "coração"), ("león", "leão"),
]


@pytest.mark.parametrize("spanish,portuguese", IDENTICAL)
def test_regular_correspondences_collapse(spanish, portuguese):
    assert normalise(spanish, "es") == normalise(portuguese, "pt"), (
        f"{spanish}/{portuguese} -> "
        f"{normalise(spanish, 'es')}/{normalise(portuguese, 'pt')}"
    )


# Genuinely unrelated words must NOT be forced together by the rules.
DISTINCT = [
    ("ventana", "janela"), ("perro", "cão"), ("olvidar", "esquecer"),
    ("rodilla", "joelho"), ("calle", "rua"), ("alfombra", "tapete"),
]


@pytest.mark.parametrize("spanish,portuguese", DISTINCT)
def test_unrelated_words_stay_apart(spanish, portuguese):
    assert similarity(normalise(spanish, "es"), normalise(portuguese, "pt")) < 0.43


def test_normalisation_never_increases_distance_for_known_cognates():
    """Sound normalisation should help, or at worst not hurt, on true cognates."""
    worse = [
        (es, pt) for es, pt in IDENTICAL
        if similarity(normalise(es, "es"), normalise(pt, "pt")) < similarity(es, pt)
    ]
    assert not worse, f"normalisation hurt these pairs: {worse}"


def test_rejects_unknown_language():
    with pytest.raises(ValueError):
        normalise("casa", "fr")


def test_reflexive_lemma_fallback():
    """Regression: IDS lists reflexives; Wiktionary indexes bare infinitives."""
    from spcomp.etymology import WiktionaryClient

    client = WiktionaryClient.__new__(WiktionaryClient)
    assert "arrodillar" in client._lemma_candidates("arrodillarse", "es")
    assert "zambullir" in client._lemma_candidates("zambullirse", "es")
    assert client._lemma_candidates("casa", "es") == ["casa"]
