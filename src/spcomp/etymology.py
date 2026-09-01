"""Etymology extraction from English Wiktionary.

This is what turns the project from conjecture into measurement. Rather than
inferring *why* a pair diverged by staring at clusters, we read the donor
language straight off Wiktionary's machine-readable etymology templates:

    janela -> {{inh+|pt|roa-opt|janella}}, from {{inh|pt|la-vul|*ianuella}},
              diminutive of {{der|pt|la|ianua||door}}

Wiktionary is mid-migration to a newer compact format, so both are parsed:

    azeitona -> {{ety|pt|:inh|roa-opt:azeitona<id:olive>|id=olive|tree=1}}

Handling only the classic format silently loses a large slice of entries.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

API = "https://en.wiktionary.org/w/api.php"
USER_AGENT = (
    "spanish-portuguese-comp/0.1 (research project; "
    "https://github.com/carlosvera8/Spanish-Portuguese-Comp)"
)

SECTION = {"es": "Spanish", "pt": "Portuguese"}

# Wiktionary usually records only ONE hop ("azeitona < Old Galician-Portuguese
# azeitona"), leaving the Arabic origin on the ancestor's own page. Without
# following the chain, foreign donors are systematically under-counted and the
# Latin-inheritance bucket is inflated. These are the ancestors worth chasing.
ANCESTOR_SECTIONS = {
    "roa-opt": "Old Galician-Portuguese",
    "osp": "Old Spanish",
    "roa-oan": "Old Leonese",
    "gl": "Galician",
}

# Relations that denote a real source. `cog` (cognate) and `m`/`l` (mention)
# are references, not sources, and must not be treated as donors.
_SOURCE_RELATIONS = {
    "inh", "inh+", "bor", "bor+", "der", "der+",
    "lbor", "slbor", "obor", "uder", "learned borrowing",
}

_CLASSIC = re.compile(
    r"\{\{(inh\+?|bor\+?|der\+?|lbor|slbor|obor|uder)\s*\|([^|}]+)\|([^|}]+)(?:\|([^|}]*))?",
    re.IGNORECASE,
)

# Word-formation relations. `arrodillar` is not borrowed from anywhere -- it is
# built inside Spanish from `rodilla` (knee) plus a prefix and suffix. Treating
# such words as "unknown origin" was the second-largest source of unresolved
# etymologies, and wrongly implies mystery where there is none. The base word
# is resolved instead, which yields the true ultimate origin.
_WORD_FORMATION = {"af", "affix", "suf", "suffix", "pre", "prefix",
                   "com", "compound", "blend", "univerbation", "clipping"}

_CLASSIC_FORMATION = re.compile(
    r"\{\{(af|affix|suf|suffix|pre|prefix|com|compound|blend)\s*\|([^|}]+)\|([^}]*)\}\}",
    re.IGNORECASE,
)
# NOTE: the argument body must not be matched with [^:]* -- inline annotations
# such as <pos:state-entering prefix> legitimately contain colons.
_COMPACT_FORMATION = re.compile(
    r":(af|affix|suf|suffix|pre|prefix|com|compound)\|(.*?)(?=\|:|\|text=|\|tree=|\}\}|$)",
    re.IGNORECASE | re.DOTALL,
)


def _formation_bases(raw_args: str) -> list[str]:
    """Pull real word stems out of an affix template, dropping the affixes."""
    bases = []
    for part in raw_args.split("|"):
        term = re.sub(r"<[^>]*>", "", part).strip()
        if "=" in term or not term:
            continue
        # Affixes are written with a leading or trailing hyphen.
        if term.startswith("-") or term.endswith("-"):
            continue
        if len(term) > 2:
            bases.append(term)
    return bases


def parse_word_formation(wikitext: str, language: str) -> list[str]:
    """Return same-language base words this term was built from."""
    section = _section_wikitext(wikitext, _section_name(language))
    if not section:
        return []
    parts = re.split(r"^===+\s*Etymology[^=]*\s*===+\s*$", section, flags=re.MULTILINE)
    scope = parts[1] if len(parts) > 1 else section

    bases: list[str] = []
    for match in _CLASSIC_FORMATION.finditer(scope):
        if match.group(2).strip() == language:
            bases.extend(_formation_bases(match.group(3)))
    for match in _COMPACT_FORMATION.finditer(scope):
        bases.extend(_formation_bases(match.group(2)))
    return bases
# Compact {{ety|pt|:inh|roa-opt:azeitona<...>|:bor|ar:...}} form.
# Both {{ety|...}} and {{etymon|...}} spellings occur in the wild.
_COMPACT_BLOCK = re.compile(r"\{\{ety(?:mon)?\s*\|([^|}]+)\|(.+?)\}\}", re.DOTALL)
_COMPACT_STEP = re.compile(r":(\w+)\s*\|\s*([a-z][a-z0-9-]*)\s*:\s*([^|<>{}]+)")


@dataclass
class Etymology:
    word: str
    language: str
    found: bool = False
    chain: list[tuple[str, str, str]] = field(default_factory=list)  # (relation, lang, term)
    raw: str = ""
    derived_from: str | None = None  # set when origin came via internal word-formation

    @property
    def source_languages(self) -> list[str]:
        return [lang for _, lang, _ in self.chain]


def _section_wikitext(wikitext: str, language_name: str) -> str:
    """Slice out one language's section from a multi-language Wiktionary page."""
    pattern = re.compile(rf"^==\s*{re.escape(language_name)}\s*==\s*$", re.MULTILINE)
    match = pattern.search(wikitext)
    if not match:
        return ""
    start = match.end()
    nxt = re.search(r"^==\s*[^=].*?\s*==\s*$", wikitext[start:], re.MULTILINE)
    return wikitext[start:start + nxt.start()] if nxt else wikitext[start:]


def _section_name(language: str) -> str:
    return SECTION.get(language) or ANCESTOR_SECTIONS[language]


def parse_etymology(wikitext: str, language: str) -> list[tuple[str, str, str]]:
    """Return the ordered (relation, source-language, term) chain."""
    section = _section_wikitext(wikitext, _section_name(language))
    if not section:
        return []

    # Homograph pages carry several numbered etymologies ("cao" is dog <
    # canis, grey-haired < canus, AND khan < Persian). Concatenating them
    # attributes the wrong origin entirely, so only the FIRST block is used --
    # it is the primary sense and the one IDS's core concept list matches.
    parts = re.split(r"^===+\s*Etymology[^=]*\s*===+\s*$", section, flags=re.MULTILINE)
    scope = parts[1] if len(parts) > 1 else section

    chain: list[tuple[str, str, str]] = []
    for match in _CLASSIC.finditer(scope):
        relation, target_lang, source_lang = (
            match.group(1).lower(), match.group(2).strip(), match.group(3).strip()
        )
        if relation not in _SOURCE_RELATIONS:
            continue
        if target_lang != language:
            continue  # a template describing some other language's history
        term = (match.group(4) or "").strip()
        chain.append((relation.rstrip("+"), source_lang, term))

    for block in _COMPACT_BLOCK.finditer(scope):
        if block.group(1).strip() != language:
            continue
        for step in _COMPACT_STEP.finditer(block.group(2)):
            relation = step.group(1).lower()
            if relation not in _SOURCE_RELATIONS:
                continue
            chain.append((relation, step.group(2).strip(), step.group(3).strip()))

    return chain


class WiktionaryClient:
    """Cached, rate-limited client. The cache makes reruns free."""

    def __init__(self, cache_dir: Path, delay: float = 0.12):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_call = 0.0

    def _cache_path(self, word: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", word)[:80]
        return self.cache_dir / f"{safe}.json"

    def wikitext(self, word: str) -> str | None:
        path = self._cache_path(word)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get("wikitext")

        elapsed = time.time() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.time()

        try:
            response = self.session.get(API, params={
                "action": "parse", "page": word, "prop": "wikitext",
                "format": "json", "formatversion": "2", "redirects": "1",
            }, timeout=30)
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None

        text = payload.get("parse", {}).get("wikitext")
        path.write_text(json.dumps({"wikitext": text}), encoding="utf-8")
        return text

    def _lemma_candidates(self, word: str, language: str) -> list[str]:
        """Wiktionary indexes verbs under the bare infinitive.

        IDS lists reflexives ("arrodillarse", "zambullirse"); those pages do
        not exist, which was the single largest source of unresolved
        etymologies before this fallback.
        """
        candidates = [word]
        if word.endswith("se") and len(word) > 4:
            candidates.append(word[:-2])          # arrodillarse -> arrodillar
        if word.endswith("rse") and len(word) > 5:
            candidates.append(word[:-3] + "r")
        lowered = word.lower()
        if lowered != word:
            candidates.append(lowered)
        seen, unique = set(), []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return unique

    def etymology(self, word: str, language: str) -> Etymology:
        # A reflexive like "arrodillarse" often HAS a page, but it is a
        # form-of stub with no etymology. Accepting the first page that merely
        # has a section for the language would stop there and never reach the
        # infinitive that carries the real etymology -- so keep going until a
        # candidate actually yields a chain.
        fallback_text = None
        for candidate in self._lemma_candidates(word, language):
            text = self.wikitext(candidate)
            if not text or not _section_wikitext(text, _section_name(language)):
                continue
            chain = parse_etymology(text, language)
            if chain:
                return Etymology(
                    word=word, language=language, found=True, chain=chain,
                    raw=_section_wikitext(text, _section_name(language))[:400],
                )
            fallback_text = fallback_text or text

        if fallback_text is None:
            return Etymology(word=word, language=language, found=False)
        return Etymology(
            word=word, language=language, found=False, chain=[],
            raw=_section_wikitext(fallback_text, _section_name(language))[:400],
        )

    def etymology_deep(self, word: str, language: str, max_hops: int = 3) -> Etymology:
        """Follow intermediate ancestors so the ultimate donor is reached.

        `azeitona` records only "< Old Galician-Portuguese azeitona"; the
        Arabic source lives on that ancestor's page. Chasing the chain is what
        makes the Arabic layer visible at all.
        """
        result = self.etymology(word, language)
        seen = {(word, language)}

        # Internally derived words ("arrodillar" < "rodilla") have no donor of
        # their own; their origin is whatever the base word's origin is.
        if not result.chain:
            for candidate in self._lemma_candidates(word, language):
                text = self.wikitext(candidate)
                if not text:
                    continue
                for base in parse_word_formation(text, language):
                    if (base, language) in seen:
                        continue
                    seen.add((base, language))
                    base_result = self.etymology(base, language)
                    if base_result.chain:
                        result.chain = base_result.chain
                        result.found = True
                        result.derived_from = base
                        break
                if result.chain:
                    break

        for _ in range(max_hops):
            if not result.chain:
                break
            _, source_lang, term = result.chain[-1]
            if source_lang not in ANCESTOR_SECTIONS or not term:
                break
            term = term.lstrip("*")
            if (term, source_lang) in seen:
                break
            seen.add((term, source_lang))

            text = self.wikitext(term)
            if not text:
                break
            extension = parse_etymology(text, source_lang)
            if not extension:
                break
            # An ancestor's page often restates steps already on the chain
            # (`cao` yielded roa-opt > la > roa-opt > la > ine-pro > la).
            # Duplicates do not change the proximate donor but make the
            # recorded chain misleading, so drop repeats while keeping order.
            for step in extension:
                if step not in result.chain:
                    result.chain.append(step)
            result.found = True

        return result
