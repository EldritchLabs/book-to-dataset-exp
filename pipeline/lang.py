# -*- coding: utf-8 -*-
"""Language packs.

Every prompt, every directive and every canned phrase lives in
`prompts/<lang>.yaml`, not in the scripts. Adding a language means copying
`prompts/en.yaml`, translating the strings, and setting `lang:` in the config.
No Python is touched.

Placeholders use double square brackets — `[[scene]]`, `[[char]]` — instead of
Python's `{}` formatting. The prompts are full of literal JSON braces, and with
`str.format` every one of them would have to be escaped by whoever translates
the pack. Square brackets need no escaping, and `render` asserts that nothing
was left unsubstituted, so a typo in a placeholder name fails loudly instead of
sending `[[chr]]` to the model.
"""
import os
import re

import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

_PLACEHOLDER = re.compile(r"\[\[([a-z_]+)\]\]")

# Fallback chapter words, used when a pack does not declare its own. Covers the
# languages the shipped packs cover plus a few near neighbours; a pack for any
# other language should list its own under meta.chapter_words.
DEFAULT_CHAPTER_WORDS = (
    "chapter", "part", "prologue", "epilogue", "interlude", "book", "section",
    "глава", "часть", "пролог", "эпилог", "интерлюдия", "книга", "раздел",
)


def chapter_regex(pack=None, full_line=False):
    """Regex matching a chapter heading line.

    Built from the language pack rather than hard-coded, so adding a language
    does not mean editing the segmentation code. Bare numbers and Roman
    numerals alone on a line are recognised for every language.
    """
    words = list(pack.get("meta.chapter_words", []) if pack else []) or list(DEFAULT_CHAPTER_WORDS)
    alternation = "|".join(re.escape(w) for w in words)
    tail = r"\b.*$" if full_line else r"\b"
    return re.compile(
        rf"^\s*(?:{alternation}){tail}"
        r"|^\s*[IVXLC]{1,7}\s*$"      # a Roman numeral alone on its line
        r"|^\s*\d{1,3}\s*$",          # a bare number alone on its line
        re.I | re.UNICODE,
    )


class LanguagePack:
    """Read-only view over prompts/<lang>.yaml with helpful failures."""

    def __init__(self, lang, data, path):
        self.lang = lang
        self.path = path
        self._data = data

    def get(self, dotted, default=None):
        """Fetch `section.key` from the pack, returning `default` if absent."""
        node = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted):
        """Fetch `section.key`, or stop with a message naming the missing key."""
        value = self.get(dotted, None)
        assert value is not None, (
            f"language pack {self.path} has no '{dotted}'. "
            f"Copy that key from prompts/en.yaml and translate it.")
        return value

    # -- convenience accessors used by more than one step ------------------

    @property
    def unknown_speaker(self):
        """The word the model is told to use when it cannot identify a speaker."""
        return self.require("meta.unknown_speaker")

    @property
    def generic_speakers(self):
        """Words that are never a character name: pronouns, roles, "the voice".

        Attribution returns these regularly, and treating one as a character
        produces transcripts where {{char}} is literally "he".
        """
        return set(w.lower() for w in self.get("meta.generic_speakers", []))

    @property
    def stem_length(self):
        """Prefix length used for crude stemming in the quality gates.

        Inflected languages need this: a legitimate tense change turns "knocked"
        into "knocks" and exact word matching would call that a rewrite. 5 works
        for Russian; set it higher for languages with long roots, or very high
        to effectively disable stemming.
        """
        return int(self.get("meta.stem_length", 5))

    @property
    def refusal_markers(self):
        return list(self.get("meta.refusal_markers", []))

    @property
    def fallback_user(self):
        return self.require("meta.fallback_user")

    @property
    def fallback_char(self):
        return self.require("meta.fallback_char")


# Keys whose value must be a list of plain strings.
_STRING_LISTS = (
    "synth.situations",
    "synth.behavior_mods",
    "openers",
    "diluent_openers",
    "meta.generic_speakers",
    "meta.chapter_words",
    "meta.refusal_markers",
)


def _validate(pack):
    """Catch pack authoring mistakes at load time rather than mid-run.

    The one that actually happens: an unquoted YAML list item containing a
    colon and a space is parsed as a mapping, not a string. The pipeline then
    runs for an hour and fails on whichever row happens to draw that entry.
    Checking here turns that into an immediate, named error.
    """
    for dotted in _STRING_LISTS:
        items = pack.get(dotted)
        if items is None:
            continue
        assert isinstance(items, list), \
            f"{pack.path}: '{dotted}' must be a list, got {type(items).__name__}"
        for index, item in enumerate(items):
            assert isinstance(item, str), (
                f"{pack.path}: '{dotted}' item {index} is a {type(item).__name__}, "
                f"not a string: {item!r}\n"
                f"A YAML list item containing ': ' must be quoted.")

    for axis, values in (pack.get("axes", {}) or {}).items():
        assert isinstance(values, dict), \
            f"{pack.path}: 'axes.{axis}' must be a mapping of value name to directive"
        for name, text in values.items():
            assert isinstance(text, str), (
                f"{pack.path}: 'axes.{axis}.{name}' is a {type(text).__name__}, "
                f"not a string. Quote it if it contains ': '.")

    for dotted in ("cards.full", "cards.core"):
        card = pack.get(dotted)
        if card is not None:
            assert isinstance(card, str), f"{pack.path}: '{dotted}' must be a string"
            assert "[[char]]" in card, \
                f"{pack.path}: '{dotted}' has no [[char]] placeholder — every session " \
                f"would get the same card with no character name in it."
    return pack


class PromptCyclePack:
    """Wrapper over multiple language packs (e.g. en.yaml, en2.yaml, en3.yaml).

    Alternates between packs on consecutive property accesses and render/require calls,
    or returns an exact deterministic pack via `pack.for_index(i)` or `pack.for_id(item_id)`.
    """

    def __init__(self, packs):
        assert packs, "PromptCyclePack requires at least one LanguagePack"
        self._packs = packs
        self._cursor = 0

    def for_index(self, index):
        """Deterministically choose a language pack variant by integer index."""
        try:
            idx = int(index)
        except (TypeError, ValueError):
            idx = 0
        return self._packs[idx % len(self._packs)]

    def for_id(self, item_id):
        """Deterministically choose a language pack variant seeded by item id."""
        try:
            import hashlib
            digest = hashlib.sha1(str(item_id).encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16)
        except Exception:
            idx = 0
        return self._packs[idx % len(self._packs)]

    def _next_pack(self):
        pack = self._packs[self._cursor % len(self._packs)]
        self._cursor += 1
        return pack

    def get(self, dotted, default=None):
        return self._next_pack().get(dotted, default=default)

    def require(self, dotted):
        return self._next_pack().require(dotted)

    @property
    def packs(self):
        return self._packs

    @property
    def lang(self):
        return self._packs[0].lang

    @property
    def unknown_speaker(self):
        return self._packs[0].unknown_speaker

    @property
    def generic_speakers(self):
        # Union across all cycled packs so speaker filtering remains clean
        out = set()
        for p in self._packs:
            out |= p.generic_speakers
        return out

    @property
    def stem_length(self):
        return self._packs[0].stem_length

    @property
    def refusal_markers(self):
        out = []
        for p in self._packs:
            out.extend(p.refusal_markers)
        return list(dict.fromkeys(out))

    @property
    def fallback_user(self):
        return self._packs[0].fallback_user

    @property
    def fallback_char(self):
        return self._packs[0].fallback_char


def load_pack(lang, cfg=None):
    """Load and validate prompts/<lang>.yaml or cycle across <lang>.yaml, <lang>2.yaml, ..."""
    base_path = os.path.join(PROMPTS_DIR, f"{lang}.yaml")
    if not os.path.exists(base_path):
        available = sorted(f[:-5] for f in os.listdir(PROMPTS_DIR) if f.endswith(".yaml"))
        raise SystemExit(
            f"\nNo language pack for lang: {lang}\n"
            f"Available: {', '.join(available)}\n"
            f"To add one: cp prompts/en.yaml prompts/{lang}.yaml and translate the strings.\n")

    should_cycle = False
    if isinstance(cfg, dict):
        should_cycle = bool(cfg.get("prompt_cycle", False))

    if not should_cycle:
        with open(base_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _validate(LanguagePack(lang, data, base_path))

    # Collect <lang>.yaml, <lang>2.yaml, <lang>3.yaml ... in order
    pack_paths = [base_path]
    idx = 2
    while True:
        next_path = os.path.join(PROMPTS_DIR, f"{lang}{idx}.yaml")
        if os.path.exists(next_path):
            pack_paths.append(next_path)
            idx += 1
        else:
            break

    packs = []
    for p in pack_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        packs.append(_validate(LanguagePack(lang, data, p)))

    if len(packs) > 1:
        names = [os.path.basename(p.path) for p in packs]
        print(f"[prompt_cycle] Cycling active across {len(packs)} packs: {', '.join(names)}")
        return PromptCyclePack(packs)

    return packs[0]


def render(template, **values):
    """Substitute [[placeholders]] in a prompt template.

    Every placeholder present in the template must be supplied, and every
    supplied value must be used — both directions are asserted, because a silent
    mismatch here means the model receives an incomplete prompt and the failure
    only shows up as bad data much later.
    """
    wanted = set(_PLACEHOLDER.findall(template))
    given = set(values)
    missing = wanted - given
    assert not missing, f"prompt needs values for: {', '.join(sorted(missing))}"

    result = template
    for key in wanted:
        result = result.replace(f"[[{key}]]", str(values[key]))

    leftover = _PLACEHOLDER.findall(result)
    assert not leftover, f"unsubstituted placeholders remain: {leftover}"
    return result
