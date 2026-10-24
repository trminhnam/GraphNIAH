"""Unified reference parser that auto-detects bibliography style.

The :class:`UnifiedReferenceParser` tries each ANTLR4-backed grammar parser in
turn until it finds one that can handle every reference chunk without a grammar
error.  If none succeeds it falls back to the :class:`VLMReferenceParser`.

Public surfaces
---------------
REFERENCE_PARSER_NAME_TO_PARSER
    A module-level dict mapping every parser's canonical name to a
    freshly-instantiated (but *not yet raise_error-configured*) parser object.
    Keyed by the same short strings used in ``ANTLR4_GRAMMAR_PARSER_NAMES``.

ANTLR4_GRAMMAR_PARSER_NAMES
    An ordered list of the parser names that are backed by an ANTLR4 grammar
    (i.e. every parser except :class:`VLMReferenceParser`).  The order controls
    which grammar is tried first during auto-detection.

UnifiedReferenceParser
    The main entry point.  Overrides :meth:`ReferenceParser.parse` to run
    grammar auto-detection and VLM fallback transparently.

Usage example
-------------
>>> parser = UnifiedReferenceParser()
>>> refs = parser.parse(references_section_text)

Development / CI usage (raises instead of falling back)::

>>> parser = UnifiedReferenceParser(raise_error_without_fallback=True)
>>> refs = parser.parse(references_section_text)   # ValueError if no grammar matches
"""

from __future__ import annotations

import logging

from .aaai_reference_parser import AaaiReferenceParser
from .acl_reference_parser import AclReferenceParser
from .acm_reference_parser import AcmReferenceParser
from .cvpr_reference_parser import CvprReferenceParser
from .iclr_reference_parser import IclrReferenceParser
from .icml_reference_parser import IcmlReferenceParser
from .ieee_reference_parser import IeeeReferenceParser
from .neurips_reference_parser import NeuripsReferenceParser
from .reference_parser import Reference, ReferenceParser
from .vlm_reference_parser import VLMReferenceParser

from src.utils.preprocessing import (
    extract_references_section,
    preprocess_references_section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

# Maps a short, stable name to a *production-mode* instance (raise_error=False).
# These instances are used for final parsing once a grammar has been selected.
REFERENCE_PARSER_NAME_TO_PARSER: dict[str, ReferenceParser] = {
    "aaai": AaaiReferenceParser(raise_error=False),
    "acl": AclReferenceParser(raise_error=False),
    "acm": AcmReferenceParser(raise_error=False),
    "cvpr": CvprReferenceParser(raise_error=False),
    "ieee": IeeeReferenceParser(raise_error=False),
    "iclr": IclrReferenceParser(raise_error=False),
    "icml": IcmlReferenceParser(raise_error=False),
    "neurips": NeuripsReferenceParser(raise_error=False),
}

# ---------------------------------------------------------------------------
# Ordered list of grammar-backed parser names
# ---------------------------------------------------------------------------
# The order here determines the probing sequence in UnifiedReferenceParser.
# Put the most common / most strict grammar first so that a definitive match
# is found quickly.  Parsers with broader/looser grammars should come later to
# avoid false positives on documents that belong to a stricter style.
#
# Rationale for current ordering:
#   1. aaai  – semicolon-separated authors + immediate year; very distinctive
#   2. acl   – first-name-first, comma/"and" authors + immediate year
#   3. ieee  – initials-first authors + quoted title; distinctive quote requirement
#   4. iclr  – last-name-first natbib style (similar to icml but different year placement)
#   5. icml  – last-name-first, year at end of body
#   6. neurips – plainnat first-name-first; similar to iclr but distinct .bst
#   7. cvpr  – abbrvnat, year at end; most permissive author grammar → last
ANTLR4_GRAMMAR_PARSER_NAMES: list[str] = [
    "aaai",
    "acl",
    "acm",
    "ieee",
    "iclr",
    "icml",
    "neurips",
    "cvpr",
]

# Sanity-check at import time: every listed name must exist in the registry.
_missing = [
    n for n in ANTLR4_GRAMMAR_PARSER_NAMES if n not in REFERENCE_PARSER_NAME_TO_PARSER
]
if _missing:
    raise RuntimeError(
        f"ANTLR4_GRAMMAR_PARSER_NAMES contains names not found in "
        f"REFERENCE_PARSER_NAME_TO_PARSER: {_missing}"
    )

# ---------------------------------------------------------------------------
# Detection strategy notes
# ---------------------------------------------------------------------------
# CURRENT APPROACH – "probe every chunk with raise_error=True"
#
# For each candidate grammar we create a strict probe instance (raise_error=True)
# and attempt to parse ALL chunks.  If every chunk passes without a ValueError
# we declare that grammar the winner.  If any chunk fails we move to the next
# grammar.  Parsers that override parse() with a merge step (IcmlReferenceParser,
# IclrReferenceParser, AclReferenceParser) call super().parse() internally, so
# probing via parse_one() in a loop is the reliable approach because it avoids
# re-splitting and is grammar-agnostic.
#
# TRADE-OFFS
#   • Cost: O(N × G) parse_one calls for a document with N references and G
#     grammars in the worst case (none match).  In practice the winning grammar
#     is usually found after the first or second probe, and N is small (< 100).
#   • Accuracy: strict probing means one mis-formatted reference can cause an
#     otherwise-correct grammar to be rejected.  We mitigate this with a
#     configurable ``probe_tolerance`` (see __init__) that allows a small
#     fraction of probe failures before giving up on a grammar.
#
# ALTERNATIVE APPROACHES WORTH CONSIDERING
#   A. Heuristic pre-filter (fast, fragile):
#      Inspect the first 3–5 references and look for distinctive tokens
#      (semicolons → AAAI, year immediately after author list → AAAI/ACL, etc.)
#      before attempting full ANTLR4 probing.  Cuts the number of ANTLR4 calls
#      dramatically but requires maintaining a separate set of regex heuristics
#      that can drift out of sync with the grammars.
#
#   B. Confidence scoring (slow, robust):
#      Run all grammars on all chunks; the grammar with the fewest fallback
#      triggers (or the highest ratio of "clean" parses) wins.  More robust to
#      mixed-style documents but doubles parse work even when the answer is
#      obvious.
#
#   C. Document-level metadata hint:
#      If the caller knows the venue (e.g. from a paper's metadata), pass a
#      ``hint`` name to skip probing altogether.  Zero overhead for known venues.
#      Implement by adding an optional ``parser_name`` argument to parse().

# ---------------------------------------------------------------------------
# Unified parser
# ---------------------------------------------------------------------------


class UnifiedReferenceParser(ReferenceParser):
    """Auto-detecting reference parser with VLM fallback.

    Parsing pipeline
    ~~~~~~~~~~~~~~~~
    1. The references section is split into individual reference chunks using
       the same normalisation logic as the base :class:`ReferenceParser`.
    2. Each ANTLR4 grammar parser listed in :data:`ANTLR4_GRAMMAR_PARSER_NAMES`
       is probed in order.  Probing runs :meth:`parse_one` on every chunk with
       ``raise_error=True``; if all chunks pass the grammar is selected.
    3. Once a grammar is selected the *production* instance from
       :data:`REFERENCE_PARSER_NAME_TO_PARSER` is used to parse the full
       section (which also runs the parser-specific merge / post-processing
       step defined in each parser's :meth:`parse` override).
    4. If no grammar survives the probe phase and ``raise_error_without_fallback``
       is ``False``, the :class:`VLMReferenceParser` is used as a last resort.
    5. If ``raise_error_without_fallback`` is ``True`` a :class:`ValueError` is
       raised instead of falling back to the VLM, so developers are alerted to
       add a new grammar parser for the unrecognised style.

    Args:
        vlm_parser: The :class:`VLMReferenceParser` instance used when no
            grammar matches.  If ``None`` a default instance is created lazily.
        raise_error_without_fallback: Development flag.  When ``True``, raise
            :class:`ValueError` instead of invoking the VLM fallback.  Set to
            ``True`` during development / CI to surface unknown bibliography
            styles early.  Defaults to ``False`` (production mode).
        probe_tolerance: Maximum number of chunks that are *allowed* to fail
            the grammar probe before that grammar is rejected.  Defaults to
            ``0`` (all chunks must parse cleanly).  A value of ``1`` or ``2``
            can be useful for documents where a handful of references have
            unusual formatting but the majority match a known grammar.
    """

    def __init__(
        self,
        vlm_parser: VLMReferenceParser | None = None,
        raise_error_without_fallback: bool = False,
        probe_tolerance: int = 0,
    ) -> None:
        self._vlm_parser = vlm_parser
        self.raise_error_without_fallback = raise_error_without_fallback
        self.probe_tolerance = probe_tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_one(self, text: str) -> Reference | None:
        """Parse a single reference string.

        Tries each ANTLR4 grammar parser in order and returns the first
        successful result.  Falls back to the VLM parser if all grammars fail
        (or raises if ``raise_error_without_fallback`` is set).

        Note: For a full references section, prefer :meth:`parse` which runs
        grammar auto-detection over the whole document and then delegates to the
        winning parser's own :meth:`parse` (including its merge logic).
        """
        for name in ANTLR4_GRAMMAR_PARSER_NAMES:
            probe = self._make_probe_parser(name)
            try:
                ref = probe.parse_one(text)
                if ref is not None:
                    return ref
            except ValueError:
                continue

        # All grammars failed for this single reference.
        if self.raise_error_without_fallback:
            raise ValueError(
                f"No ANTLR4 grammar could parse the reference and "
                f"raise_error_without_fallback=True. Text: {text!r}"
            )
        return self._get_vlm_parser().parse_one(text)

    def parse(self, text: str) -> list[Reference]:
        """Parse a full references section, auto-detecting the grammar style.

        Steps
        -----
        1. Split the section text into individual reference chunks.
        2. Probe each grammar parser (in the order defined by
           :data:`ANTLR4_GRAMMAR_PARSER_NAMES`) against all chunks.
           - A probe parser is created with ``raise_error=True`` so that any
             ANTLR4 failure surfaces immediately as a :class:`ValueError`.
           - If the number of failing chunks is within ``probe_tolerance``, the
             grammar is accepted.
           - When a new grammar candidate fails a chunk, the probe restarts from
             the *first* chunk with the next candidate.  This prevents partial
             results from an incorrect grammar polluting earlier chunks.
        3. Parse the full section with the winning grammar's production instance
           (which includes each parser's merge / post-processing :meth:`parse`
           override).
        4. If no grammar matches:
           - Raise :class:`ValueError` if ``raise_error_without_fallback=True``.
           - Otherwise delegate to :class:`VLMReferenceParser`.

        Returns
        -------
        list[Reference]
            Parsed references in document order.
        """
        references_text = extract_references_section(text)
        if not references_text:
            # If extraction found nothing, treat the whole input as the section.
            references_text = text

        references_text = preprocess_references_section(references_text)

        chunks = self._prepare_chunks(references_text)
        if not chunks:
            return []

        detected_name = self._detect_grammar(chunks)

        if detected_name is not None:
            logger.debug(
                "UnifiedReferenceParser: detected grammar %r for %d chunk(s).",
                detected_name,
                len(chunks),
            )
            # Use the production (non-raising) parser which also runs the
            # parser-specific merge / post-processing in its parse() override.
            return REFERENCE_PARSER_NAME_TO_PARSER[detected_name].parse(references_text)

        # ── No grammar matched ──────────────────────────────────────────────
        if self.raise_error_without_fallback:
            tried = ", ".join(ANTLR4_GRAMMAR_PARSER_NAMES)
            raise ValueError(
                f"No ANTLR4 grammar parser could handle this references section "
                f"(tried: {tried}).  Add a grammar parser for this bibliography "
                f"style or set raise_error_without_fallback=False to use the VLM "
                f"fallback.  First chunk sample: {chunks[0][:200]!r}"
            )

        logger.info(
            "UnifiedReferenceParser: no grammar matched; falling back to VLM parser."
        )
        return self._get_vlm_parser().parse(references_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_grammar(self, chunks: list[str]) -> str | None:
        """Return the name of the first grammar that survives probing all chunks.

        Implements the "restart from chunk 0 when switching candidate" contract:
        if grammar G fails on chunk i, the probe for grammar G+1 starts over at
        chunk 0.  This ensures earlier chunks are not left parsed by a grammar
        that ultimately turned out to be wrong.

        Args:
            chunks: Pre-normalised reference strings from :meth:`_prepare_chunks`.

        Returns:
            The winning grammar name, or ``None`` if every grammar failed.
        """
        for name in ANTLR4_GRAMMAR_PARSER_NAMES:
            probe = self._make_probe_parser(name)
            failures = 0
            accepted = True

            for chunk in chunks:
                try:
                    probe.parse_one(chunk)
                except ValueError:
                    failures += 1
                    if failures > self.probe_tolerance:
                        # This grammar cannot handle the document → restart with
                        # the next candidate from chunk 0.
                        accepted = False
                        logger.debug(
                            "UnifiedReferenceParser: grammar %r rejected "
                            "(failed chunk: %r).",
                            name,
                            chunk[:80],
                        )
                        break

            if accepted:
                return name

        return None

    @staticmethod
    def _make_probe_parser(name: str) -> ReferenceParser:
        """Create a *strict* (raise_error=True) instance of the named grammar parser.

        A fresh instance is created for every probe so that probe calls are
        completely isolated and cannot mutate shared state in
        :data:`REFERENCE_PARSER_NAME_TO_PARSER`.
        """
        parser_classes: dict[str, type] = {
            "aaai": AaaiReferenceParser,
            "acl": AclReferenceParser,
            "acm": AcmReferenceParser,
            "cvpr": CvprReferenceParser,
            "ieee": IeeeReferenceParser,
            "iclr": IclrReferenceParser,
            "icml": IcmlReferenceParser,
            "neurips": NeuripsReferenceParser,
        }
        cls = parser_classes[name]
        return cls(raise_error=True)

    def _get_vlm_parser(self) -> VLMReferenceParser:
        """Return the VLM parser, creating a default instance on first access."""
        if self._vlm_parser is None:
            self._vlm_parser = VLMReferenceParser()
        return self._vlm_parser


if __name__ == "__main__":
    parser = UnifiedReferenceParser()
    texts = [
        #######################################################################
        # IEEE
        """[23] J. Ahn and S. Kwak, 'Learning pixel-level semantic affinity with imagelevel supervision for weakly supervised semantic segmentation,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2018, pp. 4981-4990.""",
        """[24] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2016, pp. 770-778.""",
        """[25] S. Jie and Z.-H. Deng, 'Convolutional bypasses are better vision transformer adapters,' arXiv preprint arXiv:2207.07039 , 2022.""",
        #######################################################################
        # ACM
        """[40] Anbang Xu and Brian Bailey. 2012. What do you think?: a case study of benefit, expectation, and interaction in a large online critique community. In Proceedings of the ACM 2012 conference on Computer Supported Cooperative Work . 295-304. https://doi.org/10.1145/2145204.2145252""",
        """[41] Anbang Xu, Haibin Liu, Liang Gou, Rama Akkiraju, Jalal Mahmud, Vibha Sinha, Yuheng Hu, and Mu Qiao. 2016. Predicting Perceived Brand Personality with Social Media. In Proceedings of the Tenth International Conference on Web and Social Media . 436-445. http://www.aaai.org/ocs/index.php/ICWSM/ICWSM16/paper/view/13078""",
    ]
    for text in texts:
        refs = parser.parse(text)
        logger.info("%s", refs)
        logger.info("%s", "-" * 100)
