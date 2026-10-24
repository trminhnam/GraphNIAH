from __future__ import annotations

from .reference_parser import Reference, ReferenceParser
from .registry import get_lazy_reference_export

__all__ = [
    "ANTLR4_GRAMMAR_PARSER_NAMES",
    "AaaiReferenceParser",
    "AclReferenceParser",
    "AcmReferenceParser",
    "CvprReferenceParser",
    "IclrReferenceParser",
    "IcmlReferenceParser",
    "IeeeReferenceParser",
    "NeuripsReferenceParser",
    "REFERENCE_PARSER_NAME_TO_PARSER",
    "Reference",
    "ReferenceParser",
    "TexReferenceParser",
    "UnifiedReferenceParser",
    "VLMReferenceParser",
]


def __getattr__(name: str):
    value = get_lazy_reference_export(name, __name__)
    globals()[name] = value
    return value
