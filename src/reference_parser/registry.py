from __future__ import annotations

import importlib
import sys
import types

from .reference_parser import ReferenceParser

OPTIONAL_GRAMMAR_MODULES = [
    ("src.reference_parser.aaai_reference_parser", ["AaaiReferenceParser"]),
    ("src.reference_parser.acl_reference_parser", ["AclReferenceParser"]),
    ("src.reference_parser.acm_reference_parser", ["AcmReferenceParser"]),
    ("src.reference_parser.cvpr_reference_parser", ["CvprReferenceParser"]),
    ("src.reference_parser.iclr_reference_parser", ["IclrReferenceParser"]),
    ("src.reference_parser.icml_reference_parser", ["IcmlReferenceParser"]),
    ("src.reference_parser.ieee_reference_parser", ["IeeeReferenceParser"]),
    ("src.reference_parser.neurips_reference_parser", ["NeuripsReferenceParser"]),
    ("src.reference_parser.tex_reference_parser", ["TexReferenceParser"]),
]

LAZY_EXPORTS = {
    "AaaiReferenceParser": ".aaai_reference_parser",
    "AclReferenceParser": ".acl_reference_parser",
    "AcmReferenceParser": ".acm_reference_parser",
    "CvprReferenceParser": ".cvpr_reference_parser",
    "IclrReferenceParser": ".iclr_reference_parser",
    "IcmlReferenceParser": ".icml_reference_parser",
    "IeeeReferenceParser": ".ieee_reference_parser",
    "NeuripsReferenceParser": ".neurips_reference_parser",
    "TexReferenceParser": ".tex_reference_parser",
    "UnifiedReferenceParser": ".unified_reference_parser",
    "REFERENCE_PARSER_NAME_TO_PARSER": ".unified_reference_parser",
    "ANTLR4_GRAMMAR_PARSER_NAMES": ".unified_reference_parser",
    "VLMReferenceParser": ".vlm_reference_parser",
}


def build_unavailable_parser_stub(class_name: str) -> type:
    """Return a placeholder grammar parser that fails with a clear message."""

    class UnavailableParser:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def parse(self, text):
            del text
            raise ValueError("ANTLR grammar parser is unavailable")

        def parse_one(self, text):
            del text
            raise ValueError("ANTLR grammar parser is unavailable")

    UnavailableParser.__name__ = class_name
    return UnavailableParser


def is_optional_grammar_import_error(exc: ModuleNotFoundError, module_name: str) -> bool:
    missing_name = exc.name or ""
    return (
        missing_name == module_name
        or missing_name.startswith(f"{module_name}.")
        or missing_name.startswith("src.reference_parser.antlr4_grammar")
        or missing_name == "antlr4"
    )


def ensure_optional_grammar_modules() -> None:
    for module_name, class_names in OPTIONAL_GRAMMAR_MODULES:
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if not is_optional_grammar_import_error(exc, module_name):
                raise
            stub = types.ModuleType(module_name)
            for class_name in class_names:
                setattr(stub, class_name, build_unavailable_parser_stub(class_name))
            sys.modules[module_name] = stub


def import_reference_parsers() -> tuple[type[ReferenceParser], type[ReferenceParser]]:
    """Import VLM and unified reference parsers with optional grammar stubs."""
    ensure_optional_grammar_modules()
    from .unified_reference_parser import UnifiedReferenceParser
    from .vlm_reference_parser import VLMReferenceParser

    return VLMReferenceParser, UnifiedReferenceParser


def get_lazy_reference_export(name: str, package_name: str):
    module_name = LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    ensure_optional_grammar_modules()
    module = importlib.import_module(module_name, package_name)
    return getattr(module, name)
