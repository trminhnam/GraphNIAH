"""Shared prompt text used by prompt building and inference."""

from .academic_paper_qa import FORMAT_REMINDER, SYSTEM_PROMPT
from .reachability_yesno_qa import (
    FORMAT_REMINDER as REACHABILITY_FORMAT_REMINDER,
    SUPPORTED_TRAVERSAL_MODES as REACHABILITY_TRAVERSAL_MODES,
    TRAVERSAL_MODE_MIXED as DEFAULT_REACHABILITY_TRAVERSAL_MODE,
    build_system_prompt as build_reachability_system_prompt,
)
from .reachability_multiple_qa import (
    FORMAT_REMINDER as REACHABILITY_MULTIPLE_FORMAT_REMINDER,
    build_system_prompt as build_reachability_multiple_system_prompt,
)
from .reachability_counting_qa import (
    FORMAT_REMINDER as REACHABILITY_COUNTING_FORMAT_REMINDER,
    build_system_prompt as build_reachability_counting_system_prompt,
)

__all__ = [
    "FORMAT_REMINDER",
    "DEFAULT_REACHABILITY_TRAVERSAL_MODE",
    "REACHABILITY_FORMAT_REMINDER",
    "REACHABILITY_COUNTING_FORMAT_REMINDER",
    "REACHABILITY_MULTIPLE_FORMAT_REMINDER",
    "REACHABILITY_TRAVERSAL_MODES",
    "SYSTEM_PROMPT",
    "build_reachability_system_prompt",
    "build_reachability_counting_system_prompt",
    "build_reachability_multiple_system_prompt",
]
