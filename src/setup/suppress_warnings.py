import warnings


def suppress_warnings() -> None:
    """Suppress only FutureWarning messages globally."""
    warnings.filterwarnings("ignore", category=FutureWarning)
