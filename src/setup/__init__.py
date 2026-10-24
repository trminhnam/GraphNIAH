from .runtime import bootstrap_runtime
from .suppress_warnings import suppress_warnings


def setup_litellm() -> None:
    from .litellm import setup_litellm as configure_litellm

    configure_litellm()


def setup_all() -> None:
    suppress_warnings()
    setup_litellm()


__all__ = ["bootstrap_runtime", "setup_litellm", "suppress_warnings", "setup_all"]
