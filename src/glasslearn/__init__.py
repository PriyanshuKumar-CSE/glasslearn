"""glasslearn: scikit-learn you can see through."""

from importlib.metadata import PackageNotFoundError, version

from . import datasets
from ._config import config_context, get_config, set_config
from ._validation import ConvergenceWarning, InvalidParameterError, NotFittedError

try:
    __version__ = version("glasslearn")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "ConvergenceWarning",
    "InvalidParameterError",
    "NotFittedError",
    "__version__",
    "config_context",
    "datasets",
    "get_config",
    "set_config",
]
