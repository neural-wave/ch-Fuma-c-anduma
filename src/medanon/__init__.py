"""medanon - context-aware, clinically informed anonymisation of doctor-patient encounters."""
__version__ = "0.1.0"

from .config import load_config  # noqa: E402
from .pipeline import Anonymizer  # noqa: E402

__all__ = ["Anonymizer", "load_config", "__version__"]
