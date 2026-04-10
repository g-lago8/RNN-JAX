"""Top-level package for RNN-Jax.

This file intentionally avoids importing heavy submodules at import time.
It exposes a simple `__version__` and lists the main subpackages.
"""

# ...existing code...
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Inform type checkers / linters that these submodules exist without importing them at runtime
    from . import cells, layers, utils, ssm, datasets  # noqa: F401
# ...existing code...
__version__ = "0.0.2"  # Define the version of the package

__all__ = ["cells", "layers", "utils", "ssm", "datasets"]  # List of submodules to be exposed when using `from rnn_jax import *`

# Consumers can `import rnn_jax` and then import subpackages as needed:
# from rnn_jax import cells
