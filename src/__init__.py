"""Deprecated compatibility namespace.

Import from `guess_init` instead of `src`.
This compatibility bridge will be removed in the next release.
"""

import warnings

warnings.warn(
    "The `src` package namespace is deprecated. "
    "Use `guess_init` imports instead.",
    DeprecationWarning,
    stacklevel=2,
)
