"""Deprecated shim for `guess_init.config`."""

import warnings

warnings.warn(
    "`src.config` is deprecated; use `guess_init.config`.",
    DeprecationWarning,
    stacklevel=2,
)

from guess_init.config import *  # noqa: F401,F403
