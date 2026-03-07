"""Deprecated shim for `guess_init.init_guess_loader`."""

import warnings

warnings.warn(
    "`src.init_guess_loader` is deprecated; use `guess_init.init_guess_loader`.",
    DeprecationWarning,
    stacklevel=2,
)

from guess_init.init_guess_loader import *  # noqa: F401,F403
