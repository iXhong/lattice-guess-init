"""Deprecated shim for `guess_init.initial_guess`."""

import warnings

warnings.warn(
    "`src.initial_guess` is deprecated; use `guess_init.initial_guess`.",
    DeprecationWarning,
    stacklevel=2,
)

from guess_init.initial_guess import *  # noqa: F401,F403
