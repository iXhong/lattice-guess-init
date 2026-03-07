"""Deprecated shim for `guess_init.initial_guess_full`."""

import warnings

warnings.warn(
    "`src.initial_guess_full` is deprecated; use `guess_init.initial_guess_full`.",
    DeprecationWarning,
    stacklevel=2,
)

from guess_init.initial_guess_full import *  # noqa: F401,F403
