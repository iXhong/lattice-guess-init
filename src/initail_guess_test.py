"""Deprecated shim for `guess_init.initail_guess_test`."""

import warnings

warnings.warn(
    "`src.initail_guess_test` is deprecated; use `guess_init.initail_guess_test`.",
    DeprecationWarning,
    stacklevel=2,
)

from guess_init.initail_guess_test import *  # noqa: F401,F403
