"""Demo CLI for initial-guess extraction from (n_boot, n_time) arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# from guess_init.initial_guess import estimate_two_state_initial_guess
# from guess_init.initial_guess_full import estimate_two_state_initial_guess
from guess_init.initail_guess_test import estimate_two_state_initial_guess


def load_bootstrap_array(input_path: Path, npz_key: str | None = None) -> np.ndarray:
    """Load a 2D array with shape (n_boot, n_time) from disk."""
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(input_path)
    elif suffix == ".npz":
        archive = np.load(input_path)
        keys = list(archive.keys())
        if not keys:
            raise ValueError(f"No arrays found in NPZ file: {input_path}")
        if npz_key is None:
            npz_key = keys[0]
        if npz_key not in archive:
            raise ValueError(f"npz_key '{npz_key}' not found. Available keys: {keys}")
        data = archive[npz_key]
    elif suffix in {".csv", ".txt", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(input_path, delimiter=delimiter)
    else:
        raise ValueError("Unsupported format. Use one of: .npy, .npz, .csv, .txt, .dat")

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")
    return data


def format_interval(interval: Any) -> str:
    """Convert interval list to compact [start, ..., end] form."""
    if interval is None:
        return "None"
    if not interval:
        return "[]"
    return f"[{interval[0]}, ..., {interval[-1]}] (len={len(interval)})"


def print_formatted_result(result: dict[str, Any]) -> None:
    """Print a concise, readable summary of initial guesses."""
    print("\nInitial Guess Summary")
    print("=====================")
    print(f"end time ground:{result["end_time_ground"]}")
    print(f"end time excited:{result["end_time_excited"]}")
    print(f"median_ground_log_amp:{result["median_groud_log_amp"]}")
    print(f"median ground mass:{result["median_ground_mass"]}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load (n_boot, n_time) samples and compute formatted initial guesses."
    )
    parser.add_argument(
        "input", type=Path, help="Input array file (.npy/.npz/.csv/.txt/.dat)"
    )
    parser.add_argument(
        "--npz-key", type=str, default=None, help="Array key for .npz input"
    )
    parser.add_argument(
        "--nt-half", type=int, default=48, help="Half temporal extent Nt/2"
    )
    parser.add_argument(
        "--start-time-index", type=int, default=4, help="Ground fit start index"
    )
    parser.add_argument(
        "--min-interval-length",
        type=int,
        default=6,
        help="Minimum interval length for ground-state linear seeds",
    )
    parser.add_argument(
        "--relative-error-threshold",
        type=float,
        default=0.2,
        help="Relative error cutoff threshold for end_time_ground",
    )
    parser.add_argument(
        "--sigma-band",
        type=float,
        default=5.0,
        help="Sigma band for excited-state end-time selection",
    )
    return parser.parse_args()


def main() -> None:
    """Run demo entrypoint."""
    args = parse_args()
    samples_boot_time = load_bootstrap_array(args.input, args.npz_key)
    result = estimate_two_state_initial_guess(
        samples_boot_time=samples_boot_time,
        nt_half=args.nt_half,
        start_time_index=args.start_time_index,
        min_interval_length=args.min_interval_length,
        relative_error_threshold=args.relative_error_threshold,
        sigma_band=args.sigma_band,
    )
    print_formatted_result(result)


if __name__ == "__main__":
    main()
