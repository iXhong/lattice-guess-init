"""Run one-state and two-state initial guess estimators and optionally save per-p2 JSON outputs.

Usage examples:
    python demo_run.py data/p2_0_bs.npy --save
    python demo_run.py data/processed/bs --save --output-dir data/processed/init_guess
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Updated imports to include both estimators
from guess_init.initial_guess_full import (
    estimate_one_state_initial_guess,
    estimate_two_state_initial_guess,
)

P2_PATTERN = re.compile(r"(p2_\d+)")


def _load_array(path: Path, npz_key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        npz = np.load(path)
        keys = list(npz.keys())
        if not keys:
            raise ValueError(f"NPZ file has no arrays: {path}")
        key = npz_key if npz_key is not None else keys[0]
        if key not in npz:
            raise ValueError(f"Key '{key}' not found in {path}. Available keys: {keys}")
        arr = npz[key]
    elif suffix in {".csv", ".txt", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        arr = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError("Unsupported input format. Use .npy/.npz/.csv/.txt/.dat")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n_boot, n_time), got shape {arr.shape}")
    return arr


def _extract_p2_id(path: Path) -> str:
    match = P2_PATTERN.search(path.stem)
    if match:
        return match.group(1)
    return path.stem


def _jsonify(value: Any) -> Any:
    """Convert numpy/scalar/container values to JSON-safe native types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _normalize_one_state_record(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one-state interval record to fixed schema."""
    success = bool(item.get("fit_success", False))
    return {
        "interval_start": int(item["interval_start"]),
        "interval_end": int(item["interval_end"]),
        "interval": list(range(int(item["interval_start"]), int(item["interval_end"]) + 1)),
        "fit_status": success,
        "fit_result": {
            "log_amp": float(item["log_amp"]),
            "mass": float(item["mass"]),
            "chisqr": float(item["chisqr"]) if "chisqr" in item else None,
            "redchi": float(item["redchi"]) if "redchi" in item else None,
        } if success else None,
    }


def _normalize_two_state_record(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize two-state interval record to fixed schema for future loading."""
    success = bool(item.get("two_state_fit_success", False))
    fit_obj = item.get("two_state_result") if success else None

    return {
        "interval_start": int(item["interval_start"]),
        "interval_end": int(item["interval_end"]),
        "interval": list(item["interval"]),
        "seed": {
            "local_ground_log_amp": float(item["local_ground_log_amp"]),
            "local_ground_mass": float(item["local_ground_mass"]),
            "guess_excited_log_amp": float(item["guess_excited_log_amp"]),
            "guess_mass_gap": float(item["guess_mass_gap"]),
        },
        "fit_status": success,
        "fit_result": _jsonify(fit_obj) if fit_obj is not None else None,
    }


def _build_save_payload(
    one_state_result: list[dict[str, Any]],
    two_state_result: dict[str, Any],
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    
    # Process one state results
    norm_one_state = [_normalize_one_state_record(item) for item in one_state_result]
    norm_one_state.sort(key=lambda x: (x["interval_start"], x["interval_end"]))

    # Process two state results
    all_intervals_raw = two_state_result.get("all_intervals_results", [])
    norm_two_state = [_normalize_two_state_record(item) for item in all_intervals_raw]
    norm_two_state.sort(key=lambda x: (x["interval_start"], x["interval_end"]))

    return {
        "schema_version": "v2",
        "unit": {
            "p2_id": _extract_p2_id(input_path),
            "input_file": str(input_path),
            "output_file": str(output_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "config": _jsonify(config),
        "metadata": {
            "end_time_ground": two_state_result.get("end_time_ground"),
            "end_time_excited": two_state_result.get("end_time_excited"),
            "median_ground_log_amp": two_state_result.get("median_ground_log_amp"),
            "median_ground_mass": two_state_result.get("median_ground_mass"),
        },
        "one_state_results": norm_one_state,
        "two_state_results": norm_two_state,
    }


def _save_result_json(
    one_state_result: list[dict[str, Any]],
    two_state_result: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    p2_id = _extract_p2_id(input_path)
    output_path = output_dir / f"{p2_id}_init_guess.json"
    payload = _build_save_payload(one_state_result, two_state_result, input_path, output_path, config)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def _resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("p2_*_bs.npy"))
        if not files:
            raise ValueError(
                f"No files matching 'p2_*_bs.npy' found under: {input_path}"
            )
        return files
    raise ValueError(f"Input path does not exist: {input_path}")


def _print_result(
    file_path: Path, 
    one_state_res: list[dict[str, Any]], 
    two_state_res: dict[str, Any], 
    top_k: int
) -> None:
    print("\n========================================")
    print(f"Estimator Output for: {file_path.name}")
    print("========================================")
    print(f"end_time_ground      : {two_state_res.get('end_time_ground')}")
    print(f"end_time_excited     : {two_state_res.get('end_time_excited')}")
    print(f"median_ground_log_amp: {two_state_res.get('median_ground_log_amp', 0):.10e}")
    print(f"median_ground_mass   : {two_state_res.get('median_ground_mass', 0):.10e}")

    # --- Print One-State Top Results ---
    successful_one = [item for item in one_state_res if item.get("fit_success")]
    successful_one = sorted(successful_one, key=lambda x: (x["interval_start"], x["interval_end"]))
    shown_one = successful_one[:top_k]

    print(f"\n[ One-State ] Top successful interval fits (showing up to {top_k})")
    print("----------------------------------------------------------------")
    if not shown_one:
        print("No successful one-state interval fits.")
    else:
        for idx, item in enumerate(shown_one, start=1):
            print(
                f"{idx:02d}. interval=[{item['interval_start']},{item['interval_end']}] "
                f"mass={item['mass']:.8e} log_amp={item['log_amp']:.8e} "
                f"redchi={item.get('redchi', 0):.8e}"
            )

    # --- Print Two-State Top Results ---
    all_intervals_two = two_state_res.get("all_intervals_results", [])
    successful_two = [item for item in all_intervals_two if item.get("two_state_fit_success")]
    successful_two = sorted(successful_two, key=lambda x: (x["interval_start"], x["interval_end"]))
    shown_two = successful_two[:top_k]

    print(f"\n[ Two-State ] Top successful interval fits (showing up to {top_k})")
    print("----------------------------------------------------------------")
    if not shown_two:
        print("No successful two-state interval fits.")
    else:
        for idx, item in enumerate(shown_two, start=1):
            two = item["two_state_result"]
            print(
                f"{idx:02d}. interval=[{item['interval_start']},{item['interval_end']}] "
                f"m1={two['mass_1']:.8e} m2={two['mass_2']:.8e} "
                f"dm={two['mass_gap']:.8e} redchi={two['redchi']:.8e}"
            )
    print("========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-state and two-state initial guess estimators and save per-p2 JSON."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory. Directory mode scans p2_*_bs.npy files.",
    )
    parser.add_argument(
        "--npz-key", type=str, default=None, help="Key used when input is .npz"
    )
    parser.add_argument("--nt-half", type=int, default=48, help="Nt/2 value")
    parser.add_argument(
        "--start-time-index", type=int, default=4, help="Ground fit start index"
    )
    parser.add_argument(
        "--min-interval-length-1st",
        type=int,
        default=2,
        help="Minimum interval length for one-state fits",
    )
    parser.add_argument(
        "--min-interval-length-2nd",
        type=int,
        default=6,
        help="Minimum interval length for two-state fits",
    )
    parser.add_argument(
        "--relative-error-threshold",
        type=float,
        default=0.2,
        help="Relative-error cutoff threshold for ground-state end time",
    )
    parser.add_argument(
        "--sigma-band",
        type=float,
        default=5.0,
        help="Sigma band for excited-state cutoff",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of successful interval fits to print per state",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save structured per-p2 JSON output with both one-state and two-state results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/init_guess"),
        help="Directory to save per-p2 JSON outputs when --save is enabled.",
    )
    args = parser.parse_args()

    config = {
        "nt_half": args.nt_half,
        "start_time_index": args.start_time_index,
        "min_interval_length_1st": args.min_interval_length_1st,
        "min_interval_length_2nd": args.min_interval_length_2nd,
        "relative_error_threshold": args.relative_error_threshold,
        "sigma_band": args.sigma_band,
    }

    input_files = _resolve_input_files(args.input)
    
    for file_path in input_files:
        samples = _load_array(file_path, npz_key=args.npz_key)
        
        # 1. Estimate one-state parameters
        result_one_state = estimate_one_state_initial_guess(
            samples_boot_time=samples,
            nt_half=args.nt_half,
            start_time_index=args.start_time_index,
            min_interval_length=args.min_interval_length_1st,
            relative_error_threshold=args.relative_error_threshold,
        )

        # 2. Estimate two-state parameters
        result_two_state = estimate_two_state_initial_guess(
            samples_boot_time=samples,
            nt_half=args.nt_half,
            start_time_index=args.start_time_index,
            min_interval_length=args.min_interval_length_2nd,
            relative_error_threshold=args.relative_error_threshold,
            sigma_band=args.sigma_band,
        )
        
        # Print combined results
        _print_result(
            file_path=file_path, 
            one_state_res=result_one_state, 
            two_state_res=result_two_state, 
            top_k=max(args.top_k, 1)
        )
        
        # Save payload if required
        if args.save:
            saved_path = _save_result_json(
                one_state_result=result_one_state,
                two_state_result=result_two_state,
                input_path=file_path,
                output_dir=args.output_dir,
                config=config,
            )
            print(f"Successfully saved JSON to: {saved_path}")


if __name__ == "__main__":
    main()
