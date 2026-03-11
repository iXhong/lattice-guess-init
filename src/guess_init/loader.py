"""Tiny loader utility for per-p2 initial-guess JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# def load_interval_initial_params(json_path: str | Path) -> dict[str, Any]:
#     """Load structured initial parameters for each interval from saved JSON.

#     Returns a dict with:
#     - unit: source/unit metadata
#     - intervals: list of interval records, each containing:
#       - interval_start, interval_end, interval
#       - one_state_init: {log_amp, mass}
#       - two_state_init: {log_amp_1, mass_1, log_amp_2, mass_gap, mass_2}
#       - fit_status
#     """
#     path = Path(json_path)
#     payload = json.loads(path.read_text(encoding="utf-8"))

#     unit = payload.get("unit", {})
#     all_intervals = payload.get("all_intervals_results", [])

#     intervals_out: list[dict[str, Any]] = []
#     for item in all_intervals:
#         seed = item.get("seed", {})
#         fit_result = item.get("fit_result", {})
#         one_log_amp = float(seed["local_ground_log_amp"])
#         one_mass = float(seed["local_ground_mass"])
#         two_log_amp_1 = float(fit_result["log_amp_1"])
#         two_mass_1 = float(fit_result["mass_1"])
#         two_log_amp_2 = float(fit_result["log_amp_2"])
#         two_mass_gap = float(fit_result["mass_gap"])
#         two_mass_2 = float(fit_result["mass_2"])
#         # two_log_amp_2 = float(seed["guess_excited_log_amp"])
#         # two_mass_gap = float(seed["guess_mass_gap"])

#         intervals_out.append(
#             {
#                 "interval_start": int(item["interval_start"]),
#                 "interval_end": int(item["interval_end"]),
#                 "interval": list(item["interval"]),
#                 "one_state_init": {
#                     "log_amp": one_log_amp,
#                     "mass": one_mass,
#                 },
#                 "two_state_init": {
#                     "log_amp_1": two_log_amp_1,
#                     "mass_1": two_mass_1,
#                     "log_amp_2": two_log_amp_2,
#                     "mass_gap": two_mass_gap,
#                     "mass_2": two_mass_2,
#                 },
#                 "fit_status": bool(item.get("fit_status", False)),
#             }
#         )

#     intervals_out.sort(key=lambda x: (x["interval_start"], x["interval_end"]))
#     return {
#         "unit": unit,
#         "intervals": intervals_out,
#     }

def load_interval_initial_params(json_path: str | Path) -> dict[str, Any]:

    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    unit = payload.get("unit", {})
    one_state_results = payload.get("one_state_results", {})
    two_state_results = payload.get("two_state_results", {})
    
    intervals_out: list[dict[str, Any]] = []

    for result in one_state_results:
        interval_start = int(result["interval_start"])
        interval_end = int(result["interval_end"])
        interval = list(result["interval"])
        # one_log_amp = float(result["log_amp"])
        one_log_amp = float(result["fit_result"]["log_amp"])
        one_mass = float(result["fit_result"]["mass"])

        intervals_out.append(
            {   "type": "one_state",
                "interval_start": interval_start,
                "interval_end": interval_end,
                "interval": interval,
                "one_state_init": {
                    "log_amp": one_log_amp,
                    "mass": one_mass,
                }
            }
        )

    for result in two_state_results:
        interval_start = int(result["interval_start"])
        interval_end = int(result["interval_end"])
        interval = list(result["interval"])
        two_log_amp_1 = float(result["fit_result"]["log_amp_1"])
        two_mass_1 = float(result["fit_result"]["mass_1"])
        two_log_amp_2 = float(result["fit_result"]["log_amp_2"])
        two_mass_gap = float(result["fit_result"]["mass_gap"])
        two_mass_2 = two_mass_1 + two_mass_gap

        intervals_out.append(
            {   
                "type": "two_state",
                "interval_start": interval_start,
                "interval_end": interval_end,
                "interval": interval,
                "two_state_init": {
                    "log_amp_1": two_log_amp_1,
                    "mass_1": two_mass_1,
                    "log_amp_2": two_log_amp_2,
                    "mass_gap": two_mass_gap,
                    "mass_2": two_mass_2,
                }
            }
        )

    return {
        "unit": unit,
        "intervals": intervals_out,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Load per-p2 initial-guess JSON and print interval initial parameters."
    )
    parser.add_argument("json_file", type=Path, help="Path to p2_x_init_guess.json")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Print first K intervals after sorting by start/end.",
    )
    args = parser.parse_args()

    data = load_interval_initial_params(args.json_file)
    unit = data.get("unit", {})
    intervals = data.get("intervals", [])

    print("Loaded unit:", unit.get("p2_id", "unknown"))
    print("Source file:", unit.get("input_file", "unknown"))
    print("Total intervals:", len(intervals))
    print("")

    for idx, row in enumerate(intervals[: max(args.top_k, 1)], start=1):
        # print(row)
        
        print(f"Interval: [{row['interval_start']}, {row['interval_end']}]")
        if row["type"] == "one_state":
            one = row["one_state_init"]
            print(f"  One-state init: A={one['log_amp']:.6e}, m={one['mass']:.6e}")
        elif row["type"] == "two_state":
            two = row["two_state_init"]
            print(
                f"  Two-state init: A1={two['log_amp_1']:.6e}, m1={two['mass_1']:.6e}, "
                f"A2={two['log_amp_2']:.6e}, dm={two['mass_gap']:.6e}, m2={two['mass_2']:.6e}"
            )
        print("")


        # print(
        #     f"{idx:02d}. [{row['interval_start']},{row['interval_end']}] "
        #     f"one: (A={one['log_amp']:.6e}, m={one['mass']:.6e}) "
        #     f"two: (A1={two['log_amp_1']:.6e}, m1={two['mass_1']:.6e}, "
        #     f"A2={two['log_amp_2']:.6e}, dm={two['mass_gap']:.6e}, m2={two['mass_2']:.6e}) "
        #     f"fit_status={row['fit_status']}"
        # )


if __name__ == "__main__":
    _main()
    # results = load_interval_initial_params("data/processed/init_guess/p2_0_init_guess.json")
    # print(results)
