#!/usr/bin/env python3
"""PathRacer CLI - Run a physics-based path race from the command line."""

import argparse
import sys
from pathracer import run_race, SimConfig


def main():
    parser = argparse.ArgumentParser(
        description="PathRacer: Physics-based path racing simulator",
    )
    parser.add_argument(
        "--road", required=True,
        help="Path to the road/track PNG (transparent = driveable area)",
    )
    parser.add_argument(
        "--paths", nargs="+", required=True,
        metavar="NAME=FILE",
        help="Hand-drawn stroke PNGs as name=path pairs",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output MP4 file path (optional)",
    )
    parser.add_argument(
        "--no-optimal", action="store_true",
        help="Skip computing the FMM optimal (cyan) path",
    )

    args = parser.parse_args()

    stroke_paths = {}
    for item in args.paths:
        parts = item.split("=")
        name, path = parts[0], parts[1]
        stroke_paths[name] = path

    cfg = SimConfig()

    run_race(
        road_path=args.road,
        stroke_paths=stroke_paths,
        cfg=cfg,
        compute_optimal=not args.no_optimal,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
