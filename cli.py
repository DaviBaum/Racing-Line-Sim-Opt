#!/usr/bin/env python3
"""PathRacer CLI - Run a physics-based path race from the command line."""

import argparse
import sys
from pathracer import run_race, SimConfig


def main():
    parser = argparse.ArgumentParser(
        description="PathRacer: Physics-based path racing simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python cli.py --road examples/inputs/road_map.png \\
                --paths yellow=examples/inputs/path_yellow.png \\
                        green=examples/inputs/path_green.png \\
                        pink=examples/inputs/path_pink.png \\
                --output race.mp4
        """,
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
    parser.add_argument("--fps", type=int, default=40, help="Frames per second")
    parser.add_argument("--no-show", action="store_true", help="Don't display the animation window")

    args = parser.parse_args()

    # Parse name=file pairs
    stroke_paths = {}
    for item in args.paths:
        if "=" not in item:
            print(f"Error: --paths expects NAME=FILE format, got: {item}", file=sys.stderr)
            sys.exit(1)
        name, path = item.split("=", 1)
        stroke_paths[name] = path

    cfg = SimConfig(fps=args.fps)

    run_race(
        road_path=args.road,
        stroke_paths=stroke_paths,
        cfg=cfg,
        compute_optimal=not args.no_optimal,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
