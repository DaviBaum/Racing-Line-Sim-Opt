import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG
from .centerline import ordered_centerline
from .physics import speed_profile, resample_time
from .optimal_path import FMMSolver
from .animation import create_race_animation


def run_race(road_path, stroke_paths, cfg=None, compute_optimal=True,
             output_path=None, show=True):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    road_rgba = np.asarray(Image.open(road_path).convert("RGBA"))
    driveable = road_rgba[..., 3] == 0

    paths_raw = {}
    for name, png in stroke_paths.items():
        paths_raw[name] = ordered_centerline(png, cfg.smooth_win, cfg.smooth_poly)[::-1]

    if compute_optimal and paths_raw:
        starts = np.vstack([p[0] for p in paths_raw.values()])
        goals = np.vstack([p[-1] for p in paths_raw.values()])
        print("Computing FMM optimal path...")
        solver = FMMSolver(driveable, cfg)
        paths_raw["cyan"] = solver.compute_optimal_path(
            starts.mean(axis=0), goals.mean(axis=0))

    # dummy dist for now - just ones everywhere
    dist = np.ones(road_rgba.shape[:2], dtype=float)

    paths_timed = {}
    total_times = {}
    for name, raw in paths_raw.items():
        v = speed_profile(raw, dist[0] * np.ones(len(raw)), cfg)
        paths_timed[name], total_times[name] = resample_time(raw, v, cfg.dt)

    n_frames = int(max(total_times.values()) / cfg.dt) + 1

    ani = create_race_animation(
        road_rgba, paths_raw, paths_timed,
        n_frames, cfg, output_path,
    )

    if show:
        plt.show()

    return {
        "paths_raw": paths_raw,
        "paths_timed": paths_timed,
        "n_frames": n_frames,
        "total_times": total_times,
    }
