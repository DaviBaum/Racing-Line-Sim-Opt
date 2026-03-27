import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .config import DEFAULT_CONFIG
from .centerline import ordered_centerline
from .physics import speed_profile, resample_time, compute_stats
from .optimal_path import FMMSolver
from .animation import create_race_animation


def load_road(road_path):
    road_rgba = np.asarray(Image.open(road_path).convert("RGBA"))
    # alpha 0 = road everything else is wall
    driveable = road_rgba[..., 3] == 0
    # how far each pixel is from a wall used for speed penalties
    dist = distance_transform_edt(driveable).astype(float)
    return road_rgba, driveable, dist


def run_race(road_path, stroke_paths, cfg=None, compute_optimal=True,
             output_path=None, show=True):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    road_rgba, driveable, dist = load_road(road_path)

    # pull centerlines out of the strokes and flip so its start to finish
    paths_raw = {}
    for name, png in stroke_paths.items():
        paths_raw[name] = ordered_centerline(png, cfg.smooth_win, cfg.smooth_poly)[::-1]

    if compute_optimal and paths_raw:
        # avg everyones start and end to get a common origin then fmm does its thing
        starts = np.vstack([p[0] for p in paths_raw.values()])
        goals = np.vstack([p[-1] for p in paths_raw.values()])
        print("Computing FMM optimal path...")
        solver = FMMSolver(driveable, cfg)
        paths_raw["cyan"] = solver.compute_optimal_path(
            starts.mean(axis=0), goals.mean(axis=0))

    # run physics on each path then resample to even time steps
    paths_timed = {}
    total_times = {}
    stats = {}
    for name, raw in paths_raw.items():
        # clip to image bounds bc path coords can be slightly off
        iy = np.clip(raw[:, 1].astype(int), 0, dist.shape[0] - 1)
        ix = np.clip(raw[:, 0].astype(int), 0, dist.shape[1] - 1)
        d = dist[iy, ix]
        v = speed_profile(raw, d, cfg)
        paths_timed[name], total_times[name] = resample_time(raw, v, cfg.dt)
        stats[name] = compute_stats(raw, d, cfg)

    n_frames = int(max(total_times.values()) / cfg.dt) + 1

    print("\nPhysics summary (peak values):")
    for n, s in stats.items():
        print(f"  {n}: v={s['v_max']:.2f}  a={s['a_max']:.2f}  "
              f"j={s['j_max']:.2f}  lat_g={s['g_lat_max']:.2f}")

    # put cyan last so hand drawn ones show on top
    order = [k for k in paths_raw if k != "cyan"]
    if "cyan" in paths_raw:
        order.append("cyan")

    ani = create_race_animation(
        road_rgba,
        {k: paths_raw[k] for k in order},
        {k: paths_timed[k] for k in order},
        dist, n_frames, cfg, output_path,
    )

    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return {
        "paths_raw": paths_raw,
        "paths_timed": paths_timed,
        "stats": stats,
        "n_frames": n_frames,
        "total_times": total_times,
    }
