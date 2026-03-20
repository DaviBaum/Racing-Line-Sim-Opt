import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from .config import DEFAULT_CONFIG


def create_race_animation(road_rgba, paths_raw, paths_timed,
                          n_frames, cfg=None, output_path=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    h, w = road_rgba.shape[:2]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_facecolor("k")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(road_rgba, zorder=0)

    # just draw each path as a plain line for now
    for name, raw in paths_raw.items():
        ax.plot(raw[:, 0], raw[:, 1], linewidth=2, zorder=1)

    markers = {}
    for n, pts in paths_timed.items():
        markers[n], = ax.plot(pts[0][0], pts[0][1], "o", markersize=10, zorder=3)

    def update(i):
        for n in markers:
            markers[n].set_data([paths_timed[n][i][0]], [paths_timed[n][i][1]])
        return markers.values()

    result = anim.FuncAnimation(fig, update, frames=n_frames,
                                interval=cfg.dt * 1000)
    if output_path:
        print(f"Saving to {output_path}...")
        result.save(output_path, fps=cfg.fps, dpi=150)

    return result
