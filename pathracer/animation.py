import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.collections import LineCollection

from .config import DEFAULT_CONFIG
from .physics import speed_profile


def create_race_animation(road_rgba, paths_raw, paths_timed, dist_map,
                          n_frames, cfg=None, output_path=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    h, w = road_rgba.shape[:2]

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_facecolor("k")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(road_rgba, zorder=0)

    cmap = plt.get_cmap("turbo")
    for _, raw in paths_raw.items():
        iy = np.clip(raw[:, 1].astype(int), 0, h - 1)
        ix = np.clip(raw[:, 0].astype(int), 0, w - 1)
        v = speed_profile(raw, dist_map[iy, ix], cfg)
        vnorm = v / v.max()
        segs = np.column_stack([raw[:-1], raw[1:]]).reshape(-1, 2, 2)
        ax.add_collection(
            LineCollection(segs, array=vnorm[:-1], cmap=cmap, linewidth=3, zorder=1)
        )

    markers = {}
    for n, pts in paths_timed.items():
        markers[n], = ax.plot(pts[0][0], pts[0][1], "o", markersize=12, zorder=3)

    def update(i):
        for n in markers:
            markers[n].set_data([paths_timed[n][i][0]], [paths_timed[n][i][1]])
        return markers.values()

    # blit=True makes a huge difference was like 5fps without it
    result = anim.FuncAnimation(fig, update, frames=n_frames,
                                interval=cfg.dt * 1000, blit=True)
    if output_path:
        print(f"Saving to {output_path}...")
        result.save(output_path, fps=cfg.fps, dpi=150)

    return result
