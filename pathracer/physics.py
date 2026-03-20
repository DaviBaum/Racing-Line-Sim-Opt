import numpy as np

from .config import DEFAULT_CONFIG


def curvature(xy):
    # classic formula for signed curvature from x' y'' - y' x'' over speed^1.5
    # just using finite diffs here bc the points are close enough
    d1 = np.gradient(xy, axis=0)
    d2 = np.gradient(d1, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denom = (d1[:, 0]**2 + d1[:, 1]**2)**1.5 + DEFAULT_CONFIG.eps
    return cross / denom


def speed_profile(path_xy, dist_from_wall, cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    eps = cfg.eps

    kappa = np.abs(curvature(path_xy))

    # tighter curves = slower
    v = cfg.v_base / (1 + cfg.curv_penalty * kappa**2)

    # lateral g limit
    v_lat = np.sqrt(np.maximum(0, cfg.a_lat_max / (kappa + eps)))
    v = np.minimum(v, v_lat)

    # segment lengths between consecutive points
    ds = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1])) + eps

    # forward pass caps how fast you can speed up
    for i in range(1, len(v)):
        dv = cfg.a_max * ds[i-1] / v[i-1]
        if v[i] > v[i-1] + dv:
            v[i] = v[i-1] + dv

    # same thing backwards so you slow down in time for corners
    for i in range(len(v) - 2, -1, -1):
        dv = cfg.a_max * ds[i] / v[i+1]
        if v[i] > v[i+1] + dv:
            v[i] = v[i+1] + dv

    return v
