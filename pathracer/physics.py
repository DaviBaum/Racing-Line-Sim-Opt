import numpy as np
from scipy.signal import savgol_filter

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

    # closer to the wall = slower and tighter curves = slower
    # this is the main thing that makes different paths have different times
    v_wall = cfg.v_base / (1 + cfg.wall_penalty / (dist_from_wall + eps)
                           + cfg.curv_penalty * kappa**2)

    # you also cant go faster than what the lateral g limit allows
    # v = sqrt(a_lat / kappa) basically centripetal acceleration
    v_lat = np.sqrt(np.maximum(0, cfg.a_lat_max / (kappa + eps)))
    v = np.minimum(v_wall, v_lat)

    # segment lengths between consecutive points
    ds = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1])) + eps

    # forward pass caps how fast you can speed up
    # basically v^2 = v0^2 + 2*a*ds but simplified
    for i in range(1, len(v)):
        dv = cfg.a_max * ds[i-1] / v[i-1]
        if v[i] > v[i-1] + dv:
            v[i] = v[i-1] + dv

    # same thing backwards so you slow down in time for corners
    for i in range(len(v) - 2, -1, -1):
        dv = cfg.a_max * ds[i] / v[i+1]
        if v[i] > v[i+1] + dv:
            v[i] = v[i+1] + dv

    # savgol smoothing to keep the jerk from going nuts
    # without this the speed profile has these nasty discontinuities
    v_lim = np.minimum(v_wall, v_lat)
    if len(v) >= cfg.j_window:
        v = savgol_filter(v, cfg.j_window, 3, mode="interp")
        v = np.minimum(v, v_lim)  # clamp back down after smoothing

    return v


def resample_time(path_xy, v_node, dt):
    # convert from spatial samples to time samples at uniform dt
    # average speed between consecutive nodes gives us time per segment
    # then just interpolate x and y onto the new time grid
    eps = DEFAULT_CONFIG.eps
    ds = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1])) + eps
    v_avg = 0.5 * (v_node[:-1] + v_node[1:])
    t_cum = np.concatenate(([0.0], np.cumsum(ds / v_avg)))
    t_new = np.arange(0.0, t_cum[-1] + 1e-9, dt)
    x = np.interp(t_new, t_cum, path_xy[:, 0])
    y = np.interp(t_new, t_cum, path_xy[:, 1])
    return np.column_stack([x, y]), t_cum[-1]


def compute_stats(path_xy, dist_from_wall, cfg=None):
    # just grabs the peak values for the summary printout
    if cfg is None:
        cfg = DEFAULT_CONFIG

    v = speed_profile(path_xy, dist_from_wall, cfg)
    ds = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1])) + cfg.eps
    v_avg = 0.5 * (v[:-1] + v[1:])
    dt_seg = ds / v_avg

    a = np.diff(v) / dt_seg
    jerk = np.diff(a) / dt_seg[1:] if len(a) > 1 else np.array([])
    lat_g = v_avg**2 * np.abs(curvature(path_xy)[:-1]) / 9.81

    return {
        "v_max": float(v.max()),
        "a_max": float(np.abs(a).max()) if len(a) else 0.0,
        "j_max": float(np.abs(jerk).max()) if len(jerk) else 0.0,
        "g_lat_max": float(lat_g.max()) if len(lat_g) else 0.0,
    }
