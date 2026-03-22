import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage.graph import route_through_array

from .config import DEFAULT_CONFIG

# 3x3 kernel for counting neighbors minus the center
_NEIGHBOR_KERN = np.ones((3, 3), dtype=int)
_NEIGHBOR_KERN[1, 1] = 0


def _find_endpoints(skel):
    neighbor_count = convolve(skel.astype(int), _NEIGHBOR_KERN, mode="constant")
    ys, xs = np.where(skel & (neighbor_count == 1))
    return list(zip(ys, xs))


def ordered_centerline(png_path, smooth_win=None, smooth_poly=None):
    smooth_win = smooth_win or DEFAULT_CONFIG.smooth_win
    smooth_poly = smooth_poly or DEFAULT_CONFIG.smooth_poly

    rgba = np.asarray(Image.open(png_path).convert("RGBA"))
    mask = rgba[..., 3] > 0  # non transparent = stroke

    lbl, n = label(mask, connectivity=2, return_num=True)
    if n == 0:
        raise ValueError(f"No painted pixels in {png_path}")
    # keep only the biggest blob or stray pixels mess up skeletonize
    biggest = np.argmax(np.bincount(lbl.flat)[1:]) + 1
    mask = (lbl == biggest)

    skel = skeletonize(mask)
    deg1 = _find_endpoints(skel)

    # fallback to all skeleton coords if endpoints are weird
    if len(deg1) >= 2:
        ends = deg1
    else:
        ys, xs = np.where(skel)
        ends = list(zip(ys, xs))
    if len(ends) < 2:
        raise ValueError(f"Skeleton too small to trace in {png_path}")

    ends_arr = np.array(ends)
    D = cdist(ends_arr, ends_arr)
    idx = np.argmax(D)
    i, j = idx // len(ends), idx % len(ends)
    (y1, x1), (y2, x2) = ends[i], ends[j]

    cost = np.where(skel, 1.0, 1e6)
    path_idx, _ = route_through_array(cost, (y1, x1), (y2, x2),
                                      fully_connected=True, geometric=True)
    pts = np.array(path_idx, dtype=float)
    pts = pts[:, ::-1]  # flip to x y

    if len(pts) >= smooth_win:
        pts[:, 0] = savgol_filter(pts[:, 0], smooth_win, smooth_poly)
        pts[:, 1] = savgol_filter(pts[:, 1], smooth_win, smooth_poly)

    return pts
