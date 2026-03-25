import numpy as np
from pathracer.physics import curvature


def test_curvature_straight_line():
    xy = np.column_stack([np.linspace(0, 100, 200), np.zeros(200)])
    k = curvature(xy)
    assert np.allclose(k, 0, atol=1e-6)


def test_curvature_circle():
    R = 50.0
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    xy = np.column_stack([R * np.cos(t), R * np.sin(t)])
    k = curvature(xy)
    # skip edges bc finite diffs are wonky there
    assert np.allclose(k[10:-10], 1.0 / R, atol=0.01)
