import numpy as np
from pathracer.physics import curvature, speed_profile, resample_time
from pathracer.config import SimConfig


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


def test_speed_profile_respects_limits():
    cfg = SimConfig(v_base=1.0)
    xy = np.column_stack([np.linspace(0, 200, 300), np.zeros(300)])
    dist = np.full(300, 50.0)
    v = speed_profile(xy, dist, cfg)
    assert v.max() <= cfg.v_base + 0.01


def test_speed_profile_wall_slowdown():
    cfg = SimConfig()
    xy = np.column_stack([np.linspace(0, 100, 200), np.zeros(200)])
    v_far = speed_profile(xy, np.full(200, 100.0), cfg)
    v_near = speed_profile(xy, np.full(200, 1.0), cfg)
    assert v_far.mean() > v_near.mean()


def test_resample_time_output_shape():
    xy = np.column_stack([np.linspace(0, 100, 200), np.zeros(200)])
    v = np.full(200, 1.0)
    dt = 0.025
    resampled, total_time = resample_time(xy, v, dt)
    expected = int(total_time / dt) + 1
    assert abs(len(resampled) - expected) <= 1
