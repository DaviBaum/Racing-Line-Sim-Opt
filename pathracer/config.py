from dataclasses import dataclass


@dataclass
class SimConfig:
    fps: int = 40
    v_base: float = 120.0
    wall_penalty: float = 3.0
    curv_penalty: float = 0.40
    a_max: float = 300.0
    a_lat_max: float = 500.0
    supersample: int = 8
    smooth_win: int = 11         # must be odd (savgol)
    smooth_poly: int = 3
    eps: float = 1e-6


DEFAULT_CONFIG = SimConfig()
