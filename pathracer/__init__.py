"""PathRacer - Physics-based path racing simulator."""

from .config import SimConfig, DEFAULT_CONFIG
from .centerline import ordered_centerline
from .physics import curvature, speed_profile, resample_time, compute_stats
from .optimal_path import FMMSolver
from .animation import create_race_animation
from .pipeline import run_race
