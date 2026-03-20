import numpy as np
import skfmm

from .config import DEFAULT_CONFIG


class FMMSolver:

    def __init__(self, driveable, cfg=None):
        self.cfg = cfg or DEFAULT_CONFIG
        self.driveable = driveable
        ss = self.cfg.supersample

        drive_hi = np.repeat(np.repeat(driveable, ss, 0), ss, 1)
        self.phi = np.where(drive_hi, -1.0, 1.0)
        self.T = skfmm.travel_time(self.phi, np.ones_like(self.phi))
        self.Ty, self.Tx = np.gradient(self.T)

    def _grad_at(self, p):
        iy = min(max(int(p[0]), 0), self.T.shape[0] - 2)
        ix = min(max(int(p[1]), 0), self.T.shape[1] - 2)
        return np.array([self.Ty[iy, ix], self.Tx[iy, ix]])

    def compute_optimal_path(self, start_xy, goal_xy, max_iters=25000):
        ss = self.cfg.supersample
        eps = self.cfg.eps

        start_hi = np.array(start_xy[::-1]) * ss
        goal_hi = np.array(goal_xy[::-1]) * ss
        p = goal_hi.copy()
        trace = [p[::-1]]
        h = 0.5

        for _ in range(max_iters):
            if np.linalg.norm(p - start_hi) < 1.0:
                break

            # rk4 bc euler was way too sloppy on tight corners
            k1 = self._grad_at(p)
            k1 /= np.linalg.norm(k1) + eps
            k2 = self._grad_at(p - .5*h*k1)
            k2 /= np.linalg.norm(k2) + eps
            k3 = self._grad_at(p - .5*h*k2)
            k3 /= np.linalg.norm(k3) + eps
            k4 = self._grad_at(p - h*k3)
            k4 /= np.linalg.norm(k4) + eps
            p -= h * (k1 + 2*k2 + 2*k3 + k4) / 6
            trace.append(p[::-1])

        return np.asarray(trace)[::-1] / ss
