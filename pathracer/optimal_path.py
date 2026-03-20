import numpy as np
import skfmm

from .config import DEFAULT_CONFIG


class FMMSolver:

    def __init__(self, driveable, cfg=None):
        self.cfg = cfg or DEFAULT_CONFIG
        self.driveable = driveable
        ss = self.cfg.supersample

        # blow up the mask so the path isnt all jaggy and pixelated
        drive_hi = np.repeat(np.repeat(driveable, ss, 0), ss, 1)
        self.phi = np.where(drive_hi, -1.0, 1.0)
        self.T = skfmm.travel_time(self.phi, np.ones_like(self.phi))
        self.Ty, self.Tx = np.gradient(self.T)

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

            # basic euler step following the gradient downhill
            iy = min(max(int(p[0]), 0), self.T.shape[0] - 2)
            ix = min(max(int(p[1]), 0), self.T.shape[1] - 2)
            g = np.array([self.Ty[iy, ix], self.Tx[iy, ix]])
            g /= np.linalg.norm(g) + eps
            p -= h * g
            trace.append(p[::-1])

        return np.asarray(trace)[::-1] / ss
