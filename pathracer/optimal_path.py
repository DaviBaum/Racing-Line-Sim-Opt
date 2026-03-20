import numpy as np
import skfmm

from .config import DEFAULT_CONFIG


class FMMSolver:

    def __init__(self, driveable, cfg=None):
        self.cfg = cfg or DEFAULT_CONFIG
        self.driveable = driveable
        ss = self.cfg.supersample

        drive_hi = np.repeat(np.repeat(driveable, ss, 0), ss, 1)
        # -1 = road  +1 = wall
        self.phi = np.where(drive_hi, -1.0, 1.0)
        # fmm gives us a travel time field and then we just roll downhill on it
        self.T = skfmm.travel_time(self.phi, np.ones_like(self.phi))
        self.Ty, self.Tx = np.gradient(self.T)

    def _grad_at(self, p):
        # rk4 lands between pixels so we gotta interpolate the gradient
        y, x = p
        iy = min(max(int(y), 0), self.T.shape[0] - 2)
        ix = min(max(int(x), 0), self.T.shape[1] - 2)
        dy, dx = y - iy, x - ix

        # lerp the corners
        g00 = np.array([self.Ty[iy, ix],     self.Tx[iy, ix]])
        g10 = np.array([self.Ty[iy+1, ix],   self.Tx[iy+1, ix]])
        g01 = np.array([self.Ty[iy, ix+1],   self.Tx[iy, ix+1]])
        g11 = np.array([self.Ty[iy+1, ix+1], self.Tx[iy+1, ix+1]])
        top = g00 * (1 - dy) + g10 * dy
        bot = g01 * (1 - dy) + g11 * dy
        return top * (1 - dx) + bot * dx

    def _push_onto_road(self, q):
        # if it went off road shove it back using the gradient as a normal
        eps = self.cfg.eps
        rc = np.clip(np.round(q[::-1]).astype(int), [0, 0],
                     np.array(self.phi.shape) - 1)
        phi_val = self.phi[rc[0], rc[1]]
        if phi_val > 0:
            n = self._grad_at(q[::-1])
            n /= np.linalg.norm(n) + eps
            return q - phi_val * n[::-1]
        return q

    def compute_optimal_path(self, start_xy, goal_xy, max_iters=25000):
        ss = self.cfg.supersample
        eps = self.cfg.eps

        # go from goal backwards to start bc thats how fmm descent works
        start_hi = np.array(start_xy[::-1]) * ss
        goal_hi = np.array(goal_xy[::-1]) * ss
        p = goal_hi.copy()
        trace = [p[::-1]]
        h = 0.5

        for _ in range(max_iters):
            if np.linalg.norm(p - start_hi) < 1.0:
                break

            # rk4 bc euler was way too sloppy on tight corners
            # normalize each k so we always step the same distance
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

        return self._relax(np.asarray(trace)[::-1])

    def _relax(self, path):
        # the raw fmm path is kinda crunchy so we smooth it out
        # treat it like a rubber band with beads on it and let it settle
        ss = self.cfg.supersample
        nb = self.cfg.string_beads
        t_in = np.linspace(0, 1, len(path))
        t_out = np.linspace(0, 1, nb)
        pts = np.column_stack([
            np.interp(t_out, t_in, path[:, 0]),
            np.interp(t_out, t_in, path[:, 1]),
        ])

        for _ in range(self.cfg.string_iters):
            # each bead gets pulled toward being a straight line between its neighbors
            segs = np.diff(pts, axis=0)
            lens = np.linalg.norm(segs, axis=1) + self.cfg.eps
            force = np.zeros_like(pts)
            force[1:-1] = segs[1:]/lens[1:, None] - segs[:-1]/lens[:-1, None]
            pts[1:-1] -= 0.25 * force[1:-1]
            # shove any beads that slid off the road back on
            pts = np.vstack([
                pts[0],
                [self._push_onto_road(p) for p in pts[1:-1]],
                pts[-1],
            ])

        return pts / ss
