#!/usr/bin/env python3
"""
dream_benchmark.py

Publication-grade offline benchmark for the DREAM event-triggered odometry
sampler against four non-trivial baselines, with Monte-Carlo statistics.

Baselines:
  1. uniform          : fixed-rate sampling (1 Hz).
  2. douglas_peucker  : 2-D polyline simplification (tolerance eps, m).
  3. send_on_delta    : position deadband / send-on-delta (threshold delta, m).
  4. nyquist          : the paper's 0.5 Hz straight / 1.0 Hz curved heuristic.
  5. dream            : the proposed R^2-triggered sampler.

For every method we report, at one natural operating point, the mean and 95%
confidence interval (over Monte-Carlo noise realisations) of:

  * N          : number of recorded landmarks,
  * DR         : data reduction ratio vs send-everything (1 - N / N_total),
  * psi_RMSE   : heading reconstruction RMSE (linear interpolation), rad,
  * path_max   : max distance of the true path from the reconstructed polyline,
  * path_mean  : mean distance of the true path from the reconstructed polyline.

The DREAM decision rule is imported from dream_core.py, i.e. the same code the
live ROS 2 node uses.

Usage:
    python3 dream_benchmark.py [--runs 30] [--profile sharp|gentle]
"""

import argparse
import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "src", "vslam_recovery_core", "src"))

import dream_core  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic odometry trajectory (shared with test_dream_sampler.py)
# ---------------------------------------------------------------------------
def generate_trajectory(dt=0.02, noise_psi=0.0, profile="sharp", seed=12345):
    """Return a list of (t, psi, s, x, y, v, w) samples."""
    if profile == "sharp":
        phases = [(3.0, 0.20, 0.0), (2.0, 0.10, 0.4),
                  (2.0, 0.10, -0.4), (5.0, 0.20, 0.0)]
    elif profile == "gentle":
        phases = [(3.0, 0.20, 0.0), (3.0, 0.10, 0.15),
                  (3.0, 0.10, -0.15), (11.0, 0.20, 0.0)]
    else:
        raise ValueError(f"unknown profile {profile!r}")

    rng = _LCG(seed)
    samples = []
    t = 0.0
    x = 0.0
    y = 0.0
    psi = 0.0
    s = 0.0
    prev_x, prev_y = 0.0, 0.0
    for dur, v, w in phases:
        for _ in range(int(round(dur / dt))):
            psi += w * dt
            x += v * math.cos(psi) * dt
            y += v * math.sin(psi) * dt
            s += math.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y
            t += dt
            psi_obs = psi + (noise_psi * rng.gauss() if noise_psi > 0 else 0.0)
            samples.append((t, psi_obs, s, x, y, v, w))
    return samples


class _LCG:
    """Small deterministic LCG + Box-Muller Gaussian (reproducible runs)."""

    def __init__(self, seed):
        self.state = seed & 0x7FFFFFFF
        self.spare = None

    def uniform(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF

    def gauss(self):
        if self.spare is not None:
            v = self.spare
            self.spare = None
            return v
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self.spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Samplers: each returns a sorted list of sample indices
# ---------------------------------------------------------------------------
def uniform_sample(samples, period=1.0):
    """Fixed-rate sampling."""
    idx = []
    last_t = -1e9
    for i, (t, *_rest) in enumerate(samples):
        if t - last_t >= period:
            idx.append(i)
            last_t = t
    return idx


def _is_curved(yaws, window_size=4):
    if len(yaws) < window_size:
        return False
    d0 = yaws[1] - yaws[0]
    d1 = yaws[2] - yaws[1]
    d2 = yaws[3] - yaws[2]
    if (d0 > 0 and d1 > 0 and d2 > 0) or (d0 < 0 and d1 < 0 and d2 < 0):
        return True
    return any(math.pi <= abs(d) <= 2.0 * math.pi for d in (d0, d1, d2))


def nyquist_sample(samples, straight_period=2.0, curve_period=1.0):
    """The paper's adaptive 0.5 Hz / 1.0 Hz heuristic (as landmark_sampler_client)."""
    idx = []
    yaw_window = deque(maxlen=4)
    period = straight_period
    elapsed = 0.0
    last_t = None
    for i, (t, psi, *_rest) in enumerate(samples):
        yaw_window.append(psi)
        if last_t is not None:
            elapsed += t - last_t
        last_t = t
        period = curve_period if _is_curved(yaw_window) else straight_period
        if elapsed >= period:
            elapsed = 0.0
            idx.append(i)
    return idx


def douglas_peucker(samples, eps=0.05):
    """2-D polyline simplification on the (x, y) path (recursive)."""
    pts = [(s[3], s[4]) for s in samples]
    n = len(pts)
    if n < 3:
        return list(range(n))

    keep = [False] * n
    keep[0] = keep[-1] = True

    def _recurse(lo, hi):
        if hi <= lo + 1:
            return
        ax, ay = pts[lo]
        bx, by = pts[hi]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        # distance from point p to segment a-b
        def dist(i):
            if denom < 1e-12:
                return math.hypot(pts[i][0] - ax, pts[i][1] - ay)
            u = ((pts[i][0] - ax) * dx + (pts[i][1] - ay) * dy) / denom
            u = min(max(u, 0.0), 1.0)
            return math.hypot(pts[i][0] - (ax + u * dx),
                              pts[i][1] - (ay + u * dy))
        m = max(range(lo + 1, hi), key=dist)
        if dist(m) > eps:
            keep[m] = True
            _recurse(lo, m)
            _recurse(m, hi)

    _recurse(0, n - 1)
    return [i for i in range(n) if keep[i]]


def send_on_delta(samples, delta=0.05):
    """Position deadband / send-on-delta: emit when displacement >= delta."""
    idx = [0]
    lx, ly = samples[0][3], samples[0][4]
    for i, (t, psi, s, x, y, v, w) in enumerate(samples):
        if math.hypot(x - lx, y - ly) >= delta:
            idx.append(i)
            lx, ly = x, y
    return idx


def dream_sample(samples, r2_threshold=0.98, min_window_size=5,
                 max_window_size=500, min_event_spacing=0.2):
    """Proposed R^2-triggered sampler (imported core)."""
    sampler = dream_core.DreamSampler(
        r2_threshold=r2_threshold, min_window_size=min_window_size,
        max_window_size=max_window_size, regress_speed=True,
        min_event_spacing=min_event_spacing)
    idx = []
    for i, (t, psi, s, x, y, v, w) in enumerate(samples):
        if sampler.update(t, x, y, s):
            idx.append(i)
    return idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def heading_rmse(samples, idx):
    """RMSE of psi(t) reconstructed by linear interpolation between selected
    samples."""
    if len(idx) < 2:
        return float("inf")
    err2 = 0.0
    n = 0
    j = 0
    for i, (t, psi, *_rest) in enumerate(samples):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        frac = (i - i0) / (i1 - i0) if i1 > i0 else 0.0
        psi_hat = samples[i0][1] + frac * (samples[i1][1] - samples[i0][1])
        err2 += (psi_hat - psi) ** 2
        n += 1
    return math.sqrt(err2 / n) if n else float("inf")


def path_error(samples, idx):
    """Return (max, mean) distance of the true path from the polyline through
    the selected (x, y) points."""
    if len(idx) < 2:
        return (float("inf"), float("inf"))
    seg = [(samples[i][3], samples[i][4]) for i in idx]
    worst = 0.0
    total = 0.0
    n = 0
    for (t, psi, s, x, y, v, w) in samples:
        d = _dist_to_polyline(x, y, seg)
        worst = max(worst, d)
        total += d
        n += 1
    return worst, (total / n if n else 0.0)


def _dist_to_polyline(x, y, seg):
    best = float("inf")
    for (ax, ay), (bx, by) in zip(seg, seg[1:]):
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            u = 0.0
        else:
            u = ((x - ax) * dx + (y - ay) * dy) / denom
        u = min(max(u, 0.0), 1.0)
        best = min(best, math.hypot(x - (ax + u * dx), y - (ay + u * dy)))
    return best


# ---------------------------------------------------------------------------
# Monte-Carlo benchmark
# ---------------------------------------------------------------------------
def run_method(samples, method):
    idx = method(samples)
    # The recovery layer always records the failure point (the last sample),
    # so every method's landmark set must include the trajectory endpoint for a
    # fair reconstruction comparison over the full path.
    n_total = len(samples)
    if not idx or idx[-1] != n_total - 1:
        idx = idx + [n_total - 1]
    n = len(idx)
    dr = 1.0 - n / n_total
    psi = heading_rmse(samples, idx)
    pmax, pmean = path_error(samples, idx)
    return {"N": n, "DR": dr, "psi_RMSE": psi,
            "path_max": pmax, "path_mean": pmean}


def mean_ci(values, z=1.96):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(var / len(values))
    return m, z * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--profile", default="gentle")
    ap.add_argument("--noise", type=float, default=0.01)
    args = ap.parse_args()

    methods = {
        "uniform (1 Hz)": lambda s: uniform_sample(s, 1.0),
        "douglas-peucker (0.05 m)": lambda s: douglas_peucker(s, 0.05),
        "send-on-delta (0.05 m)": lambda s: send_on_delta(s, 0.05),
        "nyquist (0.5/1.0 Hz)": nyquist_sample,
        "dream (R2_th=0.98)": lambda s: dream_sample(s, r2_threshold=0.98),
    }

    accum = {name: {k: [] for k in ("N", "DR", "psi_RMSE", "path_max", "path_mean")}
             for name in methods}

    for run in range(args.runs):
        samples = generate_trajectory(noise_psi=args.noise,
                                      profile=args.profile, seed=1000 + run)
        for name, fn in methods.items():
            res = run_method(samples, fn)
            for k in accum[name]:
                accum[name][k].append(res[k])

    print("=" * 78)
    print(f"DREAM vs baselines  (profile={args.profile}, noise_sigma={args.noise}, "
          f"{args.runs} runs)")
    print("=" * 78)
    header = (f"{'method':<26} | {'N':>6} | {'DR':>6} | "
              f"{'psi_RMSE':>9} | {'path_max':>9} | {'path_mean':>9}")
    print(header)
    print("-" * 78)
    for name in methods:
        n_m, n_c = mean_ci(accum[name]["N"])
        dr_m, dr_c = mean_ci(accum[name]["DR"])
        pr_m, pr_c = mean_ci(accum[name]["psi_RMSE"])
        pm_m, pm_c = mean_ci(accum[name]["path_max"])
        pn_m, pn_c = mean_ci(accum[name]["path_mean"])
        print(f"{name:<26} | {n_m:6.1f} | {dr_m:6.1%} | "
              f"{pr_m:9.4f} | {pm_m:9.4f} | {pn_m:9.4f}")
        print(f"{'':<26} | {'±'+str(round(n_c,1)):>6} | "
              f"{'±'+str(round(dr_c*100,1))+'%':>6} | "
              f"{'±'+str(round(pr_c,4)):>9} | "
              f"{'±'+str(round(pm_c,4)):>9} | {'±'+str(round(pn_c,4)):>9}")
    print("-" * 78)
    print("(second row per method = 95% confidence interval)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
