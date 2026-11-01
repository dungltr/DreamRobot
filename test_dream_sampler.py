#!/usr/bin/env python3
"""
test_dream_sampler.py

Offline, ROS-free benchmark of the DREAM event-triggered sampler against the
baseline Shannon-Nyquist heuristic sampler (0.5 Hz straight / 1.0 Hz curved).

It generates a representative differential-drive odometry trajectory (straight
+ obstacle-avoidance turns + straight), runs both samplers over it, and reports:

  * N_baseline / N_dream  : number of recorded landmarks,
  * DR                    : data reduction ratio = 1 - N_dream / N_baseline,
  * psi RMSE              : heading reconstruction error after linear
                            interpolation between the recorded landmarks.

The DREAM decision logic is imported from dream_core.py (the same code the live
ROS 2 node uses), so these numbers characterise the real algorithm rather than
a simplified copy.

Usage:
    python3 test_dream_sampler.py
"""

import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "src", "vslam_recovery_core", "src"))

import dream_core  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic odometry trajectory
# ---------------------------------------------------------------------------
def generate_trajectory(dt=0.02, noise_psi=0.0, profile="sharp"):
    """Return a list of (t, psi, s, x, y, v, w) samples.

    profile="sharp": straight -> left turn -> right turn -> straight
        (representative of an aggressive obstacle-avoidance slalom).
    profile="gentle": straight -> gentle S-curve -> long straight exit
        (representative of the paper's 1.7 m corridor with a long return).
    """
    if profile == "sharp":
        phases = [
            (3.0, 0.20, 0.0),    # straight, 0.2 m/s
            (2.0, 0.10, 0.4),    # left turn, 0.4 rad/s
            (2.0, 0.10, -0.4),   # right turn, -0.4 rad/s
            (5.0, 0.20, 0.0),    # straight, 0.2 m/s
        ]
    elif profile == "gentle":
        phases = [
            (3.0, 0.20, 0.0),    # straight
            (3.0, 0.10, 0.15),   # gentle left turn
            (3.0, 0.10, -0.15),  # gentle right turn
            (11.0, 0.20, 0.0),   # LONG straight exit
        ]
    else:
        raise ValueError(f"unknown profile {profile!r}")
    samples = []
    t = 0.0
    x = 0.0
    y = 0.0
    psi = 0.0
    s = 0.0
    prev_x, prev_y = 0.0, 0.0
    for dur, v, w in phases:
        steps = int(round(dur / dt))
        for _ in range(steps):
            psi += w * dt
            x += v * math.cos(psi) * dt
            y += v * math.sin(psi) * dt
            s += math.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y
            t += dt
            psi_obs = psi + (noise_psi * _gauss() if noise_psi > 0 else 0.0)
            samples.append((t, psi_obs, s, x, y, v, w))
    return samples


# A tiny deterministic Gaussian (Box-Muller) so runs are reproducible.
_GAUSS_SPARE = None


def _gauss():
    global _GAUSS_SPARE
    if _GAUSS_SPARE is not None:
        v = _GAUSS_SPARE
        _GAUSS_SPARE = None
        return v
    u1 = 1.0 - 1e-9
    u2 = 1e-9
    while u1 <= 1e-9:
        u1 = _uniform()
    u2 = _uniform()
    r = math.sqrt(-2.0 * math.log(u1))
    _GAUSS_SPARE = r * math.sin(2.0 * math.pi * u2)
    return r * math.cos(2.0 * math.pi * u2)


_SEED = 12345


def _uniform():
    global _SEED
    _SEED = (1103515245 * _SEED + 12345) & 0x7FFFFFFF
    return _SEED / 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Baseline Shannon-Nyquist heuristic sampler (landmark_sampler_client logic)
# ---------------------------------------------------------------------------
def baseline_sample(samples, straight_period=2.0, curve_period=1.0,
                    window_size=4):
    """Replicate the baseline adaptive sampler; return [(t, psi, s), ...]."""
    landmarks = []
    yaw_window = deque(maxlen=window_size)
    period = straight_period
    elapsed = 0.0
    last_t = None

    for (t, psi, s, _x, _y, _v, _w) in samples:
        yaw_window.append(psi)
        if last_t is not None:
            elapsed += t - last_t
        last_t = t

        curved = _is_curved(yaw_window, window_size)
        period = curve_period if curved else straight_period

        if elapsed >= period:
            elapsed = 0.0
            landmarks.append((t, psi, s))
    return landmarks


def _is_curved(yaw_window, window_size):
    if len(yaw_window) < window_size:
        return False
    yaws = list(yaw_window)
    d0 = yaws[1] - yaws[0]
    d1 = yaws[2] - yaws[1]
    d2 = yaws[3] - yaws[2]
    if (d0 > 0 and d1 > 0 and d2 > 0) or (d0 < 0 and d1 < 0 and d2 < 0):
        return True
    for d in (d0, d1, d2):
        if math.pi <= abs(d) <= 2.0 * math.pi:
            return True
    return False


# ---------------------------------------------------------------------------
# DREAM sampler (imported core)
# ---------------------------------------------------------------------------
def dream_sample(samples, r2_threshold=0.98, min_window_size=5,
                 max_window_size=500, regress_speed=True,
                 min_event_spacing=0.2):
    """Run DreamSampler over the trajectory; return [(t, psi, s), ...]."""
    sampler = dream_core.DreamSampler(
        r2_threshold=r2_threshold,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        regress_speed=regress_speed,
        min_event_spacing=min_event_spacing)
    landmarks = []
    for (t, psi, s, x, y, _v, _w) in samples:
        if sampler.update(t, x, y, s):
            landmarks.append((t, psi, s))
    return landmarks


# ---------------------------------------------------------------------------
# Heading reconstruction RMSE (linear interpolation between landmarks)
# ---------------------------------------------------------------------------
def heading_rmse(samples, landmarks):
    """Reconstruct psi(t) by linear interpolation and return RMSE vs truth."""
    if len(landmarks) < 2:
        return float("inf")
    err2 = 0.0
    n = 0
    j = 0
    for (t, psi, _s, _x, _y, _v, _w) in samples:
        while j + 1 < len(landmarks) and landmarks[j + 1][0] <= t:
            j += 1
        if j + 1 >= len(landmarks):
            break
        t0, p0, _s0 = landmarks[j]
        t1, p1, _s1 = landmarks[j + 1]
        span = t1 - t0
        frac = 0.0 if span <= 1e-9 else (t - t0) / span
        frac = min(max(frac, 0.0), 1.0)
        psi_hat = p0 + frac * (p1 - p0)
        err2 += (psi_hat - psi) ** 2
        n += 1
    return math.sqrt(err2 / n) if n else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_scenario("sharp", generate_trajectory(dt=0.02, profile="sharp"))
    run_scenario("gentle", generate_trajectory(dt=0.02, profile="gentle"))
    return 0


def run_scenario(name, samples):
    total_t = samples[-1][0]
    base = baseline_sample(samples)
    print("=" * 72)
    print(f"Scenario: {name}  ({total_t:.1f} s, {len(samples)} samples)")
    print("=" * 72)
    print(f"Baseline landmarks : {len(base)}  (0.5 Hz straight / 1.0 Hz curved)")

    print("\nDREAM threshold sweep (heading + displacement R^2):")
    print("-" * 72)
    print(f"{'R2_th':>7} | {'N_dream':>7} | {'DR':>7} | "
          f"{'psi RMSE':>9} | {'base RMSE':>9}")
    print("-" * 72)
    base_rmse = heading_rmse(samples, base)
    for r2_th in (0.85, 0.90, 0.95, 0.98, 1.00):
        dm = dream_sample(samples, r2_threshold=r2_th)
        dr = 1.0 - len(dm) / len(base) if base else 0.0
        rmse = heading_rmse(samples, dm)
        print(f"{r2_th:>7.2f} | {len(dm):>7d} | {dr:>6.1%} | "
              f"{rmse:>9.4f} | {base_rmse:>9.4f}")
    print("-" * 72)
    print(f"Baseline psi RMSE: {base_rmse:.4f} rad")
    print()


if __name__ == "__main__":
    sys.exit(main())
