#!/usr/bin/env python3
"""
benchmark_strategy_comparison.py

Comprehensive offline comparison of the two DREAM sampling strategies on the
recorded Gazebo odometry (bench_data/odom_{A,B,C}.csv):

  * geometry : path linearity (x, y) + speed linearity s(t)   [original]
  * velocity : linear velocity v(t) + angular velocity w(t)   [new, 2 trackers]

For every world it reports, per strategy, the landmark count N, the data
reduction vs the 0.5/1.0 Hz two-rate baseline, the closed-loop recovery error
PE, the maximum path reconstruction error, and the heading reconstruction
error (psi-RMSE).

  --mc          Monte-Carlo noise re-integration (mean +/- 95% CI)
  --runs N      number of Monte-Carlo runs (default 20)

Usage:
  python3 benchmark_strategy_comparison.py            # clean single run
  python3 benchmark_strategy_comparison.py --mc       # Monte-Carlo comparison
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import deque

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src', 'vslam_recovery_core', 'src'))

import dream_core  # noqa: E402

R2_TH = 0.98
MAX_WINDOW = 500


# ---------------------------------------------------------------------------
# Reconstruction (keeps angular velocity w, unlike run_gazebo_benchmark).
# ---------------------------------------------------------------------------
def reconstruct_full(raw):
    """Return [(t, x, y, psi, s, v, w), ...] from raw (t, x, y, v, w)."""
    out = []
    psi = 0.0
    s = 0.0
    prev = None
    for t, x, y, v, w in raw:
        if prev is not None:
            dt = t - prev[0]
            if 0.0 < dt < 0.5:
                psi += w * dt
            s += math.hypot(x - prev[1], y - prev[2])
        out.append((t, x, y, psi, s, v, w))
        prev = (t, x, y)
    return out


def reconstruct_noisy_full(raw, sigma_v, sigma_w, rng):
    """Return [(t, x, y, psi, s, vn, wn), ...] re-integrated under noise."""
    out = []
    x = y = psi = s = 0.0
    prev = None
    for (t, _x, _y, v, w) in raw:
        vn = max(0.0, v + sigma_v * rng.gauss(0.0, 1.0))
        wn = w + sigma_w * rng.gauss(0.0, 1.0)
        if prev is not None:
            dt = t - prev
            if 0.0 < dt < 0.5:
                psi += wn * dt
                x += vn * math.cos(psi) * dt
                y += vn * math.sin(psi) * dt
                s += vn * dt
        out.append((t, x, y, psi, s, vn, wn))
        prev = t
    return out


# ---------------------------------------------------------------------------
# Samplers (return landmark indices into the full tuple list).
# ---------------------------------------------------------------------------
def baseline_landmarks(full, straight_period=2.0, curve_period=1.0):
    """The 0.5/1.0 Hz two-rate yaw-window heuristic (baseline)."""
    yaw_window = deque(maxlen=4)
    period = straight_period
    elapsed = 0.0
    last_t = None
    idx = []

    def is_curved():
        if len(yaw_window) < 4:
            return False
        y = list(yaw_window)
        d0, d1, d2 = y[1] - y[0], y[2] - y[1], y[3] - y[2]
        if (d0 > 0 and d1 > 0 and d2 > 0) or (d0 < 0 and d1 < 0 and d2 < 0):
            return True
        return any(math.pi <= abs(d) <= 2.0 * math.pi for d in (d0, d1, d2))

    for i, (t, _x, _y, psi, _s, _v, _w) in enumerate(full):
        yaw_window.append(psi)
        if last_t is not None:
            elapsed += t - last_t
        last_t = t
        period = curve_period if is_curved() else straight_period
        if elapsed >= period:
            elapsed = 0.0
            idx.append(i)
    return idx


def geometry_landmarks(full):
    """DREAM on path linearity (x, y) + speed linearity s(t)."""
    sampler = dream_core.DreamSampler(
        r2_threshold=R2_TH, max_window_size=MAX_WINDOW)
    idx = []
    for i, (t, x, y, _psi, s, _v, _w) in enumerate(full):
        if sampler.update(t, x, y, s):
            idx.append(i)
    return idx


def velocity_landmarks(full):
    """DREAM on linear velocity v(t) + angular velocity w(t)."""
    sampler = dream_core.VelocityDreamSampler(
        r2_threshold=R2_TH, max_window_size=MAX_WINDOW)
    idx = []
    for i, (t, _x, _y, _psi, _s, v, w) in enumerate(full):
        if sampler.update(t, v, w):
            idx.append(i)
    return idx


def speed_error_landmarks(full):
    """Deadband on linear + angular speed, normalized by max speed."""
    sampler = dream_core.SpeedDeadbandSampler(
        threshold=0.15, v_max=0.20, w_max=0.50, min_window_size=2)
    idx = []
    for i, (t, _x, _y, _psi, _s, v, w) in enumerate(full):
        if sampler.update(t, v, w):
            idx.append(i)
    return idx


# ---------------------------------------------------------------------------
# Reconstruction-error metrics.
# ---------------------------------------------------------------------------
def heading_rmse(full, idx):
    if len(idx) < 2:
        return float('inf')
    err = 0.0
    n = 0
    j = 0
    for i, (_t, _x, _y, psi, _s, _v, _w) in enumerate(full):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        frac = (i - i0) / (i1 - i0) if i1 > i0 else 0.0
        ph = full[i0][3] + frac * (full[i1][3] - full[i0][3])
        err += (ph - psi) ** 2
        n += 1
    return math.sqrt(err / n) if n else float('inf')


def dist_polyline(x, y, seg):
    best = float('inf')
    for (ax, ay), (bx, by) in zip(seg, seg[1:]):
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        u = 0.0 if den < 1e-12 else ((x - ax) * dx + (y - ay) * dy) / den
        u = min(max(u, 0.0), 1.0)
        best = min(best, math.hypot(x - (ax + u * dx), y - (ay + u * dy)))
    return best


def path_error(full, idx):
    if len(idx) < 2:
        return float('inf'), float('inf')
    seg = [(full[i][1], full[i][2]) for i in idx]
    worst = 0.0
    total = 0.0
    n = 0
    for p in full:
        d = dist_polyline(p[1], p[2], seg)
        worst = max(worst, d)
        total += d
        n += 1
    return worst, (total / n if n else 0.0)


def load_raw(name):
    raw = []
    with open(os.path.join(_HERE, 'bench_data', f'odom_{name}.csv')) as fh:
        for r in csv.reader(fh):
            if r:
                raw.append(tuple(float(v) for v in r))
    return raw


# Synthetic profiles where BOTH v and w change across segments (the gap the
# Gazebo worlds A/B/C leave: they keep v fixed and only vary w).
SYNTH_PROFILES = {
    'D': [(0.06, 0.0, 1.0), (0.10, 0.30, 2.0), (0.14, -0.30, 2.0),
          (0.18, 0.30, 2.0), (0.12, -0.30, 2.0), (0.08, 0.0, 1.0)],
    'E': [(0.08, 0.0, 1.0), (0.16, 0.25, 2.0), (0.08, -0.25, 2.0),
          (0.16, 0.25, 2.0), (0.08, -0.25, 2.0), (0.16, 0.0, 1.0)],
    'F': [(0.06, 0.0, 1.0), (0.10, 0.20, 2.5), (0.14, -0.20, 2.5),
          (0.18, 0.20, 2.5), (0.12, 0.0, 1.5)],
    'G': [(0.20, 0.0, 1.0), (0.16, 0.35, 2.0), (0.12, -0.35, 2.0),
          (0.08, 0.35, 2.0), (0.05, 0.0, 1.0)],
}

MAX_LIN_ACCEL = 0.20
MAX_ANG_ACCEL = 0.50


def synthesize_raw(profile, dt=0.02):
    """Integrate a (v, w, T) profile kinematically with slew limits to produce
    raw (t, x, y, v, w) -- the same offline model used for Monte-Carlo."""
    out = []
    t = 0.0
    x = y = yaw = v = w = 0.0
    for (v_cmd, w_cmd, dur) in profile:
        n = max(1, int(round(dur / dt)))
        for _ in range(n):
            dv = v_cmd - v
            v = v_cmd if abs(dv) <= MAX_LIN_ACCEL * dt else \
                v + (MAX_LIN_ACCEL * dt if dv > 0 else -MAX_LIN_ACCEL * dt)
            dw = w_cmd - w
            w = w_cmd if abs(dw) <= MAX_ANG_ACCEL * dt else \
                w + (MAX_ANG_ACCEL * dt if dw > 0 else -MAX_ANG_ACCEL * dt)
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            t += dt
            out.append((t, x, y, v, w))
    return out


def get_raw(name):
    """Return raw (t, x, y, v, w) for a benchmark world (A/B/C recorded in
    Gazebo) or a synthetic profile (D/E/F/G)."""
    if name in SYNTH_PROFILES:
        return synthesize_raw(SYNTH_PROFILES[name])
    return load_raw(name)


def mean_ci(values, z=1.96):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, z * math.sqrt(var / len(values))


def recovery_pe(full, idx):
    """Closed-loop pure-pursuit return error (mirrors recovery_pe_closed_loop)."""
    if len(idx) < 2:
        return float('inf')

    forward = [(full[i][1], full[i][2]) for i in idx]
    x = full[-1][1]
    y = full[-1][2]
    yaw = full[-1][3]
    dt = 1.0 / 50.0
    kBaseSpeed = 0.12
    kLookahead = 0.45
    kFinalTolerance = 0.03

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def slew(cur, tgt, step):
        d = tgt - cur
        return tgt if abs(d) <= step else cur + (step if d > 0 else -step)

    def norm(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # pi-turn
    target = norm(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = norm(target - yaw)
        if abs(err) < 0.03:
            break
        yaw = norm(yaw + clamp(2.0 * err, -0.8, 0.8) * dt)
        rt += dt

    path = [(x, y)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))

    # dense resample
    samples = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        nseg = max(1, int(math.ceil(seg / 0.05)))
        for j in range(nseg):
            f = j / nseg
            samples.append((x0 + dx * f, y0 + dy * f))
    samples.append(path[-1])

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart = math.hypot(x, y)
        if dstart < kFinalTolerance:
            break
        best = 0
        best_d = float('inf')
        for i, (px, py) in enumerate(samples):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_d = d
                best = i
        acc = 0.0
        gx, gy = samples[-1]
        for i in range(best, len(samples) - 1):
            acc += math.hypot(samples[i + 1][0] - samples[i][0],
                              samples[i + 1][1] - samples[i][1])
            if acc >= kLookahead:
                gx, gy = samples[i + 1]
                break
        alpha = norm(math.atan2(gy - y, gx - x) - yaw)
        v_des = kBaseSpeed * max(0.0, math.cos(alpha))
        approach = clamp(dstart / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, kBaseSpeed * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / max(kLookahead, 1e-3)
        w_des = clamp(w_des, -0.25, 0.25)
        v = slew(last_v, v_des, 0.20 * dt)
        w = slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = clamp(w, -0.25, 0.25)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = norm(yaw + w * dt)
        t += dt
    return math.hypot(x, y)


STRATEGIES = {
    'geometry': geometry_landmarks,
    'velocity': velocity_landmarks,
    'speed_error': speed_error_landmarks,
}


def with_endpoint(idx, n):
    """Append the trajectory endpoint for a fair full-path reconstruction
    comparison (mirrors benchmark_baselines_gazebo.py)."""
    if not idx or idx[-1] != n - 1:
        return idx + [n - 1]
    return idx


def evaluate_clean(name):
    raw = get_raw(name)
    full = reconstruct_full(raw)
    base_idx = baseline_landmarks(full)
    print(f"--- World {name} (clean, single run) ---")
    print(f"    baseline two-rate: N={len(base_idx)}")
    print(f"{'strategy':<10} {'N':>5} {'DR':>8} {'PE_cl(m)':>9} "
          f"{'path_max(m)':>12} {'psi_RMSE(rad)':>14}")
    for sname, fn in STRATEGIES.items():
        idx = fn(full)
        dr = 1.0 - len(idx) / len(base_idx) if base_idx else 0.0
        pe = recovery_pe(full, idx)
        idx_e = with_endpoint(idx, len(full))
        pmax, _pmean = path_error(full, idx_e)
        pr = heading_rmse(full, idx_e)
        print(f"{sname:<10} {len(idx):>5} {dr:>7.1%} {pe:>9.3f} "
              f"{pmax:>12.4f} {pr:>14.4f}")


def evaluate_mc(name, runs):
    raw = get_raw(name)
    base_full = reconstruct_full(raw)
    base_idx = baseline_landmarks(base_full)
    print(f"--- World {name} (Monte-Carlo, {runs} runs, "
          f"sigma_v=0.01, sigma_w=0.02) ---")
    print(f"    baseline two-rate: N={len(base_idx)}")
    print(f"{'strategy':<10} {'N':>12} {'DR':>12} {'PE_cl(m)':>12}")
    for sname, fn in STRATEGIES.items():
        ns, drs, pes = [], [], []
        for run in range(runs):
            rng = random.Random(9000 + run)
            full = reconstruct_noisy_full(raw, 0.01, 0.02, rng)
            idx = fn(full)
            ns.append(len(idx))
            drs.append(1.0 - len(idx) / len(base_idx))
            pes.append(recovery_pe(full, idx))
        nm, nc = mean_ci(ns)
        drm, drc = mean_ci(drs)
        pem, pec = mean_ci([v for v in pes if v != float('inf')])
        print(f"{sname:<10} {nm:>7.1f}\u00b1{nc:.1f} "
              f"{drm:>7.1%}\u00b1{drc:.1%} {pem:>7.3f}\u00b1{pec:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', action='store_true')
    ap.add_argument('--runs', type=int, default=20)
    args = ap.parse_args()

    print('=' * 78)
    print('DREAM strategy comparison: geometry vs velocity')
    print('=' * 78)
    for name in 'ABC':
        if args.mc:
            evaluate_mc(name, args.runs)
        else:
            evaluate_clean(name)
        print()
    print('--- Synthetic D/E/F/G (v and w change simultaneously) ---')
    for name in 'DEFG':
        if args.mc:
            evaluate_mc(name, args.runs)
        else:
            evaluate_clean(name)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
