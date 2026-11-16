#!/usr/bin/env python3
"""
benchmark_m5_noise.py

Extended noise / operating-envelope study (reviewer M5): separate the sensor
noise seen by the dead-reckoned estimate from the process (actuation) noise
that physically disturbs the true pose, and add the systematic odometry
parameter errors the nominal model does not cover:

  (a) wheel-radius scale error  eps_r : v_meas = (1+eps_r) v_true;
  (b) gyro/heading scale error  eps_b : w_meas = (1+eps_b) w_true + bias + noise;
  (c) both-sign gyro bias  (+/- 0.005/0.01/0.02 rad/s);
  (d) process (actuation) noise  sigma_pw : the TRUE pose integrates
      w_true = w_cmd + sigma_pw*eps, i.e. wheel slip the controller cannot see;
  (e) footprint wall clearance along the noisy return (not just the endpoint).

The return replay is a progress-constrained pure-pursuit controller so the
nearest-point search cannot jump across near branches (used here and in the
M6 trace study). All experiments are offline, 50 runs, seed Random(5000+r)
forward / Random(9000+r) return.

Usage:  python3 benchmark_m5_noise.py
"""

import csv
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb  # noqa: E402

DATA = os.path.join(_HERE, "bench_data")
RUNS = 50
# robot footprint (matches benchmark_return_safety.py / corridor demo)
ROBOT_HW, ROBOT_HD = 0.15, 0.10
WALL_HALF = 1.10


def load_raw(name):
    raw = []
    with open(os.path.join(DATA, f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return raw


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


def polyline_samples(path, spacing=0.05):
    samples = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg / spacing)))
        for j in range(n):
            f = j / n
            samples.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    samples.append(path[-1])
    return samples


def forward_estimate(raw, sigma_v, sigma_w, rng, gyro_bias=0.0,
                     wheel_scale=0.0, gyro_scale=0.0, arw=0.0,
                     bias_instability=0.0):
    """Dead-reckoned estimate integrating the measured velocity with wheel
    radius scale, angular-rate scale, constant bias, white sensor noise, and
    two independent rate random walks (`arw` and `bias_instability`, both
    rad/s/sqrt(s), each accumulated as `state += coef*sqrt(dt)*gauss()` and
    then added to the rate before integrating to heading). Note: these two
    parameters are rate random walks on the angular rate, NOT angle random
    walk; they are retained for completeness but are not part of the reported
    sweep. Returns the (t, x, y, psi, s, v) tuple list."""
    out = []
    x = y = psi = s = 0.0
    rw = 0.0
    bias = 0.0
    prev = None
    for (t, _x, _y, v, w) in raw:
        dt = t - prev if prev is not None else 0.0
        vn = max(0.0, (1.0 + wheel_scale) * v + sigma_v * rng.gauss(0.0, 1.0))
        if 0.0 < dt < 0.5:
            if arw != 0.0:
                rw += arw * rng.gauss(0.0, 1.0) * math.sqrt(dt)
            if bias_instability != 0.0:
                bias += bias_instability * rng.gauss(0.0, 1.0) * math.sqrt(dt)
        wn = (1.0 + gyro_scale) * w + gyro_bias + bias + rw + sigma_w * rng.gauss(0.0, 1.0)
        if prev is not None and 0.0 < dt < 0.5:
            psi += wn * dt
            x += vn * math.cos(psi) * dt
            y += vn * math.sin(psi) * dt
            s += vn * dt
        out.append((t, x, y, psi, s, vn))
        prev = t
    return out


def two_state_return(true_recon, est_recon, landmarks_idx, sigma_v, sigma_w,
                     rng, gyro_bias=0.0, wheel_scale=0.0, gyro_scale=0.0,
                     process_v=0.0, process_w=0.0, timeout=180.0,
                     turn_timeout=12.0, noise_v=None, noise_w=None):
    """Two-state progress-constrained pure-pursuit return.

    True pose: integrates the commanded velocity perturbed by PROCESS noise
    (actuation slip) -- (v_cmd + pv, w_cmd + pw).
    Estimate: integrates the *measured* velocity -- scale*(v_cmd) + sensor
    noise + bias + scale*(w_cmd), i.e. what the controller actually observes.

    `timeout` is the return-phase time budget (s); `turn_timeout` is the
    in-place pi-turn budget. Length-scaled callers pass a larger `timeout` for
    longer paths so the fixed budget does not dominate.

    Returns (pe_true, pe_odom, clearance, completed, timed_out).
    """
    if len(landmarks_idx) < 1:
        return float("inf"), float("inf"), float("inf"), False, True

    forward = [(est_recon[i][1], est_recon[i][2]) for i in landmarks_idx]
    x = true_recon[-1][1]
    y = true_recon[-1][2]
    yaw = true_recon[-1][3]
    x_est = est_recon[-1][1]
    y_est = est_recon[-1][2]
    yaw_est = est_recon[-1][3]

    dt = 1.0 / 50.0
    kBaseSpeed = 0.12
    kLookahead = 0.45
    kFinalTolerance = 0.03

    # Common-random-number mode: measurement noise (sigma_v / sigma_w) is drawn
    # from pre-generated arrays indexed by a shared step counter, so two calls
    # with the same arrays consume identical per-step measurement noise. Process
    # noise still draws from `rng` (it is zero in the CRN paired study).
    use_crn = noise_v is not None and noise_w is not None
    meas_k = [0]

    def mv():
        if use_crn:
            val = noise_v[meas_k[0]]
            meas_k[0] += 1
            return val
        return rng.gauss(0.0, 1.0)

    def mw():
        if use_crn:
            val = noise_w[meas_k[0]]
            meas_k[0] += 1
            return val
        return rng.gauss(0.0, 1.0)

    target = norm(yaw_est + math.pi)
    rt = 0.0
    while rt < turn_timeout:
        err = norm(target - yaw_est)
        if abs(err) < 0.03:
            break
        w = clamp(2.0 * err, -0.8, 0.8)
        yaw = norm(yaw + (w + process_w * rng.gauss(0.0, 1.0)) * dt)
        yaw_est = norm(yaw_est + ((1.0 + gyro_scale) * w + gyro_bias +
                                  sigma_w * mw()) * dt)
        rt += dt

    path = [(x_est, y_est)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = polyline_samples(path)

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    # progress-constrained nearest-point index (monotonic, no cross-branch jump)
    best_idx = 0
    clearance = float("inf")
    while t < timeout:
        dstart_est = math.hypot(x_est, y_est)
        if dstart_est < kFinalTolerance:
            break
        # search only forward from the current progress index (plus a small
        # local lookback so the initial approach can re-acquire the path)
        lo = max(0, best_idx - 3)
        best = lo
        best_d = float("inf")
        for i in range(lo, len(samples)):
            d = (samples[i][0] - x_est) ** 2 + (samples[i][1] - y_est) ** 2
            if d < best_d:
                best_d = d
                best = i
            elif d > best_d and (i - best) > 20:
                break
        best_idx = best
        acc = 0.0
        gx, gy = samples[-1]
        for i in range(best_idx, len(samples) - 1):
            acc += math.hypot(samples[i + 1][0] - samples[i][0],
                              samples[i + 1][1] - samples[i][1])
            if acc >= kLookahead:
                gx, gy = samples[i + 1]
                break
        alpha = norm(math.atan2(gy - y_est, gx - x_est) - yaw_est)
        v_des = kBaseSpeed * max(0.0, math.cos(alpha))
        approach = clamp(dstart_est / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, kBaseSpeed * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / max(kLookahead, 1e-3)
        w_des = clamp(w_des, -0.25, 0.25)
        v = slew(last_v, v_des, 0.20 * dt)
        w = slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = clamp(w, -0.25, 0.25)

        # true pose: commanded + process (slip) noise
        x += (v + process_v * rng.gauss(0.0, 1.0)) * math.cos(yaw) * dt
        y += (v + process_v * rng.gauss(0.0, 1.0)) * math.sin(yaw) * dt
        yaw = norm(yaw + (w + process_w * rng.gauss(0.0, 1.0)) * dt)

        # estimate: measured velocity (scale + bias + sensor noise)
        vn = (1.0 + wheel_scale) * v + sigma_v * mv()
        wn = (1.0 + gyro_scale) * w + gyro_bias + sigma_w * mw()
        x_est += max(0.0, vn) * math.cos(yaw_est) * dt
        y_est += max(0.0, vn) * math.sin(yaw_est) * dt
        yaw_est = norm(yaw_est + wn * dt)

        clearance = min(clearance, footprint_wall_clearance(x, y, yaw))
        t += dt

    pe_true = math.hypot(x, y)
    pe_odom = math.hypot(x_est, y_est)
    completed = pe_odom < 0.03
    return pe_true, pe_odom, clearance, completed, (not completed)


def footprint_wall_clearance(x, y, yaw):
    """Minimum distance from the robot rectangle to the corridor inner face."""
    best = float("inf")
    c = math.cos(yaw)
    s = math.sin(yaw)
    for sx in (-ROBOT_HW, ROBOT_HW):
        for sy in (-ROBOT_HD, ROBOT_HD):
            cy = y + sx * s + sy * c
            best = min(best, WALL_HALF - abs(cy))
    return best


def run(name, **kw):
    raw = load_raw(name)
    true_full = rb.reconstruct(raw)
    pes, clear, n_completed = [], [], 0
    for run_i in range(RUNS):
        rng = random.Random(5000 + run_i)
        est_full = forward_estimate(raw, kw.get("sigma_v", 0.01),
                                    kw.get("sigma_w", 0.02), rng,
                                    gyro_bias=kw.get("gyro_bias", 0.0),
                                    wheel_scale=kw.get("wheel_scale", 0.0),
                                    gyro_scale=kw.get("gyro_scale", 0.0),
                                    arw=kw.get("arw", 0.0),
                                    bias_instability=kw.get("bias_instability", 0.0))
        d = rb.dream_sampler(est_full, r2_threshold=0.98)
        r1 = random.Random(9000 + run_i)
        pe, _pod, clr, done, _to = two_state_return(
            true_full, est_full, d, kw.get("sigma_v", 0.01),
            kw.get("sigma_w", 0.02), r1,
            gyro_bias=kw.get("gyro_bias", 0.0),
            wheel_scale=kw.get("wheel_scale", 0.0),
            gyro_scale=kw.get("gyro_scale", 0.0),
            process_v=kw.get("process_v", 0.0),
            process_w=kw.get("process_w", 0.0))
        pes.append(pe)
        clear.append(clr)
        n_completed += 1 if done else 0
    return pes, clear, n_completed


def succ(pes):
    return sum(1 for x in pes if x < 0.06) / len(pes)


def pct(pes, q):
    s = sorted(pes)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    print("=" * 88)
    print(f"M5 extended noise / operating-envelope study ({RUNS} runs, "
          f"sigma_v=0.01, sigma_w=0.02)")
    print("=" * 88)

    print("\n(a) wheel-radius scale error eps_r (v_meas=(1+eps_r) v)")
    for eps_r in (0.0, 0.01, 0.02, 0.05):
        row = []
        for name in "ABC":
            pes, _c, _n = run(name, wheel_scale=eps_r)
            row.append(f"{name}: {sum(pes)/len(pes):.3f} m / {succ(pes)*100:.0f}%")
        print(f"  eps_r={eps_r:>4.2f}  " + "   ".join(row))

    print("\n(b) gyro/heading scale error eps_b (w_meas=(1+eps_b) w)")
    for eps_b in (0.0, 0.01, 0.02, 0.05):
        row = []
        for name in "ABC":
            pes, _c, _n = run(name, gyro_scale=eps_b)
            row.append(f"{name}: {sum(pes)/len(pes):.3f} m / {succ(pes)*100:.0f}%")
        print(f"  eps_b={eps_b:>4.2f}  " + "   ".join(row))

    print("\n(c) both-sign gyro bias")
    for bias in (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02):
        row = []
        for name in "ABC":
            pes, _c, _n = run(name, gyro_bias=bias)
            row.append(f"{name}: {sum(pes)/len(pes):.3f} m / {succ(pes)*100:.0f}%")
        print(f"  bias={bias:>6.3f}  " + "   ".join(row))

    print("\n(d) process/actuation (slip) noise sigma_pw on the TRUE pose "
          "(controller cannot observe it)")
    for pw in (0.0, 0.01, 0.02, 0.05):
        row = []
        for name in "ABC":
            pes, _c, _n = run(name, process_w=pw)
            row.append(f"{name}: {sum(pes)/len(pes):.3f} m / {succ(pes)*100:.0f}%")
        print(f"  sigma_pw={pw:>4.2f}  " + "   ".join(row))

    print("\n(e) minimum footprint wall clearance along the noisy return "
          "(nominal noise)")
    for name in "ABC":
        pes, clear, _n = run(name)
        print(f"  {name}: mean PE={sum(pes)/len(pes):.3f} m  "
              f"min clearance={min(clear):.3f} m  "
              f"(all positive = no wall contact)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
