#!/usr/bin/env python3
"""
benchmark_missed_event.py

Characterize and fix the cap-induced missed-event failure mode of the DREAM
sampler (reviewer M3): with a finite fitting window (m_max samples or an
arc-length cap W_cap), a long, gently curved segment whose curvature the
window can no longer see may never accumulate enough sweep angle to fire a
geometric event -- so no landmark is emitted and the return waypoint set can
become arbitrarily sparse.

The analytic detection condition (Lemma 1, q_th = sqrt(60*(1-R^2_th))):

    a geometric event can fire only if the window can accumulate
        kappa * W_window >= q_th,   W_window ~ v * m_max * dt  (or W_cap)

so the geometric channel is blind to curvatures below
    kappa* = q_th / (v * m_max * dt)     (no arc cap)
    kappa* = q_th / W_cap                (arc cap active)

This script sweeps curvature, window size, and the arc cap on a long
constant-curvature arc, reports whether any geometric event fires (the total
sweep angle before the first event), and verifies that a segment-length guard
(max_segment_length) closes the gap by emitting a safety landmark every
max_segment_length metres regardless of curvature.

Offline, no ROS/Gazebo needed:

    python3 benchmark_missed_event.py
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import dream_core  # noqa: E402

R2_TH = 0.98
Q_TH = math.sqrt(60.0 * (1.0 - R2_TH))


def make_arc(kappa, v=0.2, dt=0.02, length=20.0):
    """Constant-curvature arc sampled uniformly in time -> recon-style list of
    (t, x, y, psi, s, v); speed is constant so the temporal channel never
    fires, isolating the geometric channel."""
    w = kappa * v
    n = int(round(length / (v * dt)))
    out = []
    x = y = psi = s = t = 0.0
    for _ in range(n):
        x += v * math.cos(psi) * dt
        y += v * math.sin(psi) * dt
        psi += w * dt
        s += v * dt
        t += dt
        out.append((t, x, y, psi, s, v))
    return out


def run_sampler(recon, max_window_size=500, max_window_arc=None,
                max_segment_length=None, regress_speed=True):
    """Run DreamSampler; return (events, first_event_sweep_rad)."""
    sampler = dream_core.DreamSampler(
        r2_threshold=R2_TH, max_window_size=max_window_size,
        max_window_arc=max_window_arc, max_segment_length=max_segment_length,
        regress_speed=regress_speed)
    events = []
    for i, (t, x, y, _psi, s, _v) in enumerate(recon):
        if sampler.update(t, x, y, s):
            events.append(i)
    # total sweep at the first *post-start* event (recon[..][3] == psi)
    first_sweep = 0.0
    if len(events) >= 2:
        first_sweep = abs(recon[events[1]][3] - recon[events[0]][3])
    return events, first_sweep


def main():
    kappas = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0)
    print("=" * 84)
    print(f"Missed-event map: q_th = sqrt(60*(1-{R2_TH})) = {Q_TH:.3f} rad")
    print("long constant-curvature arc, v=0.2 m/s, dt=0.02 s, 20 m")
    print("=" * 84)

    # (a) default benchmark window (m_max=500 samples -> 2 m window), no guard
    print("\n(a) default window (m_max=500 -> W~2.0 m), no arc cap, no guard")
    print(f"    kappa* = q_th/(v*m_max*dt) = {Q_TH / (0.2 * 500 * 0.02):.3f} /m")
    print(f"{'kappa':>7} | {'events':>6} | {'first sweep':>11} | verdict")
    rows = []
    for kappa in kappas:
        recon = make_arc(kappa)
        events, first = run_sampler(recon, max_window_size=500)
        verdict = "fires" if len(events) >= 2 else "MISSED"
        print(f"{kappa:>7.2f} | {len(events):>6} | {first:>11.3f} | {verdict}")
        rows.append(("default", kappa, 500, None, None, len(events), first))

    # (b) window-size sweep at a fixed low curvature
    print("\n(b) window-size sweep at kappa=0.3 /m (no arc cap, no guard)")
    print(f"    detection needs kappa >= q_th/(v*m_max*dt)")
    print(f"{'m_max':>7} | {'W(m)':>7} | {'events':>6} | verdict")
    for m_max in (50, 100, 250, 500, 1000, 5000):
        w_win = 0.2 * m_max * 0.02
        recon = make_arc(0.3)
        events, first = run_sampler(recon, max_window_size=m_max)
        verdict = "fires" if len(events) >= 2 else "MISSED"
        print(f"{m_max:>7} | {w_win:>7.1f} | {len(events):>6} | {verdict}")
        rows.append(("msweep", 0.3, m_max, None, None, len(events), first))

    # (c) arc-cap sweep at a fixed low curvature
    print("\n(c) arc-cap sweep at kappa=0.3 /m (m_max=5000, no guard)")
    print(f"    detection needs kappa >= q_th/W_cap")
    print(f"{'W_cap':>7} | {'events':>6} | verdict")
    for w_cap in (0.5, 1.0, 2.0, 4.0):
        recon = make_arc(0.3)
        events, first = run_sampler(recon, max_window_size=5000,
                                    max_window_arc=w_cap)
        verdict = "fires" if len(events) >= 2 else "MISSED"
        print(f"{w_cap:>7.1f} | {len(events):>6} | {verdict}")
        rows.append(("capsweep", 0.3, 5000, w_cap, None, len(events), first))

    # (d) the fix: segment-length guard emits a safety landmark every D metres
    print("\n(d) segment-length guard (max_segment_length=2.0 m), default window")
    print(f"{'kappa':>7} | {'events':>6} | {'first sweep':>11} | verdict")
    for kappa in kappas:
        recon = make_arc(kappa)
        events, first = run_sampler(recon, max_window_size=500,
                                    max_segment_length=2.0)
        # with the guard, every long segment is closed off; report whether the
        # guard fired (events >= 2) -- safety landmarks count as events here.
        verdict = "closed" if len(events) >= 2 else "MISSED"
        print(f"{kappa:>7.2f} | {len(events):>6} | {first:>11.3f} | {verdict}")
        rows.append(("guard", kappa, 500, None, 2.0, len(events), first))

    # persist the map for reproducibility
    out_csv = os.path.join(_HERE, "bench_data", "missed_event_map.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "kappa", "m_max", "W_cap", "max_seg_len",
                    "events", "first_sweep_rad"])
        w.writerows(rows)
    print(f"\n[OK] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
