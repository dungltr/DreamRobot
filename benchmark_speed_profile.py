#!/usr/bin/env python3
"""
benchmark_speed_profile.py

Two supplementary experiments requested by the reviewer:

  1. Speed-channel stress test (M2): a synthetic trajectory with multiple
     acceleration / deceleration / stop events, where the distance-time
     linearity should show its value; report the s(t) and v(t) reconstruction
     error with and without the speed channel, and the landmark counts.

  2. Theory validation by the actual sampler (T2): run the deployed DREAM
     sampler on discrete circular arcs of varying curvature, measure the
     triggered segment length W_event and the chord error, and compare them
     with the continuous limit W_event ~ q_th/|kappa| and delta ~ |kappa|W^2/8
     at the actual operating point R^2_th = 0.98.

Usage:  python3 benchmark_speed_profile.py
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "src", "vslam_recovery_core", "src"))
import run_gazebo_benchmark as rb          # noqa: E402
import dream_core                          # noqa: E402


# ---------------------------------------------------------------------------
# 1. Speed-channel stress test
# ---------------------------------------------------------------------------
def speed_profile_recon():
    """Integrate a piecewise (v, w) profile with speed changes into a recon
    list of (t, x, y, psi, s, v), then run DREAM with/without the speed channel."""
    segs = [
        (0.0, 0.0, 1.0),    # stationary (stop)
        (0.0, 0.0, 1.0),    # stationary (stop)
        # accelerate 0 -> 0.2 m/s over 2 s
        (0.2, 0.0, 2.0),
        (0.2, 0.0, 1.0),    # cruise
        # decelerate 0.2 -> 0.05 m/s over 1.5 s
        (0.05, 0.0, 1.5),
        (0.05, 0.3, 2.0),   # slow turn
        (0.05, 0.0, 1.0),   # straight
        # accelerate 0.05 -> 0.15 over 1 s
        (0.15, 0.0, 1.0),
        (0.15, -0.3, 2.0),  # reverse turn
        (0.0, 0.0, 1.0),    # stop
        (0.0, 0.0, 1.0),    # stationary
    ]
    dt = 0.05
    x = y = psi = s = t = 0.0
    out = []
    # velocity ramp helper: interpolate v within each segment for realism
    for (v_end, w, dur) in segs:
        n = int(round(dur / dt))
        v_prev = out[-1][5] if out else 0.0
        for j in range(n):
            f = (j + 1) / n
            v = v_prev + (v_end - v_prev) * f
            x += v * math.cos(psi) * dt
            y += v * math.sin(psi) * dt
            psi += w * dt
            s += v * dt
            t += dt
            out.append((t, x, y, psi, s, v))
    return out


def s_error(recon, idx):
    """Diagnostic arc-length decoder: max |reconstructed s - true s| by
    interpolating the accumulated arc length s (recon[i][4]) between landmark
    indices. Not the deployed chord payload (see rb.chord_payload_s_error)."""
    idx = sorted(set([0] + idx + [len(recon) - 1]))
    j = 0
    mx = 0.0
    for i, (t, _x, _y, _psi, s, _v) in enumerate(recon):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        frac = (i - i0) / (i1 - i0) if i1 > i0 else 0.0
        sh = recon[i0][4] + frac * (recon[i1][4] - recon[i0][4])
        mx = max(mx, abs(sh - s))
    return mx


def v_error(recon, idx):
    """Max |reconstructed v - true v|, with v reconstructed as ds/dt along the
    interpolated s polyline (constant per segment between landmarks)."""
    idx = sorted(set([0] + idx + [len(recon) - 1]))
    mx = 0.0
    j = 0
    for i, (t, _x, _y, _psi, _s, v) in enumerate(recon):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        if i1 == i0:
            vh = 0.0
        else:
            t0, s0 = recon[i0][0], recon[i0][4]
            t1, s1 = recon[i1][0], recon[i1][4]
            vh = (s1 - s0) / (t1 - t0)
        mx = max(mx, abs(vh - v))
    return mx


def run_speed_stress():
    recon = speed_profile_recon()
    on = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
    off = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
    print("Speed-channel stress test (synthetic, multiple speed changes):")
    print(f"  path+speed: N={len(on):>2}  e_s(chord)={rb.chord_payload_s_error(recon, on):.4f} m  "
          f"e_v={v_error(recon, on):.4f} m/s")
    print(f"  path-only : N={len(off):>2}  e_s(chord)={rb.chord_payload_s_error(recon, off):.4f} m  "
          f"(arc diagnostic {s_error(recon, off):.4f})  e_v={v_error(recon, off):.4f} m/s")


# ---------------------------------------------------------------------------
# 2. Theory validation by the actual sampler on discrete arcs
# ---------------------------------------------------------------------------
def discrete_arc(kappa, W_total, dt=0.01, v=0.2, speed_profile=None):
    """Sample a circular arc of curvature kappa (v/w = R) over arc length
    W_total; return recon list (t, x, y, psi, s, v). If speed_profile is
    'ramp'/'osc' the translational speed varies across the arc so the samples
    are non-uniform in arc length."""
    n = int(round(W_total / (v * dt)))
    x = y = psi = s = t = 0.0
    out = []
    for i in range(n):
        if speed_profile == "ramp":
            vv = 0.05 + 0.35 * (i / (n - 1)) if n > 1 else v
        elif speed_profile == "osc":
            vv = 0.225 + 0.175 * math.sin(2.0 * math.pi * i / (n - 1)) if n > 1 else v
        else:
            vv = v
        w = kappa * vv
        x += vv * math.cos(psi) * dt
        y += vv * math.sin(psi) * dt
        psi += w * dt
        s += vv * dt
        t += dt
        out.append((t, x, y, psi, s, vv))
    return out


def analytic_lemma_table():
    """Continuous uniform-arc evaluation of Lemma 1: exact L versus the
    leading-order and next-order approximations, and the relative error of the
    leading-order deviation 1-L ~ (kappa W)^2/60."""
    print("\nLemma 1 analytic vs numeric (continuous uniform arc):")
    print(f"  {'kW':>5} {'L exact':>9} {'L lead':>9} {'L next':>9} {'rel.err(1-L)':>13}")
    for q in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
        lt = 0.5 - math.sin(q) / (2.0 * q)
        ln = 0.5 + math.sin(q) / (2.0 * q) - (2.0 * math.sin(q / 2.0) / q) ** 2
        L = lt / (lt + ln)
        L_lead = 1.0 - q * q / 60.0
        L_next = 1.0 - q * q / 60.0 + q ** 4 / 25200.0
        one_minus_L = 1.0 - L
        rel = (q * q / 60.0 - one_minus_L) / one_minus_L * 100.0 if one_minus_L > 0 else 0.0
        print(f"  {q:>5.2f} {L:>9.6f} {L_lead:>9.6f} {L_next:>9.6f} {rel:>12.2f}%")


def run_theory_validation():
    analytic_lemma_table()
    print("\nTheory validation (deployed sampler on discrete arcs, R^2_th=0.98):")
    q_th = math.sqrt(60 * (1 - 0.98))
    print(f"  q_th = sqrt(60*(1-0.98)) = {q_th:.4f} rad")
    print(f"  {'kappa':>6} | {'W_event':>8} | {'W_theory':>8} | "
          f"{'q_meas':>8} | {'delta_meas':>10} | {'delta_theory':>12}")
    for kappa in (0.5, 1.0, 2.0, 3.0):
        recon = discrete_arc(kappa, 5.0)
        sampler = dream_core.DreamSampler(r2_threshold=0.98, regress_speed=False,
                                          max_window_size=5000)
        events = []
        anchor_s = 0.0
        for (t, x, y, psi, s, v) in recon:
            if sampler.update(t, x, y, s):
                events.append(s - anchor_s)
                anchor_s = s
        # the first event is the start (W=0); the second is the first real trigger
        if len(events) >= 2:
            w_meas = events[1]
            q_meas = kappa * w_meas
            # exact chord sagitta: R*(1-cos(theta/2)) where theta = kappa*w
            R = 1.0 / kappa
            delta_meas_exact = R * (1.0 - math.cos(kappa * w_meas / 2.0))
            w_theory = q_th / kappa
            delta_theory = w_theory * q_th / 8.0
            print(f"  {kappa:>6.2f} | {w_meas:>8.3f} | {w_theory:>8.3f} | "
                  f"{q_meas:>8.4f} | {delta_meas_exact:>10.4f} | {delta_theory:>12.4f}")
        else:
            print(f"  {kappa:>6.2f} | (no trigger within 5 m)")

    print("\nNon-uniform arc-length sampling (varying-speed arcs, path channel only):")
    print(f"  {'kappa':>6} | {'profile':>6} | {'W_event':>8} | {'W_theory':>8} | {'dev%':>7}")
    for kappa in (0.5, 1.0, 2.0):
        for prof in ("ramp", "osc"):
            recon = discrete_arc(kappa, 6.0, speed_profile=prof)
            sampler = dream_core.DreamSampler(r2_threshold=0.98, regress_speed=False,
                                              max_window_size=5000)
            events = []
            anchor_s = 0.0
            for (t, x, y, psi, s, v) in recon:
                if sampler.update(t, x, y, s):
                    events.append(s - anchor_s)
                    anchor_s = s
            if len(events) >= 2:
                w_meas = events[1]
                w_theory = q_th / kappa
                dev = (w_meas - w_theory) / w_theory * 100.0
                print(f"  {kappa:>6.2f} | {prof:>6} | {w_meas:>8.3f} | "
                      f"{w_theory:>8.3f} | {dev:>+6.1f}%")


if __name__ == "__main__":
    run_speed_stress()
    run_theory_validation()
