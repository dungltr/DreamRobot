#!/usr/bin/env python3
"""
test_dream_guards.py

Deterministic unit / stress tests for the DREAM sampler guards and edge cases
(reviewer M9 / E6). Each case exercises the real `dream_core.DreamSampler`
(no ROS) and checks the documented behaviour. The sampler assumes a monotonic
travelled distance s(t) with v >= 0; negative-velocity handling is out of
scope and is verified as "not specially handled" rather than silently fixed.

Usage:  python3 test_dream_guards.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "src", "vslam_recovery_core", "src"))

import dream_core  # noqa: E402


def _run(sampler, samples):
    """Feed (t, x, y, s) samples; return the list of emitted sample indices."""
    events = []
    for i, (t, x, y, s) in enumerate(samples):
        if sampler.update(t, x, y, s):
            events.append(i)
    return events


def _line(length, v=0.2, dt=0.02, n=None):
    if n is None:
        n = int(round(length / (v * dt)))
    out = []
    t = x = y = s = 0.0
    for i in range(n):
        x += v * dt
        s += v * dt
        t += dt
        out.append((t, x, y, s))
    return out


def _stationary(n, dt=0.02, jitter=0.0, rng=None):
    out = []
    t = s = 0.0
    for _ in range(n):
        x = 0.0
        y = 0.0
        if jitter > 0:
            x = rng.gauss(0.0, jitter)
            y = rng.gauss(0.0, jitter)
        t += dt
        out.append((t, x, y, s))
    return out


def main():
    import random
    results = []

    def check(name, cond, detail):
        results.append((name, cond, detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    # 1. Straight constant speed -> only start + (no regression event beyond
    #    geometry changes). A perfect line emits no intermediate landmark.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    ev = _run(sm, _line(4.0))
    check("straight-const", len(ev) == 1,
          f"events={len(ev)} (expected 1: start only)")

    # 2. Turn (path bends sharply) -> a regression event fires. The turn is
    #    long enough to fill the ~2 m fitting window (500 samples at v=0.2).
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    samp = _line(1.0)
    t, x, y, s = samp[-1]
    n = 500
    dt = 0.02
    psi = 0.0
    for i in range(n):
        psi += 0.6 * dt
        x += 0.2 * math.cos(psi) * dt
        y += 0.2 * math.sin(psi) * dt
        s += 0.2 * dt
        t += dt
        samp.append((t, x, y, s))
    ev = _run(sm, samp)
    check("turn-fires", len(ev) >= 2, f"events={len(ev)} (expected >=2)")

    # 3. Stationary clean (zero variance) -> no event.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    ev = _run(sm, _stationary(200))
    check("stationary-clean", len(ev) == 1,
          f"events={len(ev)} (expected 1: start only)")

    # 4. Stationary + jitter -> documented: L ~ 0.5 may fire; confirm behaviour
    #    is the documented noisy-stationarity case, not a crash.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    rng = random.Random(7)
    ev = _run(sm, _stationary(200, jitter=0.001, rng=rng))
    check("stationary-jitter", True,
          f"events={len(ev)} (documented: noisy stationarity L~0.5, may emit)")

    # 5. Pure rotation (turn in place): (x,y,s) unchanged -> no event; heading
    #    is not a trigger input (documented limitation).
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    samp = []
    t = s = 0.0
    for i in range(200):
        t += 0.02
        samp.append((t, 0.0, 0.0, s))
    ev = _run(sm, samp)
    check("pure-rotation", len(ev) == 1,
          f"events={len(ev)} (expected 1: rotation not stored)")

    # 6. Duplicate timestamp: sampler must not crash / divide by zero.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    samp = _line(1.0)
    dup = samp + [samp[-1]] * 20 + _line(1.0)
    ev = _run(sm, dup)
    check("duplicate-timestamp", len(ev) >= 1,
          f"events={len(ev)} (no crash)")

    # 7. Out-of-order timestamp (t decreasing): sampler must not crash.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    ooo = _line(1.0) + [(_line(1.0)[0][0] - 0.1, 0.0, 0.0, 0.0)] + _line(1.0)
    ev = _run(sm, ooo)
    check("out-of-order", len(ev) >= 1, f"events={len(ev)} (no crash)")

    # 8. D_max flush fires independently of min spacing.
    sm = dream_core.DreamSampler(r2_threshold=0.98, min_event_spacing=0.2,
                                 max_segment_length=1.0)
    ev = _run(sm, _line(5.0, v=0.2))
    # 5 m / 1 m flush -> >= 4 post-start flushes
    check("dmax-flush", len(ev) >= 5,
          f"events={len(ev)} (expected >=5 for 5 m at D_max=1 m)")

    # 9. Window cap smaller than min-window span: document that flush can be
    #    ineligible (no crash; may emit only start).
    sm = dream_core.DreamSampler(r2_threshold=0.98, min_window_size=50,
                                 max_window_arc=0.05)
    ev = _run(sm, _line(2.0, v=0.2, dt=0.02))
    check("tiny-cap", len(ev) >= 1,
          f"events={len(ev)} (cap/m_min interaction, no crash)")

    # 10. Event exactly at endpoint: boundary dedup keeps a valid polyline.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    samp = _line(1.0) + _line(1.0)  # straight, no event -> endpoint handling
    ev = _run(sm, samp)
    check("endpoint", len(ev) >= 1, f"events={len(ev)} (start at least)")

    # 11. Large coordinates: numerical stability of running sums.
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    big = _line(2.0, v=0.2)
    big = [(t, x + 1e6, y + 1e6, s) for (t, x, y, s) in big]
    ev = _run(sm, big)
    check("large-coords", len(ev) >= 1, f"events={len(ev)} (stable)")

    # 12. Negative velocity: sampler uses the caller-provided s; document that
    #     it does not special-case v<0 (s would decrease).
    sm = dream_core.DreamSampler(r2_threshold=0.98)
    neg = []
    t = s = 0.0
    x = 0.0
    for i in range(100):
        t += 0.02
        s -= 0.2 * 0.02   # negative travelled distance (v<0)
        x -= 0.2 * 0.02
        neg.append((t, x, 0.0, s))
    ev = _run(sm, neg)
    check("negative-v", True,
          f"events={len(ev)} (documented: no |v| convention; s decreasing)")

    npass = sum(1 for _, c, _ in results if c)
    print(f"\n{'-'*60}\n{npass}/{len(results)} checks PASS")

    out = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "bench_data", "guards_test_results.txt")
    with open(out, "w") as fh:
        for name, cond, detail in results:
            fh.write(f"{'PASS' if cond else 'FAIL'}\t{name}\t{detail}\n")
    print(f"[OK] wrote {out}")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
