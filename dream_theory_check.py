#!/usr/bin/env python3
"""
dream_theory_check.py

Numerical validation of the two theoretical anchors for the DREAM sampler:

  1. PATH-LINEARITY SCALING: for a circular arc of curvature kappa and window
     arc length W, the path linearity (principal-axis fraction) obeys
         linearity ~= 1 - c * (kappa * W)^2          (c a small constant).
     This is the "R^2_th <-> curvature" relationship: a threshold on linearity
     is equivalent to a bound on kappa * W.

  2. RECONSTRUCTION ERROR: the perpendicular deviation of a chord of length W
     from its circular arc is
         delta ~= kappa * W^2 / 8.
     Combined with (1), the DREAM threshold implicitly bounds delta per segment,
     which is why DREAM keeps the path reconstruction error roughly uniform
     across straight and curved segments.

Run:  python3 dream_theory_check.py
"""

import math
import sys


def arc_points(kappa, W, n=200):
    """Sample a circular arc of curvature kappa and arc length W, centred so the
    chord is horizontal. Returns list of (x, y)."""
    R = 1.0 / kappa
    theta = kappa * W          # total sweep angle
    pts = []
    for i in range(n + 1):
        a = -theta / 2.0 + theta * i / n
        x = R * math.sin(a)
        y = R * (1.0 - math.cos(a))
        pts.append((x, y))
    return pts


def path_linearity(pts):
    """Principal-axis fraction lambda1/(lambda1+lambda2) of the (x,y) points."""
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    syy = sum(p[1] * p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    cxx = sxx - sx * sx / n
    cyy = syy - sy * sy / n
    cxy = sxy - sx * sy / n
    trace = cxx + cyy
    if trace < 1e-12:
        return 1.0
    disc = (cxx - cyy) ** 2 + 4.0 * cxy * cxy
    lambda1 = 0.5 * (trace + math.sqrt(disc))
    return lambda1 / trace


def chord_deviation(kappa, W):
    """Max perpendicular deviation of the arc from its chord (delta ~= kappa W^2/8)."""
    R = 1.0 / kappa
    theta = kappa * W
    return R * (1.0 - math.cos(theta / 2.0))


def main():
    print("=" * 74)
    print("DREAM theory check: path-linearity <-> curvature, and chord error")
    print("=" * 74)

    print("\n(1) linearity vs kappa*W  (should be ~ 1 - c*(kappa*W)^2)")
    print("-" * 74)
    print(f"{'kappa*W':>9} | {'linearity':>10} | {'1-linear':>10} | "
          f"{'(1-linear)/(kW)^2':>16}")
    print("-" * 74)
    kappa = 1.0  # 1/rad (R = 1 m)
    consts = []
    for kW in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        W = kW / kappa
        lin = path_linearity(arc_points(kappa, W))
        gap = 1.0 - lin
        c = gap / (kW * kW) if kW > 1e-9 else 0.0
        consts.append(c)
        print(f"{kW:>9.3f} | {lin:>10.6f} | {gap:>10.6f} | {c:>16.5f}")
    print("-" * 74)
    print(f"c ~= {sum(consts[2:])/len(consts[2:]):.5f}  (averaged over kW >= 0.4)")

    print("\n(2) chord deviation delta vs kappa*W^2/8")
    print("-" * 74)
    print(f"{'kappa':>7} | {'W':>7} | {'delta':>9} | {'kappa*W^2/8':>12}")
    print("-" * 74)
    for kappa, W in ((1.0, 0.5), (1.0, 1.0), (2.0, 0.5), (0.5, 1.0)):
        d = chord_deviation(kappa, W)
        approx = kappa * W * W / 8.0
        print(f"{kappa:>7.2f} | {W:>7.2f} | {d:>9.5f} | {approx:>12.5f}")
    print("-" * 74)

    print("\nImplication: a linearity threshold ~= 1 - c*(kappa*W)^2 implies")
    print("  kappa*W <= sqrt((1 - threshold)/c), hence per-segment error")
    print("  delta <= W * sqrt((1 - threshold)/c) / 8, i.e. the threshold")
    print("  bounds the reconstruction error of every segment uniformly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
