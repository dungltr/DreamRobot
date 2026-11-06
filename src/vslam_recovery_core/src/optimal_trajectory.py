#!/usr/bin/env python3
"""
optimal_trajectory.py

Shared optimal obstacle-avoidance trajectory for the 1.7 m recovery corridor.

The reference path is a C2-continuous quintic (degree-5) Hermite spline
parameterized by x (monotonic from 0 m to 12 m). Every segment satisfies
position, first-derivative (slope) and second-derivative (curvature)
continuity at the knots. Zero slope and zero curvature are imposed at every
knot, which yields a smooth S-curve with no velocity or acceleration
discontinuity - the practical "zero jerk" reference used throughout the
corridor demonstration (manuscript Fig. 9).

Clearance geometry (obstacle centre -> path apex, |dy| = 0.55 m):

    x = 0.00   y =  0.00   start, heading 0
    x = 1.50   y = -0.25   apex avoiding obstacle 1 at (1.5, +0.3)
    x = 2.25   y =  0.00   cross the corridor centre line
    x = 3.00   y = +0.25   apex avoiding obstacle 2 at (3.0, -0.3)
    x = 4.00   y =  0.00   exit the S-curve back on the centre line
    x = 12.00  y =  0.00   goal

Corridor limits: |y| <= 0.55 m keeps the body clear of the +/-0.85 m walls by
at least 0.30 m.
"""

import math

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
KNOT_X = [0.0, 1.5, 2.25, 3.0, 4.0, 12.0]
KNOT_Y = [0.0, -0.25, 0.0, 0.25, 0.0, 0.0]

# Natural spline boundary conditions: zero slope and zero curvature at knots.
KNOT_DY = [0.0] * len(KNOT_X)
KNOT_DDY = [0.0] * len(KNOT_X)

GOAL_X = 12.0
CORRIDOR_HALF_WIDTH = 0.85       # m, walls at y = +/-0.85
BODY_SAFETY_LIMIT = 0.55         # m, body safety bound |y| <= 0.55
CLEARANCE = 0.55                 # m, path <-> obstacle centre clearance

OBSTACLES = [
    (1.5, 0.3),
    (3.0, -0.3),
]

# Adaptive pure-pursuit lookahead (manuscript Fig. 9).
LOOKAHEAD_CURVE = 0.45           # m, inside the S-curve
LOOKAHEAD_STRAIGHT = 0.8         # m, on the straight exit
CURVE_END_X = 4.0                # x >= this value is treated as straight

MAX_ANGULAR_VELOCITY = 0.25      # rad/s, |omega| hard limit


# ---------------------------------------------------------------------------
# Quintic Hermite spline
# ---------------------------------------------------------------------------
# Basis functions expressed as coefficient vectors in t (degree 0..5).
_H00 = [1.0, 0.0, 0.0, -10.0, 15.0, -6.0]
_H01 = [0.0, 0.0, 0.0, 10.0, -15.0, 6.0]
_H10 = [0.0, 1.0, 0.0, -6.0, 8.0, -3.0]
_H11 = [0.0, 0.0, 0.0, -4.0, 7.0, -3.0]
_H20 = [0.0, 0.0, 0.5, -1.5, 1.5, -0.5]
_H21 = [0.0, 0.0, 0.0, 0.5, -1.0, 0.5]


def _add_scaled(dst, coeffs, scale):
    for k in range(6):
        dst[k] += scale * coeffs[k]


def _segment_coeffs(x0, x1, y0, y1, m0, m1, a0, a1):
    """Return the 6 quintic coefficients in the normalized parameter t."""
    h = x1 - x0
    if h <= 1e-9:
        raise ValueError("Degenerate spline segment (x0 == x1).")

    # Convert slope (dy/dx) and curvature (d2y/dx2) to the t-domain quantities
    # required by the Hermite basis (dy/dt and d2y/dt2).
    v0 = m0 * h
    v1 = m1 * h
    c0 = a0 * h * h
    c1 = a1 * h * h

    coeff = [0.0] * 6
    _add_scaled(coeff, _H00, y0)
    _add_scaled(coeff, _H01, y1)
    _add_scaled(coeff, _H10, v0)
    _add_scaled(coeff, _H11, v1)
    _add_scaled(coeff, _H20, c0)
    _add_scaled(coeff, _H21, c1)
    return coeff


class QuinticSpline:
    """C2-continuous quintic Hermite spline y(x)."""

    def __init__(self, knot_x, knot_y, knot_dy=None, knot_ddy=None):
        if len(knot_x) != len(knot_y) or len(knot_x) < 2:
            raise ValueError("Spline needs >= 2 knots with matching arrays.")
        n = len(knot_x)
        if knot_dy is None:
            knot_dy = [0.0] * n
        if knot_ddy is None:
            knot_ddy = [0.0] * n

        self.x_min = knot_x[0]
        self.x_max = knot_x[-1]
        self.segments = []
        for i in range(n - 1):
            x0, x1 = knot_x[i], knot_x[i + 1]
            coeff = _segment_coeffs(
                x0, x1,
                knot_y[i], knot_y[i + 1],
                knot_dy[i], knot_dy[i + 1],
                knot_ddy[i], knot_ddy[i + 1])
            self.segments.append((x0, x1, coeff))

    def _locate(self, x):
        """Return (segment_index, t) with x clamped to the spline domain."""
        x = min(max(x, self.x_min), self.x_max)
        for idx, (x0, x1, _coeff) in enumerate(self.segments):
            if x <= x1 or idx == len(self.segments) - 1:
                h = x1 - x0
                t = 0.0 if h <= 1e-9 else (x - x0) / h
                return idx, min(max(t, 0.0), 1.0)
        return len(self.segments) - 1, 1.0

    def _eval_segment(self, seg, t):
        x0, x1, coeff = seg
        h = x1 - x0
        y = 0.0
        dy = 0.0
        ddy = 0.0
        for k in range(5, -1, -1):
            y = y * t + coeff[k]
        for k in range(5, 0, -1):
            dy = dy * t + k * coeff[k]
        for k in range(5, 1, -1):
            ddy = ddy * t + k * (k - 1) * coeff[k]
        return y, dy / h, ddy / (h * h)

    def evaluate(self, x):
        """Return (y, dy/dx, d2y/dx2) at x."""
        idx, t = self._locate(x)
        return self._eval_segment(self.segments[idx], t)

    def y(self, x):
        return self.evaluate(x)[0]

    def heading(self, x):
        _, dy, _ = self.evaluate(x)
        return math.atan2(dy, 1.0)

    def curvature(self, x):
        _, dy, ddy = self.evaluate(x)
        return ddy / ((1.0 + dy * dy) ** 1.5)

    def sample_arc(self, num_points=1200):
        """Sample the spline into parallel lists: xs, ys, ss (arc length), yaws."""
        xs = []
        ys = []
        ss = []
        yaws = []
        s = 0.0
        px, py = None, None
        for i in range(num_points):
            x = self.x_min + (self.x_max - self.x_min) * i / (num_points - 1)
            y, dy, _ = self.evaluate(x)
            if px is not None:
                s += math.hypot(x - px, y - py)
            xs.append(x)
            ys.append(y)
            ss.append(s)
            yaws.append(math.atan2(dy, 1.0))
            px, py = x, y
        return xs, ys, ss, yaws


# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------
SPLINE = QuinticSpline(KNOT_X, KNOT_Y, KNOT_DY, KNOT_DDY)
_SAMPLED = SPLINE.sample_arc(1200)
_SAMPLE_X = _SAMPLED[0]
_SAMPLE_Y = _SAMPLED[1]
_SAMPLE_S = _SAMPLED[2]
_SAMPLE_YAW = _SAMPLED[3]


# World -> (knot_x, knot_y, curve_end_x). Each S-curve weaves between the
# staggered obstacles of the matching benchmark world so the demo robot avoids
# them instead of driving through them.
WORLD_KNOTS = {
    'recovery_corridor.sdf': ([0.0, 1.5, 2.25, 3.0, 4.0, 12.0],
                              [0.0, -0.45, 0.0, 0.45, 0.0, 0.0], 4.0),
    'world_a_straight.sdf': ([0.0, 12.0], [0.0, 0.0], 0.0),
    'world_b_slalom.sdf': ([0.0, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 7.0, 12.0],
                           [0.0, -0.45, 0.0, 0.45, 0.0, -0.45, 0.0, 0.45, 0.0, 0.0], 7.0),
    'world_c_scurve.sdf': ([0.0, 1.5, 2.25, 3.0, 4.0, 6.0, 8.0, 12.0],
                           [0.0, -0.45, 0.0, 0.45, 0.0, -0.45, 0.0, 0.0], 8.0),
}


def configure(world):
    """Switch the shared spline to the geometry of `world` (name or path)."""
    global SPLINE, _SAMPLED, _SAMPLE_X, _SAMPLE_Y, _SAMPLE_S, _SAMPLE_YAW, CURVE_END_X
    name = world.rsplit('/', 1)[-1]
    kx, ky, curve_end = WORLD_KNOTS.get(name, WORLD_KNOTS['recovery_corridor.sdf'])
    SPLINE = QuinticSpline(kx, ky, [0.0] * len(kx), [0.0] * len(kx))
    _SAMPLED = SPLINE.sample_arc(1200)
    _SAMPLE_X = _SAMPLED[0]
    _SAMPLE_Y = _SAMPLED[1]
    _SAMPLE_S = _SAMPLED[2]
    _SAMPLE_YAW = _SAMPLED[3]
    CURVE_END_X = curve_end


def waypoints(world='recovery_corridor.sdf'):
    """Return the (x, y) waypoint list for `world` (used by turn-then-drive
    auto-run modes 1/2)."""
    name = world.rsplit('/', 1)[-1]
    kx, ky, _ = WORLD_KNOTS.get(name, WORLD_KNOTS['recovery_corridor.sdf'])
    return list(zip(kx, ky))


def waypoint_polyline_points(world='recovery_corridor.sdf'):
    """Return (xs, ys, yaws) of the turn-then-drive planned path (modes 1/2):
    the straight segments connecting the spline knots, with each point oriented
    toward the next waypoint."""
    wps = waypoints(world)
    xs = [p[0] for p in wps]
    ys = [p[1] for p in wps]
    yaws = []
    for i in range(len(wps)):
        if i + 1 < len(wps):
            yaw = math.atan2(wps[i + 1][1] - wps[i][1],
                             wps[i + 1][0] - wps[i][0])
        else:
            yaw = yaws[-1] if yaws else 0.0
        yaws.append(yaw)
    return xs, ys, yaws


def nearest_index(x, y):
    """Return the sample index closest to (x, y)."""
    best = 0
    best_d = float("inf")
    for i in range(len(_SAMPLE_X)):
        d = (_SAMPLE_X[i] - x) ** 2 + (_SAMPLE_Y[i] - y) ** 2
        if d < best_d:
            best_d = d
            best = i
    return best


def lookahead_point(x, y, lookahead):
    """Return (x_ref, y_ref, yaw_ref) at `lookahead` arc length ahead of the
    robot's nearest point on the path."""
    start = nearest_index(x, y)
    target_s = _SAMPLE_S[start] + lookahead
    for i in range(start, len(_SAMPLE_X)):
        if _SAMPLE_S[i] >= target_s:
            return _SAMPLE_X[i], _SAMPLE_Y[i], _SAMPLE_YAW[i]
    return _SAMPLE_X[-1], _SAMPLE_Y[-1], _SAMPLE_YAW[-1]


def adaptive_lookahead(x):
    return LOOKAHEAD_CURVE if x < CURVE_END_X else LOOKAHEAD_STRAIGHT


def planned_path_points(num_points=600):
    """Return (xs, ys) used to publish the planned nav_msgs/Path."""
    xs = []
    ys = []
    for i in range(num_points):
        x = SPLINE.x_min + (SPLINE.x_max - SPLINE.x_min) * i / (num_points - 1)
        xs.append(x)
        ys.append(SPLINE.y(x))
    return xs, ys
