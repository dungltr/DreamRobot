"""
dream_core.py

Pure-Python core of the DREAM event-triggered odometry sampler. This module has
no ROS dependency so the same decision logic can be exercised offline (unit
tests / Monte-Carlo benchmarks) and inside the live ROS 2 node.

DREAM decision rule:

  * a growing window of samples (t, x, y, s) is kept, where
      t : window-local time (s),
      x, y : robot position (m),
      s : cumulative travelled distance (m);
  * two local goodness-of-fit quantities are evaluated:
      1. PATH LINEARITY  -- the fraction of spatial variance explained by the
         dominant (principal) axis of the (x, y) points. It equals 1 for a
         perfectly straight path and drops towards 0.5 as the path bends.
      2. SPEED LINEARITY -- the coefficient of determination R^2 of s(t),
         i.e. how well the travelled distance is explained by a constant-speed
         linear model.
  * while BOTH stay above a threshold the motion is a "linear regime" (steady
    straight-line, uniform motion) and no landmark is emitted;
  * when either drops below the threshold a transition (turn / speed change) is
    signalled: the caller records a boundary landmark and a fresh window starts.

Why path linearity instead of heading-vs-time R^2:
  A straight segment has a *constant* heading, so a heading-vs-time regression
  has no trend to explain and its R^2 is ill-conditioned (it collapses to ~0
  under sensor noise). The 2-D path linearity is robust: on a straight segment
  the (x, y) points lie along a single axis regardless of the heading value or
  the noise magnitude, so the principal-axis fraction remains near 1.
"""

import math


class LinFitAccumulator:
    """Running-sum OLS accumulator; computes R^2 of y = b0 + b1 * t in O(1).

    The running sums use the window-local time base so they remain numerically
    well-conditioned (no catastrophic cancellation from large absolute stamps).
    """

    __slots__ = ("n", "st", "sy", "stt", "sty", "syy")

    def __init__(self):
        self.n = 0
        self.st = 0.0    # sum t
        self.sy = 0.0    # sum y
        self.stt = 0.0   # sum t^2
        self.sty = 0.0   # sum t*y
        self.syy = 0.0   # sum y^2

    def add(self, t, y):
        self.n += 1
        self.st += t
        self.sy += y
        self.stt += t * t
        self.sty += t * y
        self.syy += y * y

    def remove(self, t, y):
        self.n -= 1
        self.st -= t
        self.sy -= y
        self.stt -= t * t
        self.sty -= t * y
        self.syy -= y * y

    def r2(self):
        """Coefficient of determination; 1.0 for a constant (perfectly linear)
        signal and for a degenerate single-point fit."""
        n = self.n
        if n < 2:
            return 1.0
        sst = self.syy - self.sy * self.sy / n          # total sum of squares
        if sst < 1e-12:
            return 1.0                                   # y constant -> linear
        denom = n * self.stt - self.st * self.st
        if abs(denom) < 1e-12:
            return 0.0                                   # no time spread -> no fit
        b1 = (n * self.sty - self.st * self.sy) / denom
        b0 = (self.sy - b1 * self.st) / n
        sse = self.syy - b0 * self.sy - b1 * self.sty    # residual sum of squares
        r2 = 1.0 - sse / sst
        if r2 > 1.0:
            r2 = 1.0
        if r2 < 0.0:
            r2 = 0.0
        return r2


class PathLinearityAccumulator:
    """Running 2-D covariance accumulator; returns the fraction of spatial
    variance explained by the principal axis of the (x, y) points.

    linearity = lambda1 / (lambda1 + lambda2) in [0.5, 1], where lambda1 >=
    lambda2 are the eigenvalues of the (x, y) covariance. A perfectly straight
    path has lambda2 = 0, hence linearity = 1; a symmetric cloud has
    lambda1 = lambda2, hence linearity = 0.5.
    """

    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self):
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0

    def add(self, x, y):
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    def remove(self, x, y):
        self.n -= 1
        self.sx -= x
        self.sy -= y
        self.sxx -= x * x
        self.syy -= y * y
        self.sxy -= x * y

    def linearity(self):
        n = self.n
        if n < 3:
            return 1.0
        cxx = self.sxx - self.sx * self.sx / n
        cyy = self.syy - self.sy * self.sy / n
        cxy = self.sxy - self.sx * self.sy / n
        trace = cxx + cyy
        if trace < 1e-12:
            return 1.0  # collapsed point cloud -> treat as linear
        disc = (cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy
        lambda1 = 0.5 * (trace + math.sqrt(max(disc, 0.0)))
        lin = lambda1 / trace
        if lin > 1.0:
            lin = 1.0
        if lin < 0.5:
            lin = 0.5
        return lin


def unwrap_delta(prev, curr):
    """Return `curr` wrapped onto the branch nearest to `prev` (for unwrapping)."""
    d = curr - prev
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return prev + d


def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert a quaternion to the yaw angle around z (rad)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class DreamSampler:
    """Stateful DREAM sampler over the (x, y, s) odometry stream.

    Call :meth:`update` with each new sample; it returns True exactly when a
    boundary landmark should be recorded at that sample.
    """

    def __init__(self, r2_threshold=0.98, min_window_size=5,
                 max_window_size=500, regress_speed=True,
                 min_event_spacing=0.2, max_window_arc=None):
        self.r2_threshold = float(r2_threshold)
        self.min_window_size = int(min_window_size)
        self.max_window_size = int(max_window_size)
        self.regress_speed = bool(regress_speed)
        self.min_event_spacing = float(min_event_spacing)
        # Optional speed-invariant cap: keep only the last `max_window_arc`
        # metres of arc, so the linearity fit spans a fixed distance rather
        # than a fixed number of samples (which depends on the robot speed).
        self.max_window_arc = (float(max_window_arc)
                               if max_window_arc is not None else None)

        self._window = []                      # (t_local, x, y, s)
        self._path = PathLinearityAccumulator()
        self._speed = LinFitAccumulator()
        self._anchor = None        # absolute time of the current window origin
        self._last_event_t = None  # absolute time of the previous boundary landmark
        self.n_events = 0          # number of landmarks emitted

    # -- window helpers ---------------------------------------------------- #
    def _append(self, t, x, y, s):
        self._window.append((t, x, y, s))
        self._path.add(x, y)
        self._speed.add(t, s)

    def _drop_oldest(self):
        t, x, y, s = self._window.pop(0)
        self._path.remove(x, y)
        self._speed.remove(t, s)

    def _goodness(self):
        """Return the combined goodness-of-fit; a transition is signalled when
        this falls below the threshold."""
        lin = self._path.linearity()
        if not self.regress_speed:
            return lin
        return min(lin, self._speed.r2())

    # -- public API -------------------------------------------------------- #
    def update(self, t, x, y, s):
        """Feed one sample at absolute time `t`.

        Returns True when the sample is a transition event and a landmark
        should be recorded (this includes the very first start landmark).
        """
        if self._anchor is None:
            self._anchor = t
            self._last_event_t = t
            self._append(0.0, x, y, s)
            self.n_events += 1
            return True

        t_local = max(0.0, t - self._anchor)
        self._append(t_local, x, y, s)
        while len(self._window) > self.max_window_size:
            self._drop_oldest()
        if self.max_window_arc is not None and len(self._window) > 1:
            while self._window[-1][3] - self._window[0][3] > self.max_window_arc:
                self._drop_oldest()

        if len(self._window) < self.min_window_size:
            return False
        if self._goodness() >= self.r2_threshold:
            return False
        if t - self._last_event_t < self.min_event_spacing:
            return False

        # Transition event: restart the window at the boundary point.
        self._last_event_t = t
        self.n_events += 1
        tl, bx, by, bs = self._window[-1]
        self._anchor += tl
        self._window = [(0.0, bx, by, bs)]
        self._path = PathLinearityAccumulator()
        self._speed = LinFitAccumulator()
        self._path.add(bx, by)
        self._speed.add(0.0, bs)
        return True

    @property
    def last_event_t(self):
        return self._last_event_t


class VelocityDreamSampler:
    """Velocity-based DREAM variant: two linear-regression trackers in parallel.

    The geometry-based :class:`DreamSampler` watches the (x, y) path linearity
    and the travelled-distance linearity s(t). This variant instead watches the
    *raw* odometry velocities directly, as two parallel trackers over one
    growing window:

      1. linear velocity   v(t) -> R^2 of the local linear fit;
      2. angular velocity  w(t) -> R^2 of the local linear fit.

    The combined goodness is G = min(R^2_v, R^2_w). While G >= R^2_th the
    motion is a linear regime (steady speed and steady turn) and nothing is
    logged; when G < R^2_th a transition is signalled and the boundary
    landmark is flushed.

    Caveat: on *clean* odometry a constant v or w yields R^2 = 1 (no trigger),
    but under sensor noise a constant signal has no time trend and its R^2
    collapses toward 0, so this variant is noise-sensitive on steady segments
    -- the very degeneracy the (x, y) path linearity is designed to avoid.
    """

    def __init__(self, r2_threshold=0.98, min_window_size=5,
                 max_window_size=500, min_event_spacing=0.2):
        self.r2_threshold = float(r2_threshold)
        self.min_window_size = int(min_window_size)
        self.max_window_size = int(max_window_size)
        self.min_event_spacing = float(min_event_spacing)

        self._window = []                  # (t_local, v, w)
        self._vel = LinFitAccumulator()    # v(t)
        self._ang = LinFitAccumulator()    # w(t)
        self._anchor = None
        self._last_event_t = None
        self.n_events = 0

    def _append(self, t, v, w):
        self._window.append((t, v, w))
        self._vel.add(t, v)
        self._ang.add(t, w)

    def _drop_oldest(self):
        t, v, w = self._window.pop(0)
        self._vel.remove(t, v)
        self._ang.remove(t, w)

    def _goodness(self):
        return min(self._vel.r2(), self._ang.r2())

    def update(self, t, v, w):
        """Feed one (time, linear velocity, angular velocity) sample.

        Returns True when a transition event occurs and a landmark should be
        recorded (this includes the very first start landmark).
        """
        if self._anchor is None:
            self._anchor = t
            self._last_event_t = t
            self._append(0.0, v, w)
            self.n_events += 1
            return True

        t_local = max(0.0, t - self._anchor)
        self._append(t_local, v, w)
        while len(self._window) > self.max_window_size:
            self._drop_oldest()

        if len(self._window) < self.min_window_size:
            return False
        if self._goodness() >= self.r2_threshold:
            return False
        if t - self._last_event_t < self.min_event_spacing:
            return False

        # Transition event: restart the window at the boundary sample.
        self._last_event_t = t
        self.n_events += 1
        tl, bv, bw = self._window[-1]
        self._anchor += tl
        self._window = [(0.0, bv, bw)]
        self._vel = LinFitAccumulator()
        self._ang = LinFitAccumulator()
        self._vel.add(0.0, bv)
        self._ang.add(0.0, bw)
        return True

    @property
    def last_event_t(self):
        return self._last_event_t
