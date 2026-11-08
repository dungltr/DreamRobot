"""
benchmark_profiles.py

Single source of truth for the scripted open-loop (v, omega) velocity profiles
of the three benchmark worlds (manuscript Table tab:drpe). Both the headless
benchmark recorder (run_gazebo_benchmark.py) and the live GUI dashboard
(recovery_dashboard_gui.py) drive the robot with these exact profiles, so a
monitored GUI run matches the reported experiments.

Each profile is a list of (v, omega, duration_s) segments. All three drive
the robot over the same ~4 m of path (arc length) and return to the corridor
centreline (net heading and net lateral offset ~ 0):

  World A : straight — 0.20 m/s for 20 s = 4.0 m;
  World B : slalom — tight symmetric S-turns (omega = +/-0.45 rad/s);
  World C : S-curve — gentle symmetric waves (omega = +/-0.25 rad/s).

No obstacles are needed: the problem is the blind return after camera loss,
not obstacle avoidance, so the worlds only provide a straight, uniform
corridor the weave stays well inside.
"""

PROFILES = {
    # A: straight hallway (max compression). 0.20 m/s x 20 s = 4.0 m.
    "A": [(0.20, 0.0, 20.0)],
    # B: tight symmetric slalom (3 S-turns), ~4.0 m arc, |y| <= 0.29 m.
    "B": [(0.15, 0.0, 0.5),
          (0.15, 0.45, 2.15),
          (0.15, -0.45, 4.30),
          (0.15, 0.45, 4.30),
          (0.15, -0.45, 4.30),
          (0.15, 0.45, 4.30),
          (0.15, -0.45, 4.30),
          (0.15, 0.45, 2.15),
          (0.15, 0.0, 0.5)],
    # C: gentle symmetric S-curve (4 S-waves), ~4.05 m arc, |y| <= 0.13 m.
    "C": [(0.12, 0.0, 0.5),
          (0.12, 0.25, 2.05),
          (0.12, -0.25, 4.10),
          (0.12, 0.25, 4.10),
          (0.12, -0.25, 4.10),
          (0.12, 0.25, 4.10),
          (0.12, -0.25, 4.10),
          (0.12, 0.25, 4.10),
          (0.12, -0.25, 4.10),
          (0.12, 0.25, 2.05),
          (0.12, 0.0, 0.5)],
}

# Key -> world file name (as passed to run_gazebo_demo.sh / the launch world:=).
PROFILE_WORLD = {
    "A": "world_a_straight.sdf",
    "B": "world_b_slalom.sdf",
    "C": "world_c_scurve.sdf",
}

# Short human-readable labels for the GUI benchmark selector.
PROFILE_LABELS = {
    "A": "World A — straight (0.20 m/s, 20 s)",
    "B": "World B — slalom (ω=±0.45 rad/s)",
    "C": "World C — S-curve (ω=±0.25 rad/s)",
}


def profile_duration(key):
    """Total nominal duration (s) of the scripted profile for `key`."""
    return sum(seg[2] for seg in PROFILES[key])


def profile_arc_length(key):
    """Total nominal arc length (m) of the profile: sum(v * duration)."""
    return sum(seg[0] * seg[2] for seg in PROFILES[key])


def benchmark_key_for_world(world):
    """Map a world file name/path to its benchmark profile key (or None)."""
    name = world.rsplit('/', 1)[-1]
    for key, fname in PROFILE_WORLD.items():
        if fname == name:
            return key
    return None
