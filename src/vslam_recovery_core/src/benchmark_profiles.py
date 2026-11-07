"""
benchmark_profiles.py

Single source of truth for the scripted open-loop (v, omega) velocity profiles
of the three benchmark worlds (manuscript Table tab:drpe). Both the headless
benchmark recorder (run_gazebo_benchmark.py) and the live GUI dashboard
(recovery_dashboard_gui.py) drive the robot with these exact profiles, so a
monitored GUI run matches the reported experiments.

Each profile is a list of (v, omega, duration_s) segments:

  World A : straight hallway — 0.20 m/s for 20 s = 4.0 m of travel;
  World B : slalom — alternating sharp turns (omega = +/-0.45 rad/s);
  World C : S-curve — alternating gentle waves (omega = +/-0.25 rad/s).
"""

PROFILES = {
    # A: straight hallway (max compression). 0.20 m/s x 20 s = 4.0 m.
    "A": [(0.20, 0.0, 20.0)],
    # B: pure obstacle slalom (alternating sharp turns, no straight tail), ~4 m.
    "B": [(0.15, 0.0, 1.0),
          (0.15, 0.45, 2.5), (0.15, -0.45, 2.5),
          (0.15, 0.45, 2.5), (0.15, -0.45, 2.5),
          (0.15, 0.45, 2.5), (0.15, -0.45, 2.5),
          (0.15, 0.45, 2.5), (0.15, -0.45, 2.5),
          (0.15, 0.45, 2.5), (0.15, -0.45, 2.5),
          (0.15, 0.0, 0.5)],
    # C: pure gentle S-curve (alternating gentle waves, no straight tail), ~4 m.
    "C": [(0.12, 0.0, 2.0),
          (0.12, 0.25, 5.0), (0.12, -0.25, 5.0),
          (0.12, 0.25, 5.0), (0.12, -0.25, 5.0),
          (0.12, 0.25, 5.0), (0.12, -0.25, 5.0),
          (0.12, 0.0, 1.0)],
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


def benchmark_key_for_world(world):
    """Map a world file name/path to its benchmark profile key (or None)."""
    name = world.rsplit('/', 1)[-1]
    for key, fname in PROFILE_WORLD.items():
        if fname == name:
            return key
    return None
