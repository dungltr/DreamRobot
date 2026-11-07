#!/usr/bin/env bash
#
# run_gazebo_demo.sh — launch the DREAM blind-recovery demo with the Gazebo GUI,
# RViz and the live dashboard, so the system can be watched as it records
# landmarks and recovers from a simulated camera failure.
#
# Usage:
#   ./run_gazebo_demo.sh                      # interactive menu (or default corridor
#                                              # when there is no terminal)
#   ./run_gazebo_demo.sh recovery_corridor.sdf # demo corridor (2 obstacles)
#   ./run_gazebo_demo.sh world_a_straight.sdf # benchmark World A (straight)
#   ./run_gazebo_demo.sh world_b_slalom.sdf   # benchmark World B (slalom)
#   ./run_gazebo_demo.sh world_c_scurve.sdf   # benchmark World C (S-curve)
#
# Optional env overrides:
#   R2_TH (default 0.98) — DREAM linearity threshold.
# NOTE: no `set -u` here — ROS setup scripts reference unset vars
# (e.g. AMENT_TRACE_SETUP_FILES) and break under nounset.
set -eo pipefail

if [ $# -ge 1 ]; then
  WORLD="$1"
else
  if [ -t 0 ]; then
    echo "Select a world to run:"
    echo "  1) recovery_corridor.sdf  (demo corridor, 2 obstacles)"
    echo "  2) world_a_straight.sdf   (benchmark World A — straight)"
    echo "  3) world_b_slalom.sdf     (benchmark World B — slalom)"
    echo "  4) world_c_scurve.sdf     (benchmark World C — S-curve)"
    printf 'Choice [1]: '
    read -r CHOICE
    case "$CHOICE" in
      1|"") WORLD="recovery_corridor.sdf" ;;
      2) WORLD="world_a_straight.sdf" ;;
      3) WORLD="world_b_slalom.sdf" ;;
      4) WORLD="world_c_scurve.sdf" ;;
      *) echo "Invalid choice, using recovery_corridor.sdf" >&2
         WORLD="recovery_corridor.sdf" ;;
    esac
  else
    WORLD="recovery_corridor.sdf"
  fi
fi
R2_TH="${R2_TH:-0.98}"
STRATEGY="${STRATEGY:-geometry}"
WS_DIR="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"

cat <<EOF
=====================================================================
DREAM blind-recovery demo
  world      : $WORLD
  sampler    : dream_sampler_node (use_dream:=true, R2_th=$R2_TH)
  displays   : Gazebo GUI + RViz 2 + live Dashboard
=====================================================================
How to monitor (match the paper):
  1. Gazebo     — watch the robot drive the corridor.
  2. RViz       — green = forward path, red = recovery path.
  3. Dashboard  — 'Path landmarks recorded' = landmarks recorded by DREAM.

To run the corridor demo:
  1. In the Dashboard press "Auto 4" (continuous curve following).
  2. Watch the landmark count grow (sparse on straights, dense at turns).
  3. Press "CUT CAMERA SIGNAL (Trigger Failure)" -> the robot turns pi and
     replays the recorded path in reverse.
  4. Press "Compute Position Error (PE %)" to read the return position error.

To run a manuscript benchmark (World A/B/C):
  1. Pick the world from the launch menu above (or pass it as an argument).
  2. In the Dashboard press "Run benchmark (auto-cut -> return)" under
     "Benchmark Experiment (open-loop profile)".
  3. The Dashboard drives the scripted open-loop (v, omega) profile (identical
     to run_gazebo_benchmark.py), auto-cuts the camera at the selected 70-100%,
     and the recovery controller returns the robot to the start.
=====================================================================
EOF

# Clear any stale landmarks from a previous run so the recovery replays only
# this run's path (otherwise the controller would replay old landmarks).
rm -f "$WS_DIR/path_landmarks.yaml"

exec ros2 launch vslam_recovery_core paper_simulation_demo.launch.py \
    world:="$WORLD" \
    use_dream:=true \
    r2_threshold:="$R2_TH" \
    strategy:="$STRATEGY" \
    max_window_arc:=1.0 \
    max_window_size:=5000 \
    rviz:=true
