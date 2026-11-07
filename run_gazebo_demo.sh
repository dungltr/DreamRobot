#!/usr/bin/env bash
#
# run_gazebo_demo.sh — launch the DREAM blind-recovery demo with the Gazebo GUI,
# RViz and the live dashboard, so the system can be watched as it records
# landmarks and recovers from a simulated camera failure.
#
# Usage:
#   ./run_gazebo_demo.sh                      # default corridor (S-curve, 2 obstacles);
#                                              # World A/B/C can be switched from the GUI
#   ./run_gazebo_demo.sh world_a_straight.sdf # straight hallway (max compression)
#   ./run_gazebo_demo.sh world_b_slalom.sdf   # 4-obstacle slalom
#   ./run_gazebo_demo.sh world_c_scurve.sdf   # long S-curve
#
# Optional env overrides:
#   R2_TH (default 0.98) — DREAM linearity threshold.
# NOTE: no `set -u` here — ROS setup scripts reference unset vars
# (e.g. AMENT_TRACE_SETUP_FILES) and break under nounset.
set -eo pipefail

WORLD="${1:-recovery_corridor.sdf}"
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

To run a manuscript benchmark (World A/B/C) from the GUI:
  1. Launch with ./run_gazebo_demo.sh (no world argument needed).
  2. In the Dashboard pick the world under "Benchmark Experiment
     (World A/B/C)" and press "Run benchmark (auto-cut -> return)".
  3. The Dashboard restarts Gazebo with the selected world, drives the
     scripted open-loop (v, omega) profile (identical to
     run_gazebo_benchmark.py), auto-cuts the camera at the selected 70-100%,
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
