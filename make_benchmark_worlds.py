#!/usr/bin/env python3
"""
make_benchmark_worlds.py

Generate the three DREAM benchmark worlds from the
base recovery_corridor.sdf template:

  * world_a_straight.sdf : 15 m straight hallway, NO obstacles (pure linear
                           motion -> maximum theoretical compression ratio);
  * world_b_slalom.sdf   : 15 m hallway with 4 staggered obstacles forcing a
                           multi-turn slalom;
  * world_c_scurve.sdf   : 20 m long S-curve circuit with 3 obstacles
                           (accumulated drift / long-range return fidelity).

The robot model, plugins and bridge topics are unchanged, so the same
bridge_config.yaml and recovery nodes work for all three worlds.

Usage:  python3 make_benchmark_worlds.py
"""

import os
import re

BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "src", "vslam_recovery_core", "worlds",
                    "recovery_corridor.sdf")
OUT_DIR = os.path.dirname(BASE)


def obstacle_block(name, x, y, r=0.2, h=0.6):
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {h / 2.0} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          <material>
            <ambient>0.9 0.15 0.15 1</ambient>
            <diffuse>0.9 0.15 0.15 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def set_corridor_length(sdf, length):
    """Rewrite the corridor length (walls + end wall + guide lines) to `length` m
    and widen the side walls to y = +/-1.15 m (2.2 m corridor)."""
    half = length / 2.0
    # Left and right walls (pose x = length/2, y = +/-1.15) and their length.
    sdf = sdf.replace("<pose>7.5 0.9 0.2 0 0 0</pose>",
                       f"<pose>{half} 1.15 0.2 0 0 0</pose>")
    sdf = sdf.replace("<pose>7.5 -0.9 0.2 0 0 0</pose>",
                       f"<pose>{half} -1.15 0.2 0 0 0</pose>")
    sdf = sdf.replace("<size>15 0.1 0.4</size>",
                       f"<size>{length} 0.1 0.4</size>")
    # End wall (pose x = length).
    sdf = sdf.replace("<pose>15 0 0.2 0 0 0</pose>",
                       f"<pose>{length} 0 0.2 0 0 0</pose>")
    # Guide lines (pose x = length/2, y = +/-0.15) and their length.
    sdf = sdf.replace("<pose>7.5 0.15 0.001 0 0 0</pose>",
                       f"<pose>{half} 0.15 0.001 0 0 0</pose>")
    sdf = sdf.replace("<pose>7.5 -0.15 0.001 0 0 0</pose>",
                       f"<pose>{half} -0.15 0.001 0 0 0</pose>")
    sdf = sdf.replace("<size>15 0.02 0.002</size>",
                       f"<size>{length} 0.02 0.002</size>")
    return sdf


def replace_obstacles(sdf, obstacles):
    """Replace every existing obstacle_* model block with the given list."""
    sdf = re.sub(r"\n    <model name=\"obstacle_[^\"]*\">.*?</model>",
                 "", sdf, flags=re.DOTALL)
    marker = "    <!-- Differential drive robot, spawned"
    blocks = "\n\n".join(obstacle_block(*o) for o in obstacles) + "\n\n"
    return sdf.replace(marker, blocks + marker)


WHEEL_COLLISION = ("<collision name=\"collision\">\n"
                   "          <geometry><sphere><radius>0.05</radius></sphere></geometry>\n"
                   "        </collision>")


def add_wheel_friction(sdf, mu=0.7, mu2=0.5):
    """Add contact friction (wheel-ground slippage) to the two drive wheels.

    Lower mu/mu2 -> more slip during turns/pivots, which makes the wheel
    odometry drift from the ground-truth pose.
    """
    friction = (f"<collision name=\"collision\">\n"
                f"          <geometry><sphere><radius>0.05</radius></sphere></geometry>\n"
                f"          <surface>\n"
                f"            <friction>\n"
                f"              <ode>\n"
                f"                <mu>{mu}</mu>\n"
                f"                <mu2>{mu2}</mu2>\n"
                f"              </ode>\n"
                f"            </friction>\n"
                f"          </surface>\n"
                f"        </collision>")
    return sdf.replace(WHEEL_COLLISION, friction)


def add_imu_noise(sdf, sigma_w=0.01, sigma_a=0.05):
    """Add Gaussian noise to the IMU angular velocity / linear acceleration.

    This models the sensor noise at the sensor level
    (the recovery odometry is still wheel-encoder based; see odom_sanitizer.py
    for the velocity-feedback noise injection).
    """
    imu = (f"<sensor name=\"imu\" type=\"imu\">\n"
           f"          <always_on>1</always_on>\n"
           f"          <update_rate>100</update_rate>\n"
           f"          <visualize>true</visualize>\n"
           f"          <topic>imu</topic>\n"
           f"          <imu>\n"
           f"            <angular_velocity>\n"
           f"              <x><noise type=\"gaussian\"><mean>0</mean><stddev>{sigma_w}</stddev></noise></x>\n"
           f"              <y><noise type=\"gaussian\"><mean>0</mean><stddev>{sigma_w}</stddev></noise></y>\n"
           f"              <z><noise type=\"gaussian\"><mean>0</mean><stddev>{sigma_w}</stddev></noise></z>\n"
           f"            </angular_velocity>\n"
           f"            <linear_acceleration>\n"
           f"              <x><noise type=\"gaussian\"><mean>0</mean><stddev>{sigma_a}</stddev></noise></x>\n"
           f"              <y><noise type=\"gaussian\"><mean>0</mean><stddev>{sigma_a}</stddev></noise></y>\n"
           f"              <z><noise type=\"gaussian\"><mean>0</mean><stddev>{sigma_a}</stddev></noise></z>\n"
           f"            </linear_acceleration>\n"
           f"          </imu>\n"
           f"        </sensor>")
    old_imu = ("<sensor name=\"imu\" type=\"imu\">\n"
               "          <always_on>1</always_on>\n"
               "          <update_rate>100</update_rate>\n"
               "          <visualize>true</visualize>\n"
               "          <topic>imu</topic>\n"
               "        </sensor>")
    return sdf.replace(old_imu, imu)


def main():
    with open(BASE, "r") as fh:
        base = fh.read()

    worlds = {
        # No obstacles: the experiment is the blind return after camera loss,
        # not obstacle avoidance. The corridor just has to be wide enough for
        # the scripted (v, omega) weave to stay clear of the walls (2.2 m).
        "world_a_straight.sdf": (15.0, []),
        "world_b_slalom.sdf": (15.0, []),
        "world_c_scurve.sdf": (15.0, []),
    }

    for fname, (length, obstacles) in worlds.items():
        sdf = set_corridor_length(base, length)
        sdf = replace_obstacles(sdf, obstacles)
        sdf = add_wheel_friction(sdf)
        sdf = add_imu_noise(sdf)
        out = os.path.join(OUT_DIR, fname)
        with open(out, "w") as fh:
            fh.write(sdf)
        print(f"[OK] wrote {out}  (length={length} m, obstacles={len(obstacles)})")


if __name__ == "__main__":
    main()
