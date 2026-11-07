# README — Self-check guide for the DREAM system (robot that recovers when the camera is lost)

This guide is written for people who are **not programmers**. You only need to
**copy-paste** the commands below into a Terminal window and compare what you see
with the **“Expected result”** sections. You do not need to modify any code.

---

## 0. What is this project?

The system simulates a robot that navigates with vSLAM. When the **camera signal is
lost**, the robot must **drive itself back to the starting point** without seeing
anything, relying only on **“landmarks”** recorded earlier along its path.

DREAM is a **smart landmark-recording** algorithm: it records landmarks only where
they are actually needed (turns, speed changes) and **skips** uniform straight
segments. As a result it **reduces the number of landmarks that must be stored**
while still returning accurately.

The project now ships **three DREAM sampling strategies**, selectable from the GUI
or the command line:

- **geometry** (default) — the paper's method: path linearity `(x,y)` + speed
  linearity `s(t)`.
- **velocity** — coefficient of determination `R²` on linear velocity `v(t)` +
  angular velocity `ω(t)` (an experimental alternative).
- **speed_error** — a deadband on velocity, normalized by the max speed
  `|x − x_anchor| / x_max` with a 15% threshold (avoids the `R²` constant-signal
  degeneracy).

The goal of this README: help you **re-run the tests** and **verify for yourself**
that the numbers in the draft (the paper) match the actual results.

---

## 1. One-time setup

### 1.1 Prerequisites (quick check)

Open a Terminal and type each command in turn. If each prints a version, you are
ready:

```bash
python3 --version        # expected: Python 3.10.x
ros2 --help | head -1    # expected: contains "usage: ros2"
ign gazebo --version     # expected: "Gazebo Sim, version 6.18.x"
Xvfb -help 2>&1 | head -1  # expected: contains "use: X"
```

If any command reports `command not found`, that component is not installed (see
section 7 for how to install it).

### 1.2 Go to the correct project folder

Every time you open a new Terminal, the first thing to do is enter the project
folder:

```bash
cd ~/codewhale_recoveryRobotDREAM
```

> If the project is located elsewhere, adjust the path. You know you are in the
> right place when `ls` (list files) shows names such as `run_gazebo_demo.sh`,
> `test_auto_recovery_sim.py`, `publish/`, `paper/`...

### 1.3 “Activate” the ROS 2 environment (type this at the start of every Terminal session)

ROS is like a “toolbox” that must be switched on before use. Type these two lines:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

> It is normal if nothing is printed after you type these lines — that means they
> succeeded. These two lines only take effect in the **current Terminal window**.
> If you open a new Terminal, you must type them again.
>
> **Tip for beginners:** think of these two lines as the “unlock” step to do at the
> start of every working session. A few commands (sections 2.2, 2.3) do not strictly
> require them, but typing them is **harmless** — so just type them to be safe and
> avoid having to remember which command needs them.

### 1.4 Rebuild everything (do this once, or after modifying code)

```bash
colcon build --packages-select recovery_interfaces vslam_recovery_core --symlink-install
```

Success indicator: the last line looks like

```
Summary: 2 packages finished [...]
```

---

## 2. The checks (run from fast → slow)

> General rule: **run them in order**. The checks in sections 2.1 → 2.3 run in a few
> seconds to a few minutes and **do not need a virtual display**. Checks 2.4 and 2.5
> need more time and/or a virtual display.

---

### 2.1 FASTEST check: 4 Auto modes + camera loss at 100% (no Gazebo needed)

This is the core “correctness” check: it runs the **actual control law from the code**
on a simple robot model, loses the camera at 100% of the path, and then measures
whether the robot returns successfully. It checks **all three DREAM strategies**
(geometry, velocity, speed_error) × the 4 Auto modes.

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 test_auto_recovery_sim.py
```

To run just one strategy, pass it as an argument: `python3 test_auto_recovery_sim.py geometry`.

**Expected result** — three `--- Strategy: ... ---` blocks, 4 `PASS` lines each, and a
final `ALL PASS` line:

```
--- Strategy: geometry ---
Mode 1: PASS  n_lm=  9  final_err=0.030 m  wall_margin=0.269 m  obst_clear=0.293 m  coll=None
Mode 2: PASS  n_lm=  9  final_err=0.030 m  wall_margin=0.273 m  obst_clear=0.299 m  coll=None
Mode 3: PASS  n_lm=  4  final_err=0.029 m  wall_margin=0.282 m  obst_clear=0.239 m  coll=None
Mode 4: PASS  n_lm=  4  final_err=0.029 m  wall_margin=0.282 m  obst_clear=0.251 m  coll=None
--- Strategy: velocity ---
... 4 PASS ...
--- Strategy: speed_error ---
... 4 PASS ...
RESULT: ALL PASS
```

How to read it:

- `n_lm` = the number of DREAM landmarks recorded (fewer = more memory saved).
- `final_err` = the error when returning to the starting point (in meters). `0.030` ≈ 3 cm.
- `coll=None` = **no collision** (a collision would print `wall` or `obstacle`).
- `wall_margin` / `obst_clear` = minimum safe distance to the wall / obstacle (m).

**This is the result you compare with the “End-to-end corridor demonstration” table in
the paper.** (Note: the paper's table uses real Gazebo data `0.060 m`; this file uses the
offline model so it is `0.030 m` — both mean the robot returns accurately, they just come
from different data sources.)

---

### 2.2 THEORY check: the scaling law (no robot needed)

This check confirms the mathematical formula in the paper
(`L ≈ 1 − c(κW)²`, `c ≈ 0.0168`).

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 dream_theory_check.py
```

**Expected result:** a line near the end prints `c ~= 0.01681`, and the values in the
`(1-linear)/(kW)^2` column are all around `0.0168` (0.01679 → 0.01683).

> The value `c ≈ 0.0168` must match the paper (an earlier version mistakenly wrote
> `0.0167`, which has been corrected — if you see any document still saying `0.0167`,
> it is an old version).

---

### 2.3 Benchmark DATA check on RECORDED Gazebo data

The `bench_data/` folder contains **real odometry data** recorded from Gazebo for 3
worlds: `odom_A.csv` (straight corridor), `odom_B.csv` (slalom), `odom_C.csv`
(S-shaped). The commands below **recompute the tables in the paper** from that data.

> To **re-record** the odometry yourself (instead of recomputing from the saved CSVs),
> the sensor-enabled worlds need a virtual display:
> `xvfb-run -a python3 run_gazebo_benchmark.py`. The offline commands below
> (`--compare`, `--mc`, `--ablate`) need no display.

#### 2.3.1 DR / recovery-error table (main table)

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 run_gazebo_benchmark.py --compare
```

**Expected result** (note the last `PE_*_cl` column is always `0.030`):

```
[A] ... N_baseline=18 N_dream=2  DR= 88.9%  ... PE_base_cl=0.030 m  PE_dream_cl=0.030 m
[B] ... N_baseline=13 N_dream=8  DR= 38.5%  ... PE_base_cl=0.030 m  PE_dream_cl=0.030 m
[C] ... N_baseline=14 N_dream=3  DR= 78.6%  ... PE_base_cl=0.030 m  PE_dream_cl=0.030 m
```

Meaning: DREAM reduces the number of landmarks by 88.9% / 38.5% / 78.6% while the
**recovery error stays at 0.030 m** (same as baseline). This is the “Landmark count,
data reduction, and closed-loop recovery error” table in the paper.

#### 2.3.2 R²_th threshold sweep table

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 sweep_r2_threshold.py
```

**Expected result:** for every reconstructable threshold (≥ 2 landmarks), the
`PE_dream_cl` column is `0.030 m`; only cases with just `1` landmark print `inf`.

#### 2.3.3 Monte-Carlo (noise) and ablation tables

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 run_gazebo_benchmark.py --mc --runs 20 --noise 0.02 --ablate
```

**Expected result:** the `Monte-Carlo` lines print `PE_dream_cl=0.030±0.000 m`; the
ablation lines print `PE_cl` = `0.030` (except `regress_speed=False` in world A, which
prints `inf`).

#### 2.3.4 Comparison with 4 other methods (on Gazebo data)

```bash
cd ~/codewhale_recoveryRobotDREAM
python3 benchmark_baselines_gazebo.py --mc
```

**Expected result:** prints 3 blocks (World A/B/C), 5 lines each. The last line of each
block is `dream (R2_th=0.98)`. Compare with the “Offline baseline comparison” table in
the paper:
- World A: DREAM `N≈3.0`, `path_max≈0.006` (much smaller than Nyquist `0.206`).
- World B: DREAM `N≈24.0`, `path_max≈0.055` (smaller than Nyquist `0.150`).
- World C: DREAM `N≈5.7`, `path_max≈0.101` (smaller than Nyquist `0.129`).

---

### 2.4 FULL check on real Gazebo (headless, slow ~15 min)

This check runs the **entire real ROS 2 system** (headless Gazebo + ROS nodes) to drive
the robot and return it home. It runs all 4 Auto modes:

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 run_headless_auto_test.py
```

**Expected result** (4 `PASS` lines):

```
Mode 1: PASS  final_err=0.060 m  wall=0.269 m  obst=0.293 m  coll=None
Mode 2: PASS  final_err=0.060 m  wall=0.273 m  obst=0.299 m  coll=None
Mode 3: PASS  final_err=0.060 m  wall=0.282 m  obst=0.239 m  coll=None
Mode 4: PASS  final_err=0.060 m  wall=0.282 m  obst=0.251 m  coll=None
```

> To run faster (one mode only), add `--mode 4` to the end of the command.

---

### 2.5 GUI check (Gazebo GUI + RViz + Dashboard, optional)

This step opens **graphical windows** so you can see the system with your own eyes.
Because the machine has no physical display, you need a **virtual Xvfb display** first.

Open a Terminal and type:

```bash
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp &
export DISPLAY=:99
```

Then, in the same Terminal:

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
./run_gazebo_demo.sh
```

To start with a different DREAM strategy, set `STRATEGY`:

```bash
STRATEGY=geometry ./run_gazebo_demo.sh      # default
STRATEGY=velocity ./run_gazebo_demo.sh
STRATEGY=speed_error ./run_gazebo_demo.sh
```

You can also switch strategy **while it is running**: in the Dashboard press the
**`DREAM Strategy`** button, which cycles `geometry → velocity → speed_error`.

Success indicator: the log shows `Dashboard GUI started.`, `rviz2`, and `ign gazebo`
running; the topics `/odom_raw`, `/odom`, `/cmd_vel` appear.

To **view from another machine** (VNC), open an additional Terminal:

```bash
x11vnc -display :99 -forever -nopw -rfbport 5900
```

then use a VNC client to connect to `machine_address:5900`.

---

## 3. Quick cross-reference: “what the paper says ↔ which command to run”

| Content in the paper | Check command | Number that must match |
|---|---|---|
| DR / recovery-error table (3 worlds) | `python3 run_gazebo_benchmark.py --compare` | DR `88.9/38.5/78.6%`, PE `0.030 m` |
| R²_th threshold sweep table | `python3 sweep_r2_threshold.py` | PE `0.030 m` (except `inf`) |
| Theory table (c ≈ 0.0168) | `python3 dream_theory_check.py` | `c ~= 0.0168` |
| Monte-Carlo table | `run_gazebo_benchmark.py --mc ...` | PE `0.030±0.000 m` |
| Ablation table | `run_gazebo_benchmark.py ... --ablate` | PE `0.030` (except `inf`) |
| 5-method comparison table | `python3 benchmark_baselines_gazebo.py --mc` | N / path_max as in 2.3.4 |
| 4 Auto modes + 100% failure table | `python3 test_auto_recovery_sim.py` (offline) or `run_headless_auto_test.py` (real Gazebo) | error `0.030 m` (offline) / `0.060 m` (Gazebo), `coll=None` |
| Strategy comparison (geometry vs velocity vs speed_error) | `python3 benchmark_strategy_comparison.py` | N / DR / path error per strategy |

---

## 4. How to tell whether a result is “correct” or “wrong”

A check is considered **PASSED** when:

1. The robot **returns** to the starting point: `final_err` or `PE_*_cl` is small (≤ 0.06 m).
2. **No collision**: `coll=None` (or the words `wall`/`obstacle` do not appear).
3. **DREAM uses fewer landmarks than baseline**: `N_dream < N_baseline` (that is the point of data reduction).

If you see:
- `FAIL` or `coll=wall` / `coll=obstacle` → the system has an error that needs fixing.
- `PE = inf` → the path was over-compressed (only 1 landmark left), so it cannot be reconstructed.
- `ImportError ... librcl_action.so` → you **forgot to type the two `source ...` lines** in section 1.3.

---

## 5. Important files (so you know where things live)

- `src/vslam_recovery_core/src/dream_core.py` — the DREAM algorithm (core; three samplers: `DreamSampler`, `VelocityDreamSampler`, `SpeedDeadbandSampler`).
- `src/vslam_recovery_core/src/dream_sampler_node.py` — the ROS 2 node that records DREAM landmarks (`strategy` param).
- `src/vslam_recovery_core/src/recovery_controller.cpp` — the **closed-loop return** controller.
- `src/vslam_recovery_core/src/recovery_recorder_server.cpp` — the node that writes landmarks to YAML.
- `src/vslam_recovery_core/src/recovery_dashboard_gui.py` — the control GUI (4 Auto modes + `DREAM Strategy` toggle).
- `test_auto_recovery_sim.py` — quick check (offline).
- `run_headless_auto_test.py` — full headless Gazebo check.
- `run_gazebo_benchmark.py`, `sweep_r2_threshold.py`, `benchmark_baselines_gazebo.py` — generate the benchmark tables.
- `benchmark_strategy_comparison.py` — compares the three DREAM strategies (geometry / velocity / speed_error) on the recorded odometry.
- `benchmark_footprint.py` — measures the DREAM per-update CPU time and memory.
- `bench_data/odom_A/B/C.csv` — recorded Gazebo odometry data.
- `publish/DREAM-ROS2-IEEE.tex` — the IEEE Access paper draft.
- `publish/DREAM-ROS2-IEEE.pdf` — the compiled PDF.
- `paper/benchmark_results.md` — detailed result notes + reproduction commands.

---

## 6. “All in one” command (if you want it quick)

Copy this entire block and paste it into a Terminal (after completing section 1.3):

```bash
cd ~/codewhale_recoveryRobotDREAM
source /opt/ros/humble/setup.bash
source install/setup.bash
echo "== 1) Offline 4 modes ==" && python3 test_auto_recovery_sim.py
echo "== 2) Theory ==" && python3 dream_theory_check.py
echo "== 3) Benchmark DR/PE ==" && python3 run_gazebo_benchmark.py --compare
echo "== 4) Threshold sweep ==" && python3 sweep_r2_threshold.py
echo "== 5) Compare 5 methods ==" && python3 benchmark_baselines_gazebo.py --mc
```

---

## 7. Installing missing components (only when section 1.1 reports something missing)

```bash
# ROS 2 Humble + Gazebo Fortress (follow the official ROS 2 guide for Ubuntu 22.04)
# Virtual display Xvfb (to open the GUI without a real screen):
sudo apt-get update
sudo apt-get install -y xvfb x11-utils
# Remote GUI viewing via VNC (optional):
sudo apt-get install -y x11vnc
```

> Note: the `sudo apt-get ...` commands require the administrator (root) password. If you
> are inside a sandbox where `sudo` is blocked, you must run these install commands on
> your own machine/container.

---

## 8. If the results do not match

Record the **entire Terminal output** (screenshot or copy) and report it, including:
1. Which step you are on.
2. The exact command you ran.
3. The last 5–10 lines shown on the screen.

That way the person helping you will know exactly what went wrong without guessing.
