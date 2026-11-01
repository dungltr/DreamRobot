#!/usr/bin/env python3
"""
make_figures.py

Generate the four IEEE-Access figures for the DREAM manuscript into publish/:

  fig1_architecture.pdf  : ROS 2 node/topic architecture of dream_sampler_node.
  fig2_linearity.pdf     : path linearity illustration (collinear vs curved).
  fig3_reconstruction.pdf: heading psi(t) true vs baseline/DREAM interpolation.
  fig4_sweep.pdf         : DREAM landmark count vs R^2_th for the three worlds.

Usage:  python3 make_figures.py
"""

import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import run_gazebo_benchmark as rb  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "publish")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Fig. 1 — architecture
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#e3ecf7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02",
                                    fc=fc, ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8)

    def arrow(x1, y1, x2, y2, label="", lx=0.0, ly=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     lw=0.9, color="black"))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                    fontsize=6.5, ha="center", va="center", color="#555")

    box(0.1, 3.4, 1.7, 0.9, "Gazebo Sim\n(world A/B/C)")
    box(2.4, 3.4, 1.9, 0.9, "ros\\_gz\\_bridge")
    box(4.9, 3.4, 1.7, 0.9, "odom\\_sanitizer\n(+noise)")
    box(7.2, 3.4, 2.4, 0.9, "dream\\_sampler\\_node\n(R² event trigger)")

    box(7.2, 1.4, 2.4, 0.9, "recovery\\_recorder\\_server\n(YAML writer)")
    box(4.6, 1.4, 1.9, 0.9, "path\\_landmarks.yaml")
    box(0.1, 1.4, 1.7, 0.9, "recovery\\_controller\n(π-turn + replay)")

    arrow(1.8, 3.85, 2.4, 3.85, "odom")
    arrow(4.3, 3.85, 4.9, 3.85, "/odom\\_raw")
    arrow(6.6, 3.85, 7.2, 3.85, "/odom")
    arrow(7.2, 3.4, 6.0, 2.3, "", lx=-0.6, ly=0.15)
    ax.text(6.55, 2.6, "record\\_landmark\n(service)", fontsize=6.5,
            ha="center", color="#555")
    arrow(8.4, 1.4, 6.5, 1.85, "")
    arrow(1.8, 1.85, 2.0, 3.4, "", lx=0.4, ly=0.1)
    ax.text(0.4, 2.6, "/cmd\\_vel", fontsize=6.5, ha="center", color="#555")
    arrow(8.4, 2.3, 8.4, 3.4, "", lx=0.0, ly=0.0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_architecture.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig1_architecture.pdf")


# ---------------------------------------------------------------------------
# Fig. 2 — path linearity illustration
# ---------------------------------------------------------------------------
def _linearity(xs, ys):
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    cxx = sum(x * x for x in xs) - sx * sx / n
    cyy = sum(y * y for y in ys) - sy * sy / n
    cxy = sum(x * y for x, y in zip(xs, ys)) - sx * sy / n
    tr = cxx + cyy
    if tr < 1e-12:
        return 1.0
    lam1 = 0.5 * (tr + math.sqrt((cxx - cyy) ** 2 + 4 * cxy * cxy))
    return lam1 / tr


def fig_linearity():
    # (a) collinear points (straight) and (b) a curved arc.
    straight = [(i * 0.1, 0.2 * i * 0.1 + 0.05 * math.sin(2 * i))
                for i in range(30)]
    arc = [(math.sin(a), 1 - math.cos(a)) for a in
           [i * 0.05 for i in range(60)]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.4))

    xs = [p[0] for p in straight]
    ys = [p[1] for p in straight]
    ax1.plot(xs, ys, "o", ms=3, color="#1f77b4")
    ax1.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "k--", lw=1)
    ax1.set_title(f"Straight: $L$ = {_linearity(xs, ys):.3f}", fontsize=9)
    ax1.set_aspect("equal")
    ax1.axis("off")

    xs = [p[0] for p in arc]
    ys = [p[1] for p in arc]
    ax2.plot(xs, ys, "o", ms=3, color="#d62728")
    ax2.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "k--", lw=1)
    ax2.set_title(f"Curved:  $L$ = {_linearity(xs, ys):.3f}", fontsize=9)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig.suptitle("Path linearity $L = \\lambda_1/(\\lambda_1+\\lambda_2)$",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_linearity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig2_linearity.pdf")


# ---------------------------------------------------------------------------
# Fig. 3 — heading reconstruction (World C, the S-curve)
# ---------------------------------------------------------------------------
def _interp_psi(recon, idx):
    """Linear interpolation of psi between landmark indices, evaluated per sample."""
    out = []
    j = 0
    for i, (t, x, y, psi, s, v) in enumerate(recon):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            out.append(None)
            continue
        i0, i1 = idx[j], idx[j + 1]
        frac = (i - i0) / (i1 - i0) if i1 > i0 else 0.0
        out.append(recon[i0][3] + frac * (recon[i1][3] - recon[i0][3]))
    return out


def fig_reconstruction():
    raw = []
    with open("bench_data/odom_C.csv") as fh:
        for r in csv.reader(fh):
            if r:
                raw.append(tuple(float(v) for v in r))
    recon = rb.reconstruct(raw)
    b_idx = rb.baseline_sampler(recon)
    d_idx = rb.dream_sampler(recon)

    t = [p[0] for p in recon]
    psi_true = [p[3] for p in recon]
    psi_b = _interp_psi(recon, b_idx)
    psi_d = _interp_psi(recon, d_idx)

    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.plot(t, psi_true, "k-", lw=1.0, label="True $\\psi(t)$")
    ax.plot(t, psi_b, "b--", lw=0.8, label=f"Baseline ($N$={len(b_idx)})")
    ax.plot(t, psi_d, "r-.", lw=0.9, label=f"DREAM ($N$={len(d_idx)})")
    for i in d_idx:
        ax.axvline(t[i], color="red", lw=0.4, alpha=0.4)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$\\psi$ [rad]")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_reconstruction.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig3_reconstruction.pdf")


# ---------------------------------------------------------------------------
# Fig. 4 — landmark count vs R^2_th for the three worlds
# ---------------------------------------------------------------------------
def fig_sweep():
    thresholds = [0.85, 0.90, 0.95, 0.98, 1.00]
    worlds = ["A", "B", "C"]
    counts = {w: [] for w in worlds}
    for w in worlds:
        raw = []
        with open(f"bench_data/odom_{w}.csv") as fh:
            for r in csv.reader(fh):
                if r:
                    raw.append(tuple(float(v) for v in r))
        recon = rb.reconstruct(raw)
        for th in thresholds:
            d_idx = rb.dream_sampler(recon, r2_threshold=th)
            counts[w].append(len(d_idx))

    x = list(range(len(thresholds)))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
    for k, name in enumerate(worlds):
        ax.bar([xi + (k - 1) * w for xi in x], counts[name], width=w,
               color=colors[name], label=f"World {name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel("$R^2_{\\mathrm{th}}$")
    ax.set_ylabel("Number of DREAM landmarks")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_sweep.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig4_sweep.pdf")


def _load_raw(name):
    raw = []
    with open(f"bench_data/odom_{name}.csv") as fh:
        for r in csv.reader(fh):
            if r:
                raw.append(tuple(float(v) for v in r))
    return raw


def _load_recon(name):
    return rb.reconstruct(_load_raw(name))


# ---------------------------------------------------------------------------
# Fig. 5 — PE vs R^2_th (compression-fidelity trade-off) for the three worlds
# ---------------------------------------------------------------------------
def fig_tradeoff():
    thresholds = [0.85, 0.90, 0.95, 0.98, 1.00]
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    for w in "ABC":
        recon = _load_recon(w)
        xs, ys = [], []
        for th in thresholds:
            d_idx = rb.dream_sampler(recon, r2_threshold=th)
            pe = rb.recovery_pe_closed_loop(recon, d_idx)
            if pe == float("inf"):
                ax.plot(th, 2.0, "x", ms=7, color=colors[w], mew=1.5)
            else:
                xs.append(th)
                ys.append(pe)
        ax.plot(xs, ys, "o-", ms=4, lw=1.0, color=colors[w], label=f"World {w}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.05, 2.25)
    ax.set_xlabel("$R^2_{\\mathrm{th}}$")
    ax.set_ylabel("PE (closed-loop) [m]")
    ax.text(1.0, 2.08, "$\\times$ = 1 landmark (non-reconstructable)",
            fontsize=6.5, ha="right", va="top")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig5_tradeoff.pdf")


# ---------------------------------------------------------------------------
# Fig. 6 — PE vs angular-rate noise sigma_w (robustness) for the three worlds
# ---------------------------------------------------------------------------
def fig_noise():
    import random
    sigma_levels = [0.0, 0.01, 0.02, 0.05]
    runs = 20
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    for w in "ABC":
        raw = _load_raw(w)
        means, cis = [], []
        for sw in sigma_levels:
            pes = []
            for run in range(runs):
                rng = random.Random(5000 + run)
                p = rb.reconstruct_noisy(raw, 0.01, sw, rng)
                d_idx = rb.dream_sampler(p)
                pes.append(rb.recovery_pe_closed_loop(p, d_idx))
            m, c = rb._mean_ci(pes)
            means.append(m)
            cis.append(c)
        ax.errorbar(sigma_levels, means, yerr=cis, fmt="o-", ms=4, lw=1.0,
                    color=colors[w], capsize=3, label=f"World {w}")
    ax.set_xlabel("$\\sigma_{\\omega}$ [rad/s]")
    ax.set_ylabel("PE (closed-loop) [m]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_noise.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig6_noise.pdf")


# ---------------------------------------------------------------------------
# Fig. 7 — landmark placement along the S-curve (World C): DREAM concentrates
# at curvature transitions, the baseline samples uniformly in time.
# ---------------------------------------------------------------------------
def fig_landmarks():
    recon = _load_recon("C")
    b_idx = rb.baseline_sampler(recon)
    d_idx = rb.dream_sampler(recon)
    xs = [p[1] for p in recon]
    ys = [p[2] for p in recon]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(xs, ys, "k-", lw=1.0, alpha=0.5, label="path")
    ax.plot([recon[i][1] for i in b_idx], [recon[i][2] for i in b_idx],
            "bo", ms=5, label=f"baseline ($N$={len(b_idx)})")
    ax.plot([recon[i][1] for i in d_idx], [recon[i][2] for i in d_idx],
            "rs", ms=6, label=f"DREAM ($N$={len(d_idx)})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_landmarks.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig7_landmarks.pdf")


def _replay_path_closed_loop(recon, landmarks_idx):
    """Return the (x, y) points followed by the closed-loop reverse replay."""
    if len(landmarks_idx) < 2:
        return []
    forward = [(recon[i][1], recon[i][2]) for i in landmarks_idx]
    x = recon[-1][1]
    y = recon[-1][2]
    yaw = recon[-1][3]
    dt = 1.0 / 50.0
    pts = [(x, y)]

    target = rb._norm_angle(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = rb._norm_angle(target - yaw)
        if abs(err) < 0.03:
            break
        w = rb._clamp(2.0 * err, -0.8, 0.8)
        yaw = rb._norm_angle(yaw + w * dt)
        pts.append((x, y))
        rt += dt

    path = [(x, y)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = rb._polyline_samples(path)

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart = math.hypot(x, y)
        if dstart < 0.03:
            break
        gx, gy = rb._lookahead_goal(x, y, samples, 0.45)
        alpha = rb._norm_angle(math.atan2(gy - y, gx - x) - yaw)
        v_des = 0.12 * max(0.0, math.cos(alpha))
        approach = rb._clamp(dstart / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, 0.12 * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / 0.45
        w_des = rb._clamp(w_des, -0.25, 0.25)
        v = rb._slew(last_v, v_des, 0.20 * dt)
        w = rb._slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = rb._clamp(w, -0.25, 0.25)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = rb._norm_angle(yaw + w * dt)
        pts.append((x, y))
        t += dt
    return pts


# ---------------------------------------------------------------------------
# Fig. 8 — forward path vs DREAM reverse-replay path (RobotPaper Fig. 12/13)
# ---------------------------------------------------------------------------
def fig_return_path():
    recon = _load_recon("C")
    d_idx = rb.dream_sampler(recon)
    ret = _replay_path_closed_loop(recon, d_idx)

    fwd_x = [p[1] for p in recon]
    fwd_y = [p[2] for p in recon]
    ret_x = [p[0] for p in ret]
    ret_y = [p[1] for p in ret]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(fwd_x, fwd_y, "g-", lw=1.2, label="forward path")
    ax.plot(ret_x, ret_y, "r--", lw=1.0, label="closed-loop replay")
    ax.plot(fwd_x[0], fwd_y[0], "ko", ms=7, label="start")
    ax.plot(ret_x[-1], ret_y[-1], "r*", ms=10, label="return end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_return_path.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig8_return_path.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_architecture()
    fig_linearity()
    fig_reconstruction()
    fig_sweep()
    fig_tradeoff()
    fig_noise()
    fig_landmarks()
    fig_return_path()
    return 0


if __name__ == "__main__":
    sys.exit(main())
