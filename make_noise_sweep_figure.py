#!/usr/bin/env python3
"""Regenerate fig_noise_sweep_2state.pdf (two-state success fraction vs noise
level) with correct axis labels (sigma_omega -> sigma_w, sigma_v -> sigma_v in
math symbols) and legends on both panels."""
import csv, os, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import benchmark_sensitivity as bs

def main():
    # S1a: success vs sigma_omega (sigma_v=0.01)
    sw_list = bs.SW_LIST
    sv_list = bs.SV_LIST
    sw_fracs = {n: [] for n in "ABC"}
    for n in "ABC":
        for sw in sw_list:
            pes = bs.two_state_pe(n, 0.01, sw)
            sw_fracs[n].append(bs.succ(pes) * 100)
    # S1b: success vs sigma_v (sigma_omega=0.02)
    sv_fracs = {n: [] for n in "ABC"}
    for n in "ABC":
        for sv in sv_list:
            pes = bs.two_state_pe(n, sv, 0.02)
            sv_fracs[n].append(bs.succ(pes) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
    ax = axes[0]
    for n in "ABC":
        ax.plot(sw_list, sw_fracs[n], "o-", ms=4, lw=1.0, color=colors[n], label=f"World {n}")
    ax.set_xlabel(r"$\sigma_\omega$ (rad/s)")
    ax.set_ylabel("success fraction (%)")
    ax.set_title("(a) vs angular-rate noise", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for n in "ABC":
        ax.plot(sv_list, sv_fracs[n], "s-", ms=4, lw=1.0, color=colors[n], label=f"World {n}")
    ax.set_xlabel(r"$\sigma_v$ (m/s)")
    ax.set_ylabel("success fraction (%)")
    ax.set_title("(b) vs velocity noise", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join("publish", "fig_noise_sweep_2state.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"[OK] {out}")

if __name__ == "__main__":
    main()
