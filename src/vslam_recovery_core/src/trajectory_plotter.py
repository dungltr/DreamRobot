#!/usr/bin/env python3
"""
trajectory_plotter.py

Offline plotter that reads the recorded path landmarks (path_landmarks.yaml)
and exports high-resolution figures in the style of the paper's Figures 11, 12
and 13:

  * trajectory_2d_<timestamp>.png / .pdf  -> robot path (x, y), reconstructed
      from the recorded distance (d) and yaw (psi) between landmarks;
  * velocity_yaw_<timestamp>.png / .pdf   -> recorded angular velocity (w) and
      yaw angle (psi) versus landmark index (Figure 11 style).

Design notes:
  * the Figure/ directory is created automatically if missing (mkdir -p);
  * filenames are timestamped so existing plots are never overwritten;
  * matplotlib uses the DejaVu Sans font (present on the system and covering
    Vietnamese diacritics) and disables the unicode-minus glyph to avoid
    "Glyph missing" / font errors;
  * empty or missing YAML input is handled gracefully.

Usage:
  python3 trajectory_plotter.py [--yaml path_landmarks.yaml] [--output-dir Figure] [--dpi 300]
"""

import argparse
import math
import os
import re
import sys
import time

import matplotlib
matplotlib.use('Agg')  # headless backend (no display required)
import matplotlib.pyplot as plt

# Robust font configuration (avoids Vietnamese-font / glyph-missing errors).
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

_LM_RE = re.compile(r"^landmark_(\d+):\s*\[([^\]]*)\]")

# Workspace root, derived from this file location (<ws>/src/vslam_recovery_core/src/).
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__)))))


def parse_yaml(path):
    """Parse the `landmark_XXXX: [v, w, d, T, psi]` file into a list of dicts."""
    landmarks = []
    if not path or not os.path.exists(path):
        return landmarks
    with open(path, 'r') as fh:
        for line in fh:
            match = _LM_RE.match(line.strip())
            if not match:
                continue
            values = [float(x) for x in match.group(2).split(',') if x.strip() != '']
            if len(values) >= 5:
                landmarks.append({
                    'index': int(match.group(1)),
                    'v': values[0], 'w': values[1], 'd': values[2],
                    'T': values[3], 'psi': values[4],
                })
    landmarks.sort(key=lambda lm: lm['index'])
    return landmarks


def reconstruct_trajectory(landmarks):
    """Reconstruct the 2D path from the recorded (d, psi) landmarks."""
    x, y = 0.0, 0.0
    xs, ys = [x], [y]
    for lm in landmarks:
        x += lm['d'] * math.cos(lm['psi'])
        y += lm['d'] * math.sin(lm['psi'])
        xs.append(x)
        ys.append(y)
    return xs, ys


def unique_path(output_dir, name, ext):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_dir, f'{name}_{stamp}.{ext}')


def export_figure(fig, output_dir, name):
    png_path = unique_path(output_dir, name, 'png')
    pdf_path = unique_path(output_dir, name, 'pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Export recovery trajectory figures.')
    parser.add_argument('--yaml', default=os.path.join(_WORKSPACE, 'path_landmarks.yaml'),
                        help='Path to path_landmarks.yaml')
    parser.add_argument('--output-dir', default=os.path.join(_WORKSPACE, 'Figure'),
                        help='Directory for exported figures')
    parser.add_argument('--dpi', type=int, default=300, help='Raster output DPI')
    args = parser.parse_args()

    if not args.yaml:
        print('[ERROR] Empty YAML path.')
        return 2

    # Ensure the Figure directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    landmarks = parse_yaml(args.yaml)
    if not landmarks:
        print(f'[WARN] No landmarks found in "{args.yaml}"; nothing to plot.')
        return 1

    indices = [lm['index'] for lm in landmarks]
    w_values = [lm['w'] for lm in landmarks]
    psi_values = [lm['psi'] for lm in landmarks]
    xs, ys = reconstruct_trajectory(landmarks)

    # Figure 12/13 style: 2D trajectory.
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(xs, ys, 'b-', linewidth=1.2)
    ax1.plot(xs[0], ys[0], 'go', label='Start')
    ax1.plot(xs[-1], ys[-1], 'ro', label='End')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Recovered trajectory (from path landmarks)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Figure 11 style: angular velocity and yaw angle.
    fig2, (ax_w, ax_psi) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_w.plot(indices, w_values, 'b-', linewidth=1.2)
    ax_w.set_ylabel('w [rad/s]')
    ax_w.set_title('Recorded angular velocity vs landmark index')
    ax_w.grid(True, alpha=0.3)
    ax_psi.plot(indices, psi_values, 'g-', linewidth=1.2)
    ax_psi.set_xlabel('landmark index')
    ax_psi.set_ylabel('psi [rad]')
    ax_psi.set_title('Recorded yaw angle vs landmark index')
    ax_psi.grid(True, alpha=0.3)
    fig2.tight_layout()

    png1, pdf1 = export_figure(fig1, args.output_dir, 'trajectory_2d')
    png2, pdf2 = export_figure(fig2, args.output_dir, 'velocity_yaw')

    print(f'[OK] {len(landmarks)} landmarks exported:')
    print(f'  {png1}')
    print(f'  {pdf1}')
    print(f'  {png2}')
    print(f'  {pdf2}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
