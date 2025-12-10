#!/usr/bin/env python3
"""
Simple plotter for recorded AS5600 frames.

Usage:
    python plot_theta.py path/to/pwm-rev-138.csv

The file is expected to contain lines like:
    <-21,3.1416,0,0.000>
or a CSV header followed by those frames. The script will extract the
`theta` (2nd field) and `thetadot` (4th field) and plot them in two
separate subplots. The x-axis is simply the sample index.
"""
import argparse
import sys
from pathlib import Path
import re
import matplotlib.pyplot as plt

FRAME_RE = re.compile(r"<([^>]*)>")


def parse_frame_line(line: str):
    """Return tuple (x, theta, v, thetadot[, tripped]) or None if not parseable."""
    line = line.strip()
    if not line:
        return None
    # allow lines that contain a frame between <...>
    m = FRAME_RE.search(line)
    if not m:
        # try comma-separated without markers
        parts = [p.strip() for p in line.split(',')]
        # if header (contains letters) skip
        if any(re.search('[a-zA-Z]', p) for p in parts):
            return None
        try:
            nums = [float(p) for p in parts if p != '']
            return nums
        except Exception:
            return None
    payload = m.group(1)
    parts = [p.strip() for p in payload.split(',')]
    # skip header-like lines
    if any(re.search('[a-zA-Z]', p) for p in parts):
        return None
    try:
        # convert as many floats as present
        nums = [float(p) for p in parts if p != '']
        return nums
    except Exception:
        return None


def load_theta_and_omega(path: Path):
    thetas = []
    thetadots = []
    xs = []
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open('r', errors='ignore') as f:
        for i, line in enumerate(f):
            parsed = parse_frame_line(line)
            if not parsed:
                continue
            # expect at least 4 fields: x,theta,v,thetadot
            if len(parsed) < 4:
                continue
            # fields: [x, theta, v, thetadot, (optional tripped)]
            try:
                x = parsed[0]
                theta = parsed[1]
                thetadot = parsed[3]
            except Exception:
                continue
            xs.append(i)
            thetas.append(theta)
            thetadots.append(thetadot)
    return xs, thetas, thetadots


def plot(xs, thetas, thetadots, out_path: Path = None):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    axs[0].plot(xs, thetas, marker='.', linestyle='-', markersize=3)
    axs[0].set_ylabel('theta (rad)')
    axs[0].grid(True)
    axs[0].set_title('Theta')

    axs[1].plot(xs, thetadots, marker='.', linestyle='-', markersize=3, color='orange')
    axs[1].set_ylabel('thetadot (rad/s)')
    axs[1].set_xlabel('sample index')
    axs[1].grid(True)
    axs[1].set_title('Theta dot')

    plt.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150)
        print(f'Saved plot to {out_path}')
    else:
        plt.show()


def main(argv=None):
    p = argparse.ArgumentParser(description='Plot theta and thetadot from recorded frames')
    p.add_argument('file', type=Path, nargs='?', default=None,
                   help='Path to recorded file (lines like "<x,theta,v,thetadot,...>"). If omitted the script will search for a .csv file in the same folder as the script.')
    p.add_argument('--out', '-o', type=Path, default=None, help='Optional output image path (PNG)')
    args = p.parse_args(argv)
    # If no file provided, search the script folder for a .csv file
    if args.file is None:
        script_dir = Path(__file__).parent
        csvs = sorted(script_dir.glob('*.csv'))
        if not csvs:
            # fallback to current working directory
            csvs = sorted(Path('.').glob('*.csv'))
        if not csvs:
            print('No CSV file found in script folder or CWD. Please provide a file path.')
            sys.exit(2)
        args.file = csvs[0]
        print(f'No file arg provided — using {args.file}')

    xs, thetas, thetadots = load_theta_and_omega(args.file)
    if not xs:
        print('No valid frames found in', args.file)
        sys.exit(2)
    plot(xs, thetas, thetadots, args.out)


if __name__ == '__main__':
    main()
