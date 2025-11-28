#!/usr/bin/env python3
"""
estimate_from_data.py

Searches CSV files in ../final_speed_param/data/, expects files with columns
like 't' (ms) and 'x' (counts). Optionally a PWM column ('pwm' or 'pww') may be
present, otherwise the script will try to parse PWM from the filename using the
same filename conventions as `integrate_estimatorV1.py` (e.g. 'pwm-225.csv' or
'pwm-rev-225.csv').

For files missing a 'v' column the script computes v (m/s) by:
  - converting x (counts) -> x (meters) using counts_per_revolution=1024,
    wheel_circumference=0.04
  - converting t from ms -> s
  - computing v by backward difference v[i] = (x[i]-x[i-1])/(t[i]-t[i-1])

After assembling experiments the script runs a non-linear least-squares fit
for parameters [a,b,c] using the analytical closed-form v(t) from
`integrate_estimatorV1.py` and plots measured vs model velocities.

Usage:
  python3 estimate_from_data.py
  python3 estimate_from_data.py --data-dir ../final_speed_param/data --pattern "*.csv"

"""

import argparse
import glob
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def parse_pwm_filename(path, supply_voltage: float = 24.0):
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    parts = name.split('-')
    if len(parts) == 2:
        sign = 1.0
        pwm_val = int(parts[1]) if parts[1].isdigit() else 0
    elif len(parts) == 3 and parts[1].lower() == 'rev':
        sign = -1.0
        pwm_val = int(parts[2]) if parts[2].isdigit() else 0
    else:
        pwm_val = 0
        sign = 1.0
    u = sign * supply_voltage * pwm_val / 255.0
    return u, pwm_val, sign

def v_closed_form(t, u, a, b, c, s, v0, t0):
    tau = t - t0
    d = b * u + c * s
    # avoid division by zero
    if np.isclose(a, 0.0):
        # linear growth approx
        return v0 + d * tau
    v_inf = d / a
    return v_inf + (v0 - v_inf) * np.exp(-a * tau)


# Conversion utilities (same as used earlier)
def counts_to_meters(counts: float) -> float:
    counts_per_revolution = 1024.0
    wheel_circumference = 0.04
    revolutions = counts / counts_per_revolution
    return revolutions * wheel_circumference


def compute_v_from_tx(t_ms: np.ndarray, x_counts: np.ndarray) -> np.ndarray:
    # convert
    t_s = t_ms.astype(float) / 1000.0
    x_m = np.vectorize(counts_to_meters)(x_counts.astype(float))
    n = len(t_s)
    v = np.zeros(n, dtype=float)
    for i in range(1, n):
        dt = t_s[i] - t_s[i - 1]
        if dt <= 0 or np.isclose(dt, 0.0):
            v[i] = 0.0
        else:
            v[i] = (x_m[i] - x_m[i - 1]) / dt
    return t_s, x_m, v


def alpha_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential (IIR) smoothing of 1D data.

    y[0] = data[0]
    y[i] = alpha * data[i] + (1-alpha) * y[i-1]

    alpha closer to 1.0 -> less smoothing / less lag.
    """
    if alpha is None or alpha >= 1.0 or alpha <= 0.0:
        return data.copy()
    n = len(data)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


# default smoothing alpha (0 < alpha <= 1). Increase toward 1.0 for less smoothing.
DEFAULT_ALPHA = 1


def discover_files(data_dir: str, pattern: str) -> List[str]:
    pattern_path = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(pattern_path))
    return files


def build_experiments(files: List[str], supply_voltage: float = 24.0, alpha: float = DEFAULT_ALPHA) -> List[Dict]:
    experiments = []
    for path in files:
        df = pd.read_csv(path)
        cols = [c.strip().lower() for c in df.columns]

        # detect columns
        time_col = None
        x_col = None
        pwm_col = None
        v_col = None
        for orig_name, low in zip(df.columns, cols):
            if low in ('t', 'time', 't_ms', 'time_ms', 'ms'):
                time_col = orig_name
            if low in ('x', 'pos', 'position', 'counts', 'encoder'):
                x_col = orig_name
            if low in ('v', 'vel', 'velocity'):
                v_col = orig_name
            if low in ('pwm', 'pww', 'u'):
                pwm_col = orig_name

        if time_col is None or x_col is None:
            print(f"Skipping {path}: could not detect time/x columns (found: {df.columns})", file=sys.stderr)
            continue

        t_ms = df[time_col].to_numpy(dtype=float)
        x_counts = df[x_col].to_numpy(dtype=float)

        # velocity: use existing or compute (always convert x->m and t->s)
        if v_col is not None:
            v = df[v_col].to_numpy(dtype=float)
            # convert time to s for consistency
            t_s = t_ms.astype(float) / 1000.0
            x_m = np.vectorize(counts_to_meters)(x_counts.astype(float))
        else:
            t_s, x_m, v = compute_v_from_tx(t_ms, x_counts)

        # apply optional alpha smoothing to velocity (0 < alpha < 1)
        try:
            a = float(alpha)
        except Exception:
            a = DEFAULT_ALPHA
        if 0.0 < a < 1.0:
            v = alpha_filter(np.asarray(v, dtype=float), a)

        # determine pwm/pwm_val/sign and u
        pwm_val = None
        sign = 1.0
        u = 0.0
        if pwm_col is not None:
            # take first value (should be constant)
            try:
                pwm_val = int(df[pwm_col].iloc[0])
            except Exception:
                pwm_val = 0
        else:
            # try to parse from filename
            try:
                u_guess, pwm_guess, sign_guess = parse_pwm_filename(path, supply_voltage=supply_voltage)
                pwm_val = pwm_guess
                sign = sign_guess
                u = u_guess
            except Exception:
                pwm_val = 0

        # if u not set from filename and pwm column present, compute u
        if u == 0.0 and pwm_val is not None:
            u = sign * supply_voltage * pwm_val / 255.0

        # ensure sign (if pwm==0, derive from mean v)
        if pwm_val == 0:
            sign = float(np.sign(np.mean(v))) if np.any(v) else 1.0

        experiments.append({
            'path': path,
            'pwm': pwm_val,
            'sign': sign,
            'u': u,
            't': t_s,
            'x': x_m,
            'v': v,
        })
    return experiments


def residuals_all(theta, experiments):
    a, b, c = theta
    res_list = []
    for exp in experiments:
        t = exp['t']
        v_meas = exp['v']
        u = exp['u']
        t0 = t[0]
        v0 = v_meas[0]
        s = float(np.sign(np.mean(v_meas))) if np.any(v_meas) else 1.0
        v_model = v_closed_form(t, u, a, b, c, s, v0, t0)
        res_v = v_model - v_meas
        res_list.append(res_v)
    if len(res_list) == 0:
        return np.array([])
    return np.concatenate(res_list)


def main():
    # defaults
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.normpath(os.path.join(base_dir, '..', 'final_speed_param', 'data'))
    pattern = '*.csv'

    # allow simple CLI overrides
    parser = argparse.ArgumentParser(description='Estimate parameters from recorded CSV files')
    parser.add_argument('--data-dir', default=data_dir, help='Directory containing CSV files (default: %(default)s)')
    parser.add_argument('--pattern', default=pattern, help='Glob pattern to match files (default: %(default)s)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Exponential smoothing alpha for velocity (0 < alpha <= 1). Larger alpha -> less smoothing.')
    args = parser.parse_args()
    data_dir = args.data_dir
    pattern = args.pattern
    alpha = args.alpha

    print(f'Looking for CSV files in: {data_dir} (pattern={pattern})')
    files = discover_files(data_dir, pattern)
    if not files:
        print('No files found. Exiting.', file=sys.stderr)
        sys.exit(2)
    print(f'Found {len(files)} files')

    experiments = build_experiments(files, alpha=alpha)
    print(f'Built {len(experiments)} experiments:')
    for e in experiments:
        print(f"  {os.path.basename(e['path'])}: pwm={e['pwm']}, u={e['u']:.3f} V, n={len(e['t'])} samples")

    if not experiments:
        print('No valid experiments. Exiting.', file=sys.stderr)
        sys.exit(2)

    # initial guess
    theta0 = [1.0, 1.0, 1.0]

    result = least_squares(
        fun=residuals_all,
        x0=theta0,
        args=(experiments,),
        bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
        method='trf',
        max_nfev=5000,
    )

    print('\nOptimization finished')
    print('Success:', result.success)
    print('Message:', result.message)
    a_hat, b_hat, c_hat = result.x
    print(f'Estimated parameters: a={a_hat:.6f}, b={b_hat:.6f}, c={c_hat:.6f}')

    # Plot measured vs model
    plt.figure(figsize=(10, 6))
    for exp in experiments:
        t = exp['t']
        v_meas = exp['v']
        u = exp['u']
        t0 = t[0]
        v0 = v_meas[0]
        s = float(np.sign(np.mean(v_meas))) if np.any(v_meas) else 1.0
        v_model = v_closed_form(t, u, a_hat, b_hat, c_hat, s, v0, t0)
        plt.plot(t, v_meas, '-', linewidth=1.2, label=f"meas u={u:.2f}V")
        plt.plot(t, v_model, '--', linewidth=1.5, label=f"model u={u:.2f}V")

    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    plt.title('Measured vs. Model velocity')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
