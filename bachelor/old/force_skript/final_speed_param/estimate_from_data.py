#!/usr/bin/env python3
"""
estimate_from_data.py

Estimates parameters alpha, beta, gamma from the velocity model:
    v_dot = alpha * U - beta * v - gamma * sgn(v)
where:
    alpha = a/m_c  (voltage-to-acceleration gain)
    beta = b/m_c   (viscous damping coefficient)
    gamma = c/m_c  (Coulomb friction coefficient)
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


def v_closed_form(t, u, alpha, beta, gamma, s, v0, t0):
    """
    Closed-form solution of the velocity model:
        v_dot = alpha * U - beta * v - gamma * sgn(v)
    
    with constant u and s = sgn(v).
    
    Args:
        t: time array
        u: voltage [V]
        alpha: voltage-to-acceleration gain [m/(s²·V)]
        beta: viscous damping [1/s]
        gamma: Coulomb friction [m/s²]
        s: sign of velocity
        v0: initial velocity at t0
        t0: initial time
    
    Returns:
        v(t) = v_inf + (v0 - v_inf) * exp(-beta * (t - t0))
        where v_inf = (alpha * u - gamma * s) / beta
    """
    tau = t - t0

    if np.isclose(beta, 0.0):
        # Limit case: no viscous damping -> linear growth
        return v0 + (alpha * u - gamma * s) * tau

    # Steady-state velocity
    v_inf = (alpha * u - gamma * s) / beta
    
    return v_inf + (v0 - v_inf) * np.exp(-beta * tau)


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
    """Exponential (IIR) smoothing of 1D data."""
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

        # velocity: use existing or compute
        if v_col is not None:
            v = df[v_col].to_numpy(dtype=float)
            t_s = t_ms.astype(float) / 1000.0
            x_m = np.vectorize(counts_to_meters)(x_counts.astype(float))
        else:
            t_s, x_m, v = compute_v_from_tx(t_ms, x_counts)

        # apply optional alpha smoothing to velocity
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
            try:
                pwm_val = int(df[pwm_col].iloc[0])
            except Exception:
                pwm_val = 0
        else:
            try:
                u_guess, pwm_guess, sign_guess = parse_pwm_filename(path, supply_voltage=supply_voltage)
                pwm_val = pwm_guess
                sign = sign_guess
                u = u_guess
            except Exception:
                pwm_val = 0

        if u == 0.0 and pwm_val is not None:
            u = sign * supply_voltage * pwm_val / 255.0

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
    """
    Compute residuals for all experiments.
    
    Args:
        theta: [alpha, beta, gamma]
        experiments: list of experiment dictionaries
    
    Returns:
        Stacked residual vector
    """
    alpha, beta, gamma = theta
    res_list = []
    
    for exp in experiments:
        t = exp['t']
        v_meas = exp['v']
        u = exp['u']
        t0 = t[0]
        v0 = v_meas[0]
        
        # Sign of velocity direction
        s = float(np.sign(np.mean(v_meas))) if np.any(v_meas) else 1.0

        v_model = v_closed_form(t, u, alpha, beta, gamma, s, v0, t0)
        res_v = v_model - v_meas
        res_list.append(res_v)

    if len(res_list) == 0:
        return np.array([])
    return np.concatenate(res_list)


def main():
    # Cart mass
    m_c = 0.1744  # kg
    
    # defaults
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.normpath(os.path.join(base_dir, '..', 'final_speed_param', 'data'))
    pattern = '*.csv'

    parser = argparse.ArgumentParser(description='Estimate parameters from recorded CSV files')
    parser.add_argument('--data-dir', default=data_dir, help='Directory containing CSV files')
    parser.add_argument('--pattern', default=pattern, help='Glob pattern to match files')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, 
                       help='Exponential smoothing alpha for velocity (0 < alpha <= 1)')
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

    # Initial guess: [alpha, beta, gamma]
    # Reasonable starting values based on physical intuition
    theta0 = [10.0, 5.0, 5.0]

    # Bounds: all parameters physically >= 0
    lower_bounds = [0.0, 0.0, 0.0]
    upper_bounds = [np.inf, np.inf, np.inf]

    result = least_squares(
        fun=residuals_all,
        x0=theta0,
        args=(experiments,),
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        max_nfev=5000,
    )

    print('\n' + '='*60)
    print('OPTIMIZATION RESULTS')
    print('='*60)
    print('Success:', result.success)
    print('Message:', result.message)
    print()
    
    alpha_hat, beta_hat, gamma_hat = result.x
    
    print('Estimated normalized parameters (used in model):')
    print(f'  α (alpha) = {alpha_hat:.6f}  [m/(s²·V)]  (voltage-to-acceleration gain)')
    print(f'  β (beta)  = {beta_hat:.6f}  [1/s]        (viscous damping coefficient)')
    print(f'  γ (gamma) = {gamma_hat:.6f}  [m/s²]      (Coulomb friction coefficient)')
    print()
    
    # Convert back to physical parameters
    a = alpha_hat * m_c  # [N/V]
    b = beta_hat * m_c   # [N·s/m]
    c = gamma_hat * m_c  # [N]
    
    print(f'Corresponding physical parameters (with m_c = {m_c} kg):')
    print(f'  a = α·m_c = {a:.6f}  [N/V]      (motor force gain)')
    print(f'  b = β·m_c = {b:.6f}  [N·s/m]    (viscous friction)')
    print(f'  c = γ·m_c = {c:.6f}  [N]        (Coulomb friction)')
    print('='*60)

    # Group experiments by absolute PWM value for plotting
    pwm_abs_values = sorted(set(abs(exp['pwm']) for exp in experiments))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pwm_abs_values)))
    pwm_to_color = {pwm_abs: colors[i] for i, pwm_abs in enumerate(pwm_abs_values)}
    
    labeled_meas = set()
    labeled_model = set()

    # Plot measured vs model (only first 0.3 seconds)
    plt.figure(figsize=(10, 6))
    for exp in experiments:
        t = exp['t']
        v_meas = exp['v']
        u = exp['u']
        pwm = exp['pwm']
        t0 = t[0]
        v0 = v_meas[0]
        s = float(np.sign(np.mean(v_meas))) if np.any(v_meas) else 1.0
        v_model = v_closed_form(t, u, alpha_hat, beta_hat, gamma_hat, s, v0, t0)

        # Filter to first 0.3 seconds
        mask = t <= 0.3
        t_filtered = t[mask]
        v_meas_filtered = v_meas[mask]
        v_model_filtered = v_model[mask]

        # Get color based on absolute PWM value
        pwm_abs = abs(pwm)
        color = pwm_to_color[pwm_abs]
        
        # Create label with ± notation
        u_abs = abs(u)
        label_base = f'±{u_abs:.1f} V'
        
        label_meas = label_base if pwm_abs not in labeled_meas else None
        label_model = f'{label_base} (model)' if pwm_abs not in labeled_model else None
        
        plt.plot(t_filtered, v_meas_filtered, '-', linewidth=1.2, color=color, label=label_meas)
        plt.plot(t_filtered, v_model_filtered, '--', linewidth=1.5, color=color, label=label_model)
        
        labeled_meas.add(pwm_abs)
        labeled_model.add(pwm_abs)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'velocity_fit.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.show()


if __name__ == '__main__':
    main()
