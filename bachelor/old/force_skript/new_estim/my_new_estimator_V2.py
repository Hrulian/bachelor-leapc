#!/usr/bin/env python3
"""
Parameteridentifikation eines DC-Motors aus mehreren CSV-Dateien
mit Runge-Kutta-Integration der DGL und nichtlinearer Least-Squares-Optimierung.

Erwartete Dateien im aktuellen Verzeichnis:
    pwm-225.csv        (Vorwärtsrichtung)
    pwm-rev-225.csv    (Rückwärtsrichtung)
    ...

Erwartete Spalten in jeder CSV:
    t, x, v

t: Zeit (z.B. in Sekunden; siehe Kommentar im Code, falls in µs)
x: Position
v: Geschwindigkeit
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ------------------------------------------------------------
# Hilfsfunktionen zum Einlesen der Daten
# ------------------------------------------------------------

def parse_pwm_filename(path, supply_voltage=12.0):
    """
    Parse Dateinamen wie 'pwm-225.csv' oder 'pwm-rev-225.csv'
    und bestimme daraus die Eingangsspannung u.

    supply_voltage: Nennspannung in Volt (z.B. 12 V)
    """
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    parts = name.split('-')

    # Erwartete Muster:
    #   'pwm-225'      -> ['pwm', '225']
    #   'pwm-rev-225'  -> ['pwm', 'rev', '225']
    if len(parts) == 2:
        # vorwärts
        sign = 1.0
        pwm_val = int(parts[1])
    elif len(parts) == 3 and parts[1].lower() == "rev":
        # rückwärts
        sign = -1.0
        pwm_val = int(parts[2])
    else:
        raise ValueError(f"Unerwartetes Dateinamensformat: {basename}")

    # Mapping: PWM [0..255] -> Spannung
    u = sign * supply_voltage * pwm_val / 255.0
    return u, pwm_val, sign


def load_experiments(pattern="pwm*.csv"):
    """
    Lade alle CSV-Dateien, die zum gegebenen Pattern passen,
    und baue eine Liste von Experiment-Dicts auf.
    """
    experiments = []

    for path in sorted(glob.glob(pattern)):
        u, pwm_val, sign = parse_pwm_filename(path)

        df = pd.read_csv(path)

        # Erwartete Spaltennamen: 't', 'x', 'v'
        t = df["t"].to_numpy()

        # WENN deine Zeit t in Mikrosekunden ist, aktiviere die nächste Zeile:
        # t = t * 1e-6

        x = df["x"].to_numpy()
        v = df["v"].to_numpy()

        experiments.append({
            "path": path,
            "pwm": pwm_val,
            "sign": sign,
            "u": u,
            "t": t,
            "x": x,
            "v": v,
        })

    if not experiments:
        raise RuntimeError(f"Keine CSV-Dateien zu Pattern {pattern!r} gefunden")

    return experiments


# ------------------------------------------------------------
# DGL-Modell und Simulation
# ------------------------------------------------------------

def motor_ode(t, y, a, b, c, u):
    """
    Einfaches Motor-Modell.

    Zustand:
        y = [x, v]

    DGL:
        dx/dt = v
        dv/dt = -a * v + b * u + c * sign(v)
    """
    x, v = y
    dxdt = v
    dvdt = -a * v + b * u + c * np.sign(v)
    return [dxdt, dvdt]


def simulate_trajectory(t_data, u, x0, v0, a, b, c):
    """
    Simuliere x(t), v(t) über die Messzeitpunkte t_data
    für konstante Eingangsspannung u und gegebene Parameter a, b, c.
    """
    t_span = (t_data[0], t_data[-1])
    y0 = [x0, v0]

    sol = solve_ivp(
        fun=lambda t, y: motor_ode(t, y, a, b, c, u),
        t_span=t_span,
        y0=y0,
        t_eval=t_data,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"Integration fehlgeschlagen: {sol.message}")

    x_sim = sol.y[0]
    v_sim = sol.y[1]
    return x_sim, v_sim


# ------------------------------------------------------------
# Startwertschätzung aus dv/dt und v(t)
# ------------------------------------------------------------

def estimate_initial_theta_from_v(experiments, v_eps=1e-3, min_points=5):
    """
    Schätzt a, b, c durch Linearisierung aus v(t):

      dv/dt ≈ -a*v + b*u + c*sign(v)

    - nutzt v direkt aus der CSV (gefiltert),
    - aggregiert alle Experimente,
    - ignoriert Punkte, wo |v| < v_eps (tote Zone / Rauschen),
    - gibt (a0, b0, c0) zurück.
    """
    Xs = []
    Ys = []

    for exp in experiments:
        t = exp["t"]
        v = exp["v"]
        u = exp["u"]

        if len(t) < 3:
            continue

        dt = np.diff(t)
        dv = np.diff(v)

        # dt > 0 sicherstellen
        mask_dt = dt > 0
        dt = dt[mask_dt]
        dv = dv[mask_dt]

        if len(dt) < min_points:
            continue

        dvdt = dv / dt           # numerische Ableitung
        v_mid = v[1:][mask_dt]   # zugehörige v-Werte

        # Tote Zone / sehr kleine v-Werte rauswerfen
        mask_v = np.abs(v_mid) > v_eps
        if np.count_nonzero(mask_v) < min_points:
            continue

        dvdt = dvdt[mask_v]
        v_mid = v_mid[mask_v]

        sign_v = np.sign(v_mid)

        X_run = np.column_stack([
            -v_mid,                 # für a
            np.full_like(v_mid, u), # für b (konstante Spannung für dieses Experiment)
            sign_v                  # für c
        ])

        Xs.append(X_run)
        Ys.append(dvdt)

    if not Xs:
        raise RuntimeError("Keine geeigneten Daten für dv/dt-Schätzung gefunden.")

    X = np.vstack(Xs)
    Y = np.hstack(Ys)

    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a0, b0, c0 = theta
    return float(a0), float(b0), float(c0)


# ------------------------------------------------------------
# Fehlerfunktion für least_squares
# ------------------------------------------------------------

def residuals(theta, experiments):
    """
    Residuen-Vektor für alle Experimente.

    theta: [a, b, c]
    experiments: Liste von Dicts mit Keys 't', 'x', 'v', 'u', ...

    Rückgabe: 1D-Array aller Residuen hintereinander
              (z.B. v_sim - v_meas).
    """
    a, b, c = theta
    res_list = []

    for exp in experiments:
        t = exp["t"]
        x_meas = exp["x"]
        v_meas = exp["v"]
        u = exp["u"]

        # Anfangswerte aus erster Messung
        x0 = x_meas[0]
        v0 = v_meas[0]

        x_sim, v_sim = simulate_trajectory(t, u, x0, v0, a, b, c)

        # Hier nehmen wir nur die v-Residuen.
        # Optional könntest du x hier auch einbauen.
        res_v = v_sim - v_meas

        res_list.append(res_v)

    return np.concatenate(res_list)


# ------------------------------------------------------------
# Hauptprogramm
# ------------------------------------------------------------

def main():
    # 1) Daten einlesen
    experiments = load_experiments(pattern="pwm*.csv")
    print(f"{len(experiments)} Experimente geladen:")
    for exp in experiments:
        print(f"  {exp['path']}  -> u = {exp['u']:.3f} V (PWM={exp['pwm']}, sign={exp['sign']:+.0f})")

    # 2) Bessere Startwerte für a, b, c aus v(t) per dv/dt-Linearregression
    try:
        a0, b0, c0 = estimate_initial_theta_from_v(experiments)
        print(f"\nInitiale dv/dt-Schätzung:")
        print(f"  a0 = {a0:.6f}")
        print(f"  b0 = {b0:.6f}")
        print(f"  c0 = {c0:.6f}")
        theta0 = np.array([a0, b0, c0])
    except Exception as e:
        print("\nWarnung: dv/dt-basierte Startwertermittlung fehlgeschlagen:", e)
        theta0 = np.array([30.0, 2.0, -2.0])
        print("Nutze Fallback-Startwert:", theta0)

    # 3) Bounds setzen, z.B. a>0, b>0
    lower_bounds = [0.0, 0.0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]

    # 4) Nichtlineare Least-Squares-Optimierung
    result = least_squares(
        fun=residuals,
        x0=theta0,
        args=(experiments,),
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        max_nfev=5000,
    )

    print("\nOptimierung fertig.")
    print("Erfolg:", result.success)
    print("Nachricht:", result.message)
    print("Anzahl Funktionsauswertungen:", result.nfev)

    a_hat, b_hat, c_hat = result.x
    print(f"\nGefundene Parameter:")
    print(f"  a = {a_hat:.6f}")
    print(f"  b = {b_hat:.6f}")
    print(f"  c = {c_hat:.6f}")

    # 5) Simulation mit den gefundenen Parametern und Plot
    plt.figure(figsize=(10, 6))

    for exp in experiments:
        t = exp["t"]
        x_meas = exp["x"]
        v_meas = exp["v"]
        u = exp["u"]

        x0 = x_meas[0]
        v0 = v_meas[0]

        x_sim, v_sim = simulate_trajectory(t, u, x0, v0, a_hat, b_hat, c_hat)

        label_meas = f"Messung u={u:.2f} V"
        label_sim = f"Sim u={u:.2f} V"

        plt.plot(t, v_meas, label=label_meas)
        plt.plot(t, v_sim, label=label_sim)

    plt.xlabel("t")
    plt.ylabel("v")
    plt.title("Geschwindigkeit v(t): Messdaten vs. simuliertes Modell")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
