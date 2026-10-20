#!/usr/bin/env python3
"""
Parameteridentifikation eines DC-Motors aus mehreren CSV-Dateien
mit Runge-Kutta-Integration der DGL und nichtlinearer Least-Squares-Optimierung.

Erwartete Dateien im aktuellen Verzeichnis:
    pwm-225.csv        (Vorwärtsrichtung)
    pwm-rev-225.csv    (Rückwärtsrichtung)
    ...

Erwartete CSV-Spalten (mit Header):
    t, x, v

wobei:
    - t: Zeit in Sekunden
    - x: Position
    - v: bereits gefilterte Geschwindigkeit (wird direkt verwendet)
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def parse_pwm_filename(path, supply_voltage: float = 12.0):
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
        raise ValueError(f"Unerwartetes Dateinamensformat: {basename!r}")

    # Mapping: PWM [0..255] -> Spannung
    u = sign * supply_voltage * pwm_val / 255.0
    return u, pwm_val, sign


def load_experiments(pattern: str = "pwm*.csv"):
    """
    Lade alle CSV-Dateien, die zum gegebenen Pattern passen,
    und baue eine Liste von Experiment-Dicts auf.

    Erwartete CSV-Spalten (mit Header):
        t, x, v

    Wir nutzen t, x und v exakt so, wie sie in der Datei stehen.
    """
    experiments = []

    for path in sorted(glob.glob(pattern)):
        u, pwm_val, sign = parse_pwm_filename(path)

        df = pd.read_csv(path)

        if not {"t", "x", "v"}.issubset(df.columns):
            raise ValueError(
                f"Erwarte Spalten 't', 'x', 'v' in {path}, "
                f"gefunden: {list(df.columns)}"
            )

        t = df["t"].to_numpy()
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
        t_eval=t_data,   # Lösung genau an Messzeitpunkten
        method="RK45",   # eingebettetes Runge-Kutta-Verfahren 4(5)
    )

    if not sol.success:
        raise RuntimeError(f"Integration fehlgeschlagen: {sol.message}")

    x_sim = sol.y[0]
    v_sim = sol.y[1]
    return x_sim, v_sim



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

        _, v_sim = simulate_trajectory(t, u, x0, v0, a, b, c)

        # v-Residuen
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
        a0, b0, c0 = 1.0, 1.0, 1.0
        print("\nInitiale dv/dt-Schätzung:")
        print(f"  a0 = {a0:.6f}")
        print(f"  b0 = {b0:.6f}")
        print(f"  c0 = {c0:.6f}")
        theta0 = np.array([a0, b0, c0])
    except Exception as e:
        print("\nWarnung: dv/dt-basierte Startwertermittlung fehlgeschlagen:", e)
        theta0 = np.array([30.0, 2.0, -2.0])
        print("Nutze Fallback-Startwert:", theta0)

    # 3) KEINE Bounds mehr, damit negative Startwerte erlaubt sind
    lower_bounds = [-np.inf, -np.inf, -np.inf]
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
    print("\nGefundene Parameter:")
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

        _, v_sim = simulate_trajectory(t, u, x0, v0, a_hat, b_hat, c_hat)

        label_meas = f"Messung u={u:.2f} V"
        label_sim = f"Sim u={u:.2f} V"

        plt.plot(t, v_meas, "-", linewidth=1.2, label=label_meas)
        plt.plot(t, v_sim, "--", linewidth=1.5, label=label_sim)

    plt.xlabel("t [s]")
    plt.ylabel("v [units]")
    plt.title("Geschwindigkeit v(t): Messdaten vs. simuliertes Modell")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
