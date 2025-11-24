import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ------------------------------------------------------------
#Parsing
# ------------------------------------------------------------

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
# Analytische Lösung von v(t)
# ------------------------------------------------------------

def v_closed_form(t, u, a, b, c, s, v0, t0):
    """
    Analytische Lösung der DGL

        dv/dt = -a v + b u + c s

    für konstantes u und konstantes s = sign(v):

        v(t) = v_inf + (v0 - v_inf) * exp(-a (t - t0))

    mit v_inf = (b u + c s) / a.

    Sonderfall a ~ 0 wird separat behandelt.
    """
    tau = t - t0

    d = b * u + c * s
    v_inf = d / a
    return v_inf + (v0 - v_inf) * np.exp(-a * tau)



# ------------------------------------------------------------
# Fehlerfunktion für least_squares: analytische v(t)
# ------------------------------------------------------------

def residuals_closed_form(theta, experiments):
    """
    Residuen-Vektor für alle Experimente basierend auf der
    analytischen Lösung v(t) (ohne Runge-Kutta).

    theta: [a, b, c]
    """
    a, b, c = theta
    res_list = []

    for exp in experiments:
        t = exp["t"]
        v_meas = exp["v"]
        u = exp["u"]

        t0 = t[0]
        v0 = v_meas[0]

        # Konstantes s pro Experiment:
        # z.B. Vorzeichen der mittleren Geschwindigkeit
        s = np.sign(np.mean(v_meas))

        v_model = v_closed_form(t, u, a, b, c, s, v0, t0)

        res_v = v_model - v_meas
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

    
    a0, b0, c0 = 1.0, 1.0, 1.0
    theta0 = [a0, b0, c0]

    # 3) Keine Bounds (erstmal alles frei)
    lower_bounds = [-np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]

    # 4) Nichtlineare Least-Squares-Optimierung mit analytischer v(t)
    result = least_squares(
        fun=residuals_closed_form,
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
    print("\nGefundene Parameter (analytische v(t)):")
    print(f"  a = {a_hat:.6f}")
    print(f"  b = {b_hat:.6f}")
    print(f"  c = {c_hat:.6f}")

    # 5) Simulation mit den gefundenen Parametern und Plot
    plt.figure(figsize=(10, 6))

    for exp in experiments:
        t = exp["t"]
        v_meas = exp["v"]
        u = exp["u"]

        t0 = t[0]
        v0 = v_meas[0]
        s = np.sign(np.mean(v_meas))

        v_model = v_closed_form(t, u, a_hat, b_hat, c_hat, s, v0, t0)

        label_meas = f"Messung u={u:.2f} V"
        label_model = f"Model (analytisch) u={u:.2f} V"

        plt.plot(t, v_meas, "-", linewidth=1.2, label=label_meas)
        plt.plot(t, v_model, "--", linewidth=1.5, label=label_model)

    plt.xlabel("t [s]")
    plt.ylabel("v [units]")
    plt.title("Geschwindigkeit v(t): Messdaten vs. analytisches Modell")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
