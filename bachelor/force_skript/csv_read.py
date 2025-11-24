from math import pi
from dc_motor_equations import integrate_curve

# ==== CHANGED / ADDED IMPORTS ====
import csv
import glob
import re
import numpy as np
import matplotlib.pyplot as pp

def map_to_float(a):
    return [float(x) for x in a]

def get_filenames_and_pwm():
    result = []
    for filename in glob.glob("./new_estim/pwm-*.csv"):
        pwm = int(re.search('(\\d+)', filename).group(1))
        result.append((pwm, filename))
    return result

def read_params(pwm):
    """
        [i, pwm, time, x, v]
        Zeit-Spalte wird hier in Sekunden (float) normalisiert.
        Erkennung:
         - wenn max Zeit > 1e5 -> vermutlich Mikrosekunden -> /1e6
         - elif max > 1e3 -> vermutlich Millisekunden -> /1e3
         - sonst bereits Sekunden
    """
    if pwm > 0:
        filename = "./data/pwm-%d.csv" % pwm
    else:
        filename = "./data/pwm-rev-%d.csv" % (-pwm)
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(map_to_float(row))
    data = np.array(data)
    rows, cols = data.shape
    ext = np.zeros((rows, cols + 1))
    ext[:,:-1] = data

    # automatische Zeiteinheitserkennung und Umrechnung -> Sekunden
    times_raw = ext[:, 2]
    max_t = times_raw.max()
    if max_t > 1e5:
        times_s = times_raw / 1e6    # Mikrosekunden -> Sekunden
    elif max_t > 1e3:
        times_s = times_raw / 1e3    # Millisekunden -> Sekunden
    else:
        times_s = times_raw.copy()   # schon Sekunden

    # berechne v = dx/dt (times_s in Sekunden)
    for i in range(1, rows):
        dt = times_s[i] - times_s[i - 1]
        if dt == 0:
            ext[i, -1] = 0.0
        else:
            ext[i, -1] = (ext[i, 3] - ext[i - 1, 3]) / dt

    # ersetze Zeitspalte durch Sekunden (praktisch für weitere Verarbeitung)
    ext[:, 2] = times_s
    return ext

def read_all():
    result = {}
    pwms = [-163, -100, 225]
    # pwms = [-250, -200, -100, -70, 90, 150, 230]
    for pwm in pwms: #get_filenames_and_pwm():
        data = read_params(pwm)
        result[pwm] = data
    return result

def plot_velocity(data):
    for value in sorted(data.values(), key=lambda x: -x[0, 1]):
        # times sind jetzt in Sekunden
        times = value[:, 2]
        u = 12.0 * value[0, 1] / 255
        curve = integrate_curve(f_a, f_b, f_c, u, times)
        pp.plot(times[:100], value[:, 4][:100], label=("%.1fV" % u))
        pp.plot(times[:100], curve[:100])
    pp.legend()
    pp.xlabel("Time, s")
    pp.ylabel("Velocity, m/s")
    pp.grid(True)
    pp.show()

def plot_set_velocity(data):
    set_vs = np.zeros((len(data), 2))
    for i, value in enumerate(data.values()):
        velocities = value[-10:-1, 4]
        set_vs[i, 0] = value[0, 1]
        set_vs[i, 1] = sum(velocities) / len(velocities)

    pp.plot(set_vs[:, 0], set_vs[:, 1], 'o')
    pp.grid(True)
    pp.show()


def get_square_error(a1, a2):
    return sum(map(lambda x: x * x, a1 - a2))


def fit_params(data):
    _a = 0.0
    _b = 0.0
    _c = 0.0
    _error = float('inf')

    for a in np.arange(33.0, 36.0, 0.02):
        print("Fitting for a = %.2f; error = %.2f" % (a, _error))
        for b in np.arange(2.0, 2.3, 0.02):
            for c in np.arange(-2.8, -2.4, 0.02):
                error = 0.0
                for item in data.values():
                    pwm = item[0, 1]
                    u = 12.0 * pwm / 255
                    times = item[:, 2] / 1000000.0
                    curve = integrate_curve(a, b, c, u, times)
                    error = error + get_square_error(item[:, 4], curve)
                if error < _error:
                    _error = error
                    _a, _b, _c = a, b, c
    return _a, _b, _c


# ==== NEW: Linear least-squares estimator (fast) ====
def estimate_params_ls(data):
    """
    Schätzt a,b,c durch Linearisierung:
      dv/dt = -a*v + b*u + c*sign(v)
    Aggregiert alle Läufe und löst per np.linalg.lstsq.
    Returns: a, b, c
    """
    Xs = []
    Ys = []
    for pwm, item in data.items():
        times = item[:, 2]            # in Sekunden (read_params stellt sicher)
        v = item[:, 4]
        if len(times) < 3:
            continue
        dt = np.diff(times)
        dv = np.diff(v)
        dvdt = dv / dt
        v_mid = v[1:]
        u = 12.0 * pwm / 255.0
        sign_v = np.sign(v_mid)
        X_run = np.column_stack((-v_mid, np.full_like(v_mid, u), sign_v))
        Xs.append(X_run)
        Ys.append(dvdt)
    if not Xs:
        raise RuntimeError("Keine geeigneten Daten zum Schätzen gefunden.")
    X = np.vstack(Xs)
    Y = np.hstack(Ys)
    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(theta[0]), float(theta[1]), float(theta[2])

# ==== OPTIONAL: Nichtlineare Verfeinerung (länger rechnen) ====
def refine_params_nonlinear(data, init_theta, max_nfev=2000):
    """
    Falls scipy installiert: verfeinere (a,b,c) durch Minimierung
    der Simulations-Residuen (L2) über alle Datensätze.
    max_nfev steuert wie lange gerechnet wird.
    Falls scipy fehlt, gibt init_theta zurück.
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        return init_theta

    def residuals(theta):
        a, b, c = theta
        res_list = []
        for pwm, item in data.items():
            times = item[:, 2]
            measured_v = item[:, 4]
            u = 12.0 * pwm / 255.0
            sim_v = integrate_curve(a, b, c, u, times)
            # nutze komplette Differenz (oder z.B. nur erste N Punkte)
            res_list.append(sim_v - measured_v)
        return np.hstack(res_list)

    res = least_squares(residuals, x0=np.array(init_theta), max_nfev=max_nfev)
    return float(res.x[0]), float(res.x[1]), float(res.x[2])

# ==== REPLACE grid-search usage by LS (+ optional refine) ====
# all_data wird unten weiterhin erzeugt durch read_all()
all_data = read_all()

# schnelle lineare Schätzung
try:
    f_a, f_b, f_c = estimate_params_ls(all_data)
    print("Least-squares initial estimate: a=%.6f, b=%.6f, c=%.6f" % (f_a, f_b, f_c))
except Exception as e:
    print("LS estimation failed:", e)
    f_a, f_b, f_c = 1.0, 0.1, 0.01

# Option: längere nichtlineare Verfeinerung (aktiviere wenn gewünscht)
# Setze max_nfev höher für längeres Rechnen (z.B. 5000 oder 10000)
LONGER_COMPUTE = True
if LONGER_COMPUTE:
    f_a, f_b, f_c = refine_params_nonlinear(all_data, (f_a, f_b, f_c), max_nfev=5000)
    print("Refined params: a=%.6f, b=%.6f, c=%.6f" % (f_a, f_b, f_c))

plot_velocity(all_data)


