import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as pp

from math import pi

from dc_motor_equations import integrate_curve

def map_to_float(a):
    return [float(x) for x in a]

def get_filenames_and_pwm():
    result = []
    for filename in glob.glob("./data/pwm-*.csv"):
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


all_data = read_all()

# nutze fit_params (Grid-Search). Dauert je nach Gittergröße.
f_a, f_b, f_c = fit_params(all_data)
print("Fitted params: a=%.4f, b=%.4f, c=%.4f" % (f_a, f_b, f_c))

plot_velocity(all_data)
# plot_set_velocity(all_data)

# Fit params:
# 1.4, 0.09, 0.04

# pwm = 100
# u = 12.0 * pwm / 255
# data = all_data[pwm]
# times = data[:, 2] / 1000000.0
# curve = integrate_curve(f_a, f_b, f_c, u, times)
# pp.plot(times, data[:, 4])
# pp.plot(times, curve)
# pp.show()


