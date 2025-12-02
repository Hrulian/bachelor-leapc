import numpy as np
import matplotlib.pyplot as plt

from my_env import CartPoleEnv, CartPoleEnvConfig


def simulate_constant_force(F=0.0, T=5.0, theta0=0.0, v0=0.0, dt_override=None, debug=True, a=40.0, c=7.0):
    """
    Debug-Version: berechnet und loggt die vorhergesagte Beschleunigung (nach Modell),
    damit man sieht, ob das System wirklich wechselnde Beschleunigungen erhält.
    a = viscous coeff (N s/m), c = coulomb (N)
    """
    # Env-Konfiguration laden
    cfg = CartPoleEnvConfig()

    # Optional Zeitschritt überschreiben
    if dt_override is not None:
        cfg.dt = dt_override

    # Env ohne Rendering
    env = CartPoleEnv(render_mode=None, cfg=cfg)

    # Reset und definierter Startzustand
    obs, _ = env.reset()
    # Zustand: [x, theta, v, dtheta]
    x0 = np.array([0.0, theta0, v0, 0.0], dtype=np.float32)
    env.x = x0
    env.reset_needed = False  # wir haben x manuell gesetzt

    M = cfg.masscart
    m = cfg.masspole
    l = cfg.length
    g = cfg.gravity

    n_steps = int(T / cfg.dt)

    # Logs
    ts = []
    xs = []
    vs = []
    thetas = []
    dthetas = []

    action = np.array([F], dtype=np.float32)

    for k in range(n_steps):
        v_before = env.x[2]
        theta = env.x[1]
        dtheta = env.x[3]

        # Reibung und erwartete Beschleunigung nach dem Modell (ohne Diskretisierungseffekte)
        F_fric = a * v_before + c * np.sign(v_before)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        den = M + m - m * cos_t * cos_t
        accel_pred = (-m * l * sin_t * dtheta * dtheta + m * g * cos_t * sin_t + float(F) - F_fric) / den

        obs, r, terminated, truncated, info = env.step(action)
        v_after = obs[2]

        if debug and (k % max(1, n_steps // 20) == 0):
            print(f"step {k:03d} t={(k+1)*cfg.dt:.3f} v_before={v_before:.4f} accel_pred={accel_pred:.4f} v_after={v_after:.4f} F_fric={F_fric:.4f}")

        ts.append((k + 1) * cfg.dt)
        xs.append(obs[0])
        vs.append(obs[2])
        thetas.append(obs[1])
        dthetas.append(obs[3])

        if terminated or truncated:
            break

    ts = np.array(ts)
    xs = np.array(xs)
    vs = np.array(vs)
    thetas = np.array(thetas)
    dthetas = np.array(dthetas)

    return ts, xs, vs, thetas, dthetas


def main():
    # --- Parameter für den Test ---
    F = 0.0          # konstante Kraft in N
    T = 20.0          # Simulationszeit in s
    theta0 = 0.1    # Pendel anfangs exakt oben
    v0 = 0.0         # Wagen anfänglich in Ruhe

    ts, xs, vs, thetas, dthetas = simulate_constant_force(F=F, T=T, theta0=theta0, v0=v0)

    print("Letzter Zustand:")
    print(f"  x      = {xs[-1]:.4f} m")
    print(f"  v      = {vs[-1]:.4f} m/s")
    print(f"  theta  = {thetas[-1]:.4f} rad")
    print(f"  dtheta = {dthetas[-1]:.4f} rad/s")

    # --- Plots ---
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

    axs[0].plot(ts, xs)
    axs[0].set_ylabel("x [m]")
    axs[0].grid(True)

    axs[1].plot(ts, vs)
    axs[1].set_ylabel("v [m/s]")
    axs[1].grid(True)

    axs[2].plot(ts, thetas)
    axs[2].set_ylabel("theta [rad]")
    axs[2].grid(True)

    axs[3].plot(ts, dthetas)
    axs[3].set_ylabel("thetadot [rad/s]")
    axs[3].set_xlabel("t [s]")
    axs[3].grid(True)

    plt.suptitle(f"Konstante Kraft F = {F} N, theta0 = {theta0} rad, v0 = {v0} m/s")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
