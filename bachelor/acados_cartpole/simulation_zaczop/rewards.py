"""Reward functions for the simulated cartpole swingup.

All rewards operate on the SI state AFTER the step (same timing as
compute_reward in the real script):

    state = (x [m], theta [rad], v [m/s], thetadot [rad/s], tripped)
    force = commanded force [N]

theta convention: 0 = upright, +-pi = hanging down.

To test a new reward, add a function here and decorate it with
@register("my_name"), then run:

    python sim_sac_zop.py --reward my_name
"""

import numpy as np



REWARDS: dict = {}


def register(name: str):
    def deco(fn):
        REWARDS[name] = fn
        return fn
    return deco


def get_reward_fn(name: str):
    try:
        return REWARDS[name]
    except KeyError:
        raise KeyError(f"Unknown reward '{name}'. Available: {sorted(REWARDS)}") from None


@register("default")
def reward_default(state, force) -> float:
    """SI-unit port of compute_reward in my_helpers.py (the one used on hardware)."""
    x, theta, v, thetadot, tripped = state

    # swingup reward, max 0.1 when upright
    reward = abs(np.pi - abs(theta)) / (10.0 * np.pi)

    # position penalty: zero in center, -0.1 at the rail ends
    reward -= 0.1 * (abs(x) / 0.39)

    # discourage high angular velocities
    if abs(thetadot) > 12.0:
        reward = 0.0

    # # bonus for being upright and slow
    # if abs(theta) < 0.15 and abs(thetadot) < 1.5:
    #     reward += 0.5

    return float(max(reward, 0.0))




@register("cosine")
def compute_reward_cos(state, force) -> float:
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    reward  = np.cos(theta)                 # +1 oben (theta=0), -1 unten (theta=+-pi)
    reward -= 0.3 * (x / x_max) ** 2        # Positions-Strafe, glatt
    reward -= 0.001 * thetadot ** 2         # milder Spin-Penalty

    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 0.5
    return float(reward)


# === Quadratische Variante ===============================================
@register("quadratic")
def compute_reward_quad(state, force) -> float:
    x, theta, v, thetadot, tripped = state
   
    x_max = 0.39

    reward  = -1.0 * (theta / np.pi) ** 2   # 0 oben, -1 unten (theta=+-pi)
    reward -= 0.3 * (x / x_max) ** 2        # Positions-Strafe
    reward -= 0.001 * thetadot ** 2         # Spin-Penalty

    if abs(theta) < 0.15 and abs(thetadot) < 1.5:
        reward += 6

    return float(reward)


# === Strikt positive Varianten (kein suicidal-tripping) ==================
# terminated == tripped, daher: positive Rewards => "länger leben" lohnt sich
# => der Agent meidet die Wand von selbst, ohne Strafterm.

@register("cos_pos")
def compute_reward_cos_pos(state, force) -> float:
    """Strikt positiver, glatter Swingup-Reward in [0, 1].

    Keine negativen Terme, keine Cliffs. Position und Spin wirken nur als
    *multiplikative* Faktoren in [0.5, 1], dämpfen also sanft, ohne das
    Aufschwing-Signal je auf 0 zu ziehen.
    """
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    upright  = 0.5 * (1.0 + np.cos(theta))      # 1 oben, 0 unten, dicht & glatt
    centered = np.exp(-(x / x_max) ** 2)        # 1 in der Mitte -> ~0 am Rand
    calm     = np.exp(-(thetadot / 8.0) ** 2)   # 1 bei ruhigem Pendel

    reward = upright * (0.5 + 0.5 * centered) * (0.5 + 0.5 * calm)
    return float(reward)




#TODO: put into realsaczop but check that the state has counts and not meters
# so needs to be converted in real script for sim is ok!!
@register("cos_bonus")
def compute_reward_cos_bonus(state, force) -> float:
    """Dichtes positives Basissignal + glatte (gaußsche) Upright-Prämie, ~[0, 1.5].

    Ersetzt den harten `if abs(theta)<0.15: += const`-Sprung durch eine glatte
    Beule, die SACs Critic deutlich leichter fittet.
    """
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    upright  = 0.5 * (1.0 + np.cos(theta))                                   # [0,1] dicht
    balanced = np.exp(-(theta / 0.25) ** 2) * np.exp(-(thetadot / 2.0) ** 2) * 3 # [0,1] glatte Praemie
    reward = (upright + 0.5 * balanced) * np.exp(-0.5 * (x / x_max) ** 2)
    return float(reward)

@register("cos_bonus_only_theta")
def compute_reward_cos_bonus(state, force) -> float:
    """Dichtes positives Basissignal + glatte (gaußsche) Upright-Prämie, ~[0, 1.5].

    Ersetzt den harten `if abs(theta)<0.15: += const`-Sprung durch eine glatte
    Beule, die SACs Critic deutlich leichter fittet.
    """
    x, theta, v, thetadot, tripped = state
    x_max = 0.39

    upright  = 0.5 * (1.0 + np.cos(theta))                                   # [0,1] dicht
    balanced = np.exp(-(theta / 0.25) ** 2) * np.exp(-(thetadot / 2.0) ** 2) * 3 # [0,3] glatte Praemie
    reward = (upright +  balanced)
    
    # if abs(thetadot) > 12.0:
    #     return 0.0
    
    return float(reward)


# Daempfungsbreite des Rotations-Terms [rad/s]. Kleiner = Rotation wird haerter
# abgewertet. Bei |thetadot| = CALM_RAD_S bleibt noch 1/e ~ 37% des Rewards uebrig.
CALM_RAD_S = 12.0

# Halbwertsbreiten der "balanced"-Beule (Gauss-Breite in exp(-(x/w)^2)). Breiter heisst:
# der dichte Bonus feuert schon bei groesseren Auslenkungen/Drehzahlen spuerbar, nicht
# erst wenn beides fast exakt bei 0 liegt. Bei theta=0.4 rad, thetadot=4 rad/s ergibt
# das mit den alten Breiten (0.25, 2.0) nur 0.004 - mit diesen hier 0.834, also ein
# Signal, das ein Agent frueh im Training tatsaechlich zu fassen bekommt statt eines,
# das nur bei Zufallstreffern exakt am Ziel je ungleich 0 ist.
BALANCE_THETA_WIDTH = 0.25    # rad, vorher 0.25
BALANCE_THETADOT_WIDTH = 2.5  # rad/s, vorher 2.0


@register("cos_bonus_spin")
def compute_reward_cos_bonus_spin(state, force) -> float:
    """Wie `cos_bonus_only_theta`, aber mit dichter Daempfung schneller Rotation.

    Statt einer Schwelle (die nichts brachte: ein Stab, der gerade eben ueber den
    Totpunkt kommt, hat unten schon ~12.8 rad/s, jede Schwelle darueber ist inaktiv)
    wird der gesamte Reward multiplikativ mit exp(-(thetadot/CALM_RAD_S)^2) gedaempft.

    Vorteile gegenueber der Schwellen-Variante:
      - ueberall glatt und streng positiv, kein Cliff, kein Clamping noetig
      - Balancieren bleibt unveraendert bei 4.0 (thetadot=0 -> Faktor 1)
      - der Gradient liegt dort, wo die Policy tatsaechlich haengt: Energieabbau
        von w_top=8 auf 3 bringt +0.32/Schritt statt +0.13 bei der Schwellenversion

    Zeitmittel ueber eine volle Umdrehung (Helikopter-Optimum), w_top = Drehzahl
    am Totpunkt: 8 -> 0.14, 5 -> 0.30, 3 -> 0.46, 1 -> 1.12, 0.5 -> 1.53.
    """
    x, theta, v, thetadot, tripped = state

    upright  =  (1.0 + np.cos(theta))                                       # [0,1] dicht
    balanced = (np.exp(-(theta / BALANCE_THETA_WIDTH) ** 2)
                * np.exp(-(thetadot / BALANCE_THETADOT_WIDTH) ** 2) * 3)  # [0,3] glatte Praemie, jetzt breiter
    #calm     = np.exp(-(thetadot / CALM_RAD_S) ** 2)                             # (0,1] Daempfung

    return float((upright + balanced) * calm)



@register("cos_bonus_spin2")
def compute_reward_cos_bonus_spin2(state, force) -> float:
    """Wie cos_bonus_spin, aber die Drehzahl wird MULTIPLIKATIV nahe oben
    eingekoppelt statt nur einseitig ab 15 rad/s bestraft. Damit:
      - faellt das Slow-Helicopter-Plateau von 0.50 auf ~0.32/Step,
      - entsteht ein MONOTONER Gradient beim Abbremsen (12->0 rad/s),
      - bleibt das Swingup-Signal unten (hohe thetadot) unangetastet,
      - bleibt alles >= 0 (kein Anreiz zum Schienen-Suizid).
    """
    x, theta, v, thetadot, tripped = state

    upright  = 0.5 * (1.0 + np.cos(theta))            # [0,1] dicht
    near_top = np.exp(-(theta / 0.7) ** 2)            # 1 oben -> ~0 unten
    calm     = np.exp(-(thetadot / 12.0) ** 2)         # BREIT: Gradient ueber ganze Drehzahl
    # nur oben zahlt upright anteilig zur Ruhe; unten (swingup) voll erhalten:
    upright  = upright * (1.0 - near_top * (1.0 - calm))

    balanced = np.exp(-(theta / 0.35) ** 2) * np.exp(-(thetadot / 4.0) ** 2) * 3

    return float(max(upright + balanced, 0.0))



# Wie weit runter die Ruhe-Forderung reicht [rad]. Muss deutlich unter pi/2
# bleiben, sonst wird der Aufschwung an der Seite mitgedaempft (=Energieverlust).
# 0.7 rad (~40 deg) laesst theta=pi/2 praktisch unberuehrt.
TOP_WIDTH = 0.7
# Daempfungsbreite NUR im Top-Bereich [rad/s]. Kleiner = Helicopter oben wird
# haerter abgewuergt. 7 statt 12, weil die Gate die Seiten ohnehin schuetzt.
CALM_RAD_S = 4.0
BALANCE_THETA_WIDTH = 0.35
BALANCE_THETADOT_WIDTH = 3.0




@register("cos_bonus_spin3")
def compute_reward_cos_bonus_spin(state, force) -> float:
    """Helicopter-Daempfung NUR nahe oben (positionsgated), damit der Aufschwung
    unten/seitlich volle Drehzahl (= Energie) behalten darf. Global calm bestrafte
    schnelles Durchschwingen an der Seite und liess die Policy zu wenig pumpen.
    """
    x, theta, v, thetadot, tripped = state

    upright  = 0.5 * (1.0 + np.cos(theta))                      # [0,1] dicht
    near_top = np.exp(-(theta / TOP_WIDTH) ** 2)               # 1 oben -> ~0 seitlich
    calm     = np.exp(-(thetadot / CALM_RAD_S) ** 2)
    upright  = upright * (1.0 - near_top * (1.0 - calm))        # nur oben zaehlt Ruhe

    balanced = (np.exp(-(theta / BALANCE_THETA_WIDTH) ** 2)
                * np.exp(-(thetadot / BALANCE_THETADOT_WIDTH) ** 2) * 3)

    return float(max(upright + balanced, 0.0))


#cd /media/julian/Shared/UniAktuell/leap-2/leap-c/bachelor/acados_cartpole/simulation_zaczop && ../../../.venv/bin/python run_parallel_sac.py --rewards cos_bonus_spin --seeds 0 1  --wandb-mode online --group r11cd /media/julian/Shared/UniAktuell/leap-2/leap-c/bachelor/acados_cartpole/simulation_zaczop && ../../../.venv/bin/python run_parallel_sac.py --rewards cos_bonus_spin --seeds 0 1  --wandb-mode online --group r11