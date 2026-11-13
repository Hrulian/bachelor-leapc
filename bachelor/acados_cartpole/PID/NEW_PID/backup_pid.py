import math
from dataclasses import dataclass


@dataclass
class PIDParams:
    Kp: float
    Ki: float
    Kd: float
    u_min: float = -30.0   # Stellgrößen-Limit (z.B. PWM)
    u_max: float =  30.0
    

class PIDAngle:
    def __init__(self, params: PIDParams, dt: float):
        self.p = params
        self.dt = dt
        self.I = 0.0  # Integratorzustand

    def reset(self):
        self.I = 0.0

    def step(self, x, theta_ref: float = 0.0):
        theta, thetadot = x

        # Fehler (so ist e = theta - theta_ref)
        e = theta - theta_ref

        # P und D (D auf Messung ist ok: -Kd * thetadot)
        P = self.p.Kp * e
        D = -self.p.Kd * thetadot

        # Kandidaten ohne/mit I
        I_new = self.I + self.p.Ki * e * self.dt
        u_unsat = P + D + I_new

        # Sättigen
        u = max(self.p.u_min, min(self.p.u_max, u_unsat))

        # Anti-Windup (korrekte Richtung!)
        saturated = (u != u_unsat)
        if (not saturated) or ((u - u_unsat) * e > 0):
            self.I = I_new

        # endgültig mit ggf. aktualisiertem I
        return max(self.p.u_min, min(self.p.u_max, P + self.I + D))
