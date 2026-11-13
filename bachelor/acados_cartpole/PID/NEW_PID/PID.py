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

        # Integrator: speichere Rohintegral (∑ e * dt), Ki wird erst beim Ausgang angewendet
        I_candidate = self.I + e * self.dt
        u_unsat = P + D + (self.p.Ki * I_candidate)

        # Sättigen
        u = max(self.p.u_min, min(self.p.u_max, u_unsat))

        # Anti-Windup: nur integrieren, wenn nicht gesättigt oder wenn die Integratoränderung
        # dazu beiträgt, die gesättigte Stellgröße zurück in Richtung des ungesättigten Wertes zu bringen.
        saturated = abs(u - u_unsat) > 1e-12
        if (not saturated) or ((u - u_unsat) * e > 0):
            self.I = I_candidate

        # endgültig mit ggf. aktualisiertem I
        return max(self.p.u_min, min(self.p.u_max, P + self.p.Ki * self.I + D))
