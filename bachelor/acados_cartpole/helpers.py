def force_to_pwm(force: float, velocity: float) -> int:
    a = 0.0475
    b = 0.0025
    c = 0.15  # so far all dummy values
    mass = 200

    u = ((force / mass) + a * velocity + c * sgn(velocity)) / b

    # map u in [-24, 24] to pwm in [-255, 255]
    max_u = 24.0
    max_pwm = 255
    scaled = u * (max_pwm / max_u)

    # round and clamp to [-255, 255]
    pwm = int(round(scaled))
    pwm = max(-max_pwm, min(max_pwm, pwm))
    return pwm


def sgn(x: float) -> int:
    # Sign function that returns -1 for negative, 1 for positive, and 0 for zero
    return (x > 0) - (x < 0)


def counts_to_meters(counts: int) -> float:
    # Convert encoder counts to position in meters
    counts_per_revolution = 1024  
    wheel_circumference = 0.04  
    revolutions = counts / counts_per_revolution
    position = revolutions * wheel_circumference
    return position


def countpersecond_to_meterspersecond(counts_per_second: int) -> float:
    # Convert encoder counts per second to velocity in meters per second
    counts_per_revolution = 1024  
    wheel_circumference = 0.04  
    revolutions_per_second = counts_per_second / counts_per_revolution
    velocity = revolutions_per_second * wheel_circumference
    return velocity

