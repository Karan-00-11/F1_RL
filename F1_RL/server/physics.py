import numpy as np

NUM_OF_LAPS = 1
DRAG_COEFFICIENT = 0.9
DOWNFORCE_COEFFICIENT = 2.37
MASS = 800 + NUM_OF_LAPS * 1.85  # including driver, in kg

def caculate_acceleration(force: float, mass: float = MASS) -> float:
    """Calculate acceleration using Newton's second law.
    
    Args:
        force: Total force in Newtons
        mass: Vehicle mass in kg
        
    Returns:
        Acceleration in m/s²
    """
    return force / mass


def update_velocity(velocity: float, acceleration: float, dt: float = 0.0125) -> float:
    """Update velocity based on acceleration and time step.
    
    Args:
        velocity: Current velocity in m/s
        acceleration: Acceleration in m/s²
        dt: Time step in seconds
    Returns:
        Updated velocity in m/s """
    return velocity + acceleration * dt  ## dt(0.05) / Based on number of substeps


def aero_efficiency(
    velocity: float,
    frontal_area: float = 1.5,
    air_density: float = 1.225,
    mode: int = 0
) -> float: 
    
    '''
    k = 1/2 * rho * Cd * A 
    k_2026 = 2.185 (publicly-determined value in general agreement with F1 data)
        
    # Another Method
    if aero_mode == 0: # X-mode
        k = 2.185 * 0.4 # 40% reduction in downforce in straight line mode
        return k * velocity ** 2
    '''
    k = 0.5 * air_density * frontal_area

    if mode == 1: # X-mode
        k = k * DRAG_COEFFICIENT
        return k * velocity ** 2
    
    return k * DOWNFORCE_COEFFICIENT * velocity ** 2

def lateral_acceleration_limit(
    velocity: float,
    slip_ratio: float,
    slip_angle_rad: float,
    temp: float,
    mu: float,
    gravity: float = 9.81,
) -> float:
    """Calculate maximum lateral acceleration.
    
    Args:
        velocity: Vehicle velocity in m/s
        mu: Tire friction coefficient
        downforce: Aerodynamic downforce in Newtons
        mass: Vehicle mass in kg
        gravity: Gravitational acceleration
        
    Returns:
        Maximum lateral acceleration in m/s²
    """
    grip = calucate_grip(mu, mode=0, velocity=velocity, temp=temp, slip_ratio=slip_ratio, slip_angle_rad=slip_angle_rad)
    downforce = aero_efficiency(velocity, mode=0)  # Get downforce at current speed
    # Total vertical force = weight + downforce
    vertical_force = MASS * gravity + downforce
    
    # Maximum lateral force = grip * vertical force
    max_lateral_force = grip * vertical_force
    
    # Maximum lateral acceleration
    return max_lateral_force / MASS

def calucate_grip(
        mu: float, 
        mode: int, 
        velocity: float, 
        temp: float, 
        slip_ratio: float, 
        slip_angle_rad: float,
        tire_wear: float = 1.0
    ) -> float:
    
    # mu_base = 1.6 # Base grip coefficient for F1 tires
    mu_base = mu
    downforce = aero_efficiency(velocity, mode=mode)
    optimal_temp = 95  # Optimal tire temperature in Celsius
    surface_factor = 1.0 # Surface factor (1.0 for dry, <1.0 for wet)
    temp_factor = np.exp(-((temp - optimal_temp) / 18.0) ** 2)

    wear_factor = 0.7 + 0.3 * float(np.clip(tire_wear, 0.0, 1.0)) # Assuming new tires for now
    slip_factor = np.exp(-2.0 * abs(slip_ratio)) * np.exp(-3.0 * abs(slip_angle_rad)) # Assuming no slip for now

    fz = MASS * 9.81 + downforce
    fz_ref = MASS * 9.81
    load_factor = (fz / fz_ref) ** (-0.05)
    mu = mu_base * temp_factor * wear_factor * slip_factor * surface_factor * load_factor
    return float(np.clip(mu, 0.4, 1.6))

def tire_degradation(
        mu: float,
        wear: float,
        velocity: float,
        temp:float, 
        throttle: int, 
        steering: int, 
        brake: float
    ) -> float:
    
    wear += 0.0001 * velocity * (1 + throttle) * (1 + abs(steering)) * (1 + brake) * (1 - temp / 100.0)
    mu = mu * (1 - wear)  # Decrease grip as wear increases
    return mu, wear


def brake_force(
        brake: float,
        velocity: float,
        mu: float,
        mode: int,
        gravity: float = 9.81
) ->float:

    brake_input = float(np.clip(brake, 0.0, 1.0))
    downforce = aero_efficiency(velocity, mode=mode)
    fz = MASS * gravity + downforce
    fx_tire_max = mu * fz
    fx_brake = brake_input * fx_tire_max
    
    return float(np.clip(fx_brake, 0.0, fx_tire_max))


def update_speed(
        velocity: float,
        throttle: float,
        brake: float,
        mu: float,
        soc: float,
        mode: int = 0,
        battery_status: str = "NEUTRAL",
        dt: float = 0.0125
    ) -> tuple[float, float]:

    battery_status = str(battery_status).upper()
    if battery_status not in {"DEPLOY", "REGEN", "NEUTRAL"}:
        battery_status = "NEUTRAL"

    f_brake = brake_force(brake=brake, velocity=velocity, mode=mode, mu=mu)
    f_drag = aero_efficiency(velocity, mode=mode)

    f_deploy = 0.0
    f_regen_lift = 0.0

    # DEPLOY: force deploy only. REGEN: force regen only. NEUTRAL: allow both.
    can_deploy = battery_status in {"DEPLOY", "NEUTRAL"}
    can_regen = battery_status in {"REGEN", "NEUTRAL"}

    if can_deploy:
        # Avoid simultaneous deploy and braking in the same step.
        deploy_throttle = float(np.clip(throttle, 0.0, 1.0)) if brake <= 1e-6 else 0.0
        soc, f_deploy = battery_deploy(
            soc=soc,
            velocity=velocity,
            throttle=deploy_throttle,
            mu=mu,
            mode=mode,
            dt=dt,
        )

    if can_regen:
        # Brake regen is energy bookkeeping; lift-off regen adds mild decel.
        soc, f_regen_lift = battery_recover(
            velocity=velocity,
            throttle=throttle,
            brake_force_n=f_brake,
            mu=mu,
            mode=mode,
            soc=soc,
            dt=dt,
        )

    f_net = (f_deploy - f_brake - f_drag - f_regen_lift)
    a_net = f_net / MASS
    v_next = max(0.0, velocity + a_net * dt)
    return v_next, soc
    

def battery_deploy(
        soc: float,
        velocity: float,
        throttle: float,
        mu: float,
        mode: int,
        dt: float = 0.0125,
        deploy_power_max_w: float = 330_000.0,
        battery_capacity_j: float = 8_500_000.0,
        deploy_efficiency: float = 0.9,
        soc_min: float = 0.05,
        gravity: float = 9.81,
    ) -> tuple[float, float]:

    soc = float(np.clip(soc, 0.0, 1.0))
    throttle = float(np.clip(throttle, 0.0, 1.0))

    if dt <= 0.0 or throttle <= 0.0 or soc <= soc_min:
        return soc, 0.0

    downforce = aero_efficiency(velocity, mode=mode)
    fz = MASS * gravity + downforce
    fx_tire_max = mu * fz

    fx_power_cap = deploy_power_max_w / max(velocity, 0.5)
    fx_req = throttle * fx_tire_max

    f_deploy = min(fx_req, fx_power_cap, fx_tire_max)
    energy_used = (f_deploy * velocity * dt) / max(deploy_efficiency, 1e-6)
    energy_available = max(0.0, (soc - soc_min) * battery_capacity_j)
    if energy_used > energy_available:
        energy_used = energy_available
        f_deploy = (energy_used * deploy_efficiency) / max(velocity * dt, 1e-6)

    soc = max(soc_min, soc - energy_used / battery_capacity_j)
    return soc, f_deploy


def battery_recover(
        velocity: float,
        throttle: float,
        brake_force_n: float,
        mu: float,
        mode: int,
        soc: float,
        dt: float = 0.0125,
        regen_power_max_w: float = 240_000.0,  # Max regenerative braking power in Watts
        rear_brake_bias: float = 0.6,  # Percentage of braking force
        lift_regen_gain: float = 0.10,
        battery_capacity_j: float = 8_500_000.0,
        regen_efficiency: float = 0.9,
        v_min_regen: float = 3.0,
        gravity: float = 9.81,
    ) -> tuple[float, float]:

    soc = float(np.clip(soc, 0.0, 1.0))
    throttle = float(np.clip(throttle, 0.0, 1.0))

    if dt <= 0.0 or velocity < v_min_regen:
        return soc, 0.0

    downforce = aero_efficiency(velocity, mode=mode)
    fz = MASS * gravity + downforce
    fx_tire_max = mu * fz
    fx_power_cap = regen_power_max_w / max(velocity, 0.5)

    # Brake regen contributes energy only; braking force is already applied elsewhere.
    f_regen_brake = min(max(0.0, brake_force_n) * rear_brake_bias, fx_power_cap, fx_tire_max)

    # Lift-off regen adds a small decel force when throttle is reduced.
    lift_factor = max(0.0, 1.0 - throttle)
    f_regen_lift_req = lift_regen_gain * lift_factor * fz
    f_regen_lift = min(
        f_regen_lift_req,
        max(0.0, fx_power_cap - f_regen_brake),
        max(0.0, fx_tire_max - max(0.0, brake_force_n)),
    )

    f_regen_total = f_regen_brake + f_regen_lift
    energy_recovered = f_regen_total * velocity * dt * regen_efficiency
    soc = min(1.0, soc + energy_recovered / battery_capacity_j)
    return soc, f_regen_lift


def fuel_consumption_rate(
    throttle: float,
    rpm: float,
    base_rate: float = 0.0001,
) -> float:
    
    """Calculate fuel consumption rate per timestep.
    
    Args:
        throttle: Throttle position [0, 1]
        rpm: Engine RPM
        base_rate: Base consumption rate
        
    Returns:
        Fuel consumed [0, 1] per timestep (fraction of tank)
    """
    throttle_factor = 0.3 + 0.7 * throttle  # Idle consumption + throttle
    rpm_factor = rpm / 15000.0  # Normalize by max RPM
    
    return base_rate * throttle_factor * rpm_factor


def Update_fuel_and_mass(
        throttle: float,
        rpm: float,
)-> float:
    """
    Update fuel mass and total mass based on consumption.
    """
    global MASS
    fuel_consumed = (NUM_OF_LAPS * 1.85) * fuel_consumption_rate(throttle, rpm)
    MASS -= fuel_consumed
    return MASS