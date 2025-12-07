"""
ev_models_simplified.py

Simplified physics-based EV model formulas WITHOUT Cd, surface_frontale_m2, facteur_roue.

Exposes:
    - calculate_objectives(design_vars: dict) -> dict
      Returns the five objectives:
        - poids_total (kg)         (minimize)
        - cout_usd ($)             (minimize)
        - vitesse_max_kmh (km/h)   (maximize)
        - autonomie_km (km)        (maximize)
        - consommation_kwh_100km (kWh/100km) (minimize)

Design variables (keys in design_vars):
    - batterie_kWh
    - puissance_moteur_kW
    - masse_chassis_kg
"""

from typing import Dict
import math

# ----------------------------
# Constants (tunable)
# ----------------------------
# Battery
RHO_BATT_KG_PER_KWH = 6.5       # kg per kWh (approx. 6-8 kg/kWh for EV packs)
COST_BATT_USD_PER_KWH = 130     # $ per kWh (academic approx.)

# Motor
MOTOR_MASS_KG_PER_KW = 0.45     # kg per kW
COST_MOTOR_USD_PER_KW = 30      # $ per kW (prototype estimate)
MOTOR_EFFICIENCY = 0.93         # fraction

# Chassis / structural
CHASSIS_COST_PER_KG = 4.0       # $ per kg structural cost
FIXED_MANUFACTURING_COST = 2500 # $ fixed

# Aerodynamics / rolling - FIXED VALUES (removed from design variables)
RHO_AIR = 1.225                 # kg/m^3
GRAVITY = 9.80665               # m/s^2
C_RR = 0.010                    # fixed rolling resistance coefficient
CD = 0.25                       # fixed drag coefficient
SURFACE_FRONTALE = 2.2          # fixed frontal area (m^2)

# Tire rotating inertia factor
TIRE_ROT_INERTIA_FACTOR = 0.02

# Conversion helpers
KMH_TO_MS = 1000.0 / 3600.0

# Nominal driving speed for consumption estimation
NOMINAL_SPEED_KMH = 90.0
NOMINAL_SPEED_MS = NOMINAL_SPEED_KMH * KMH_TO_MS

# Electrical drivetrain losses
DRIVETRAIN_LOSSES = 0.95  # fraction of power after losses


# ----------------------------
# Helper physics functions
# ----------------------------
def aerodynamic_drag_power(v_ms: float) -> float:
    """Power (W) required to overcome aerodynamic drag at speed v (m/s)."""
    return 0.5 * RHO_AIR * CD * SURFACE_FRONTALE * v_ms ** 3


def rolling_resistance_power(v_ms: float, mass_kg: float) -> float:
    """Power (W) required to overcome rolling resistance at speed v (m/s)."""
    return C_RR * mass_kg * GRAVITY * v_ms


def top_speed_from_power(P_motor_w: float, mass_kg: float) -> float:
    """
    Estimate top speed (m/s) by finding v such that motor available power equals resistive power.
    Solve 0.5*rho*Cd*A*v^3 + Crr*m*g*v = P_available
    """
    P_avail = P_motor_w * MOTOR_EFFICIENCY * DRIVETRAIN_LOSSES
    low, high = 1.0, 120.0  # 1 m/s to 120 m/s (~432 km/h)
    
    # Binary search for speed
    for _ in range(60):
        mid = 0.5 * (low + high)
        P_req = aerodynamic_drag_power(mid) + rolling_resistance_power(mid, mass_kg)
        if P_req > P_avail:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def consumption_kwh_per_100km(mass_kg: float,
                              speed_kmh: float = NOMINAL_SPEED_KMH) -> float:
    """
    Estimate energy consumption in kWh/100km at a nominal steady speed.
    """
    v_ms = speed_kmh * KMH_TO_MS
    P_total_w = aerodynamic_drag_power(v_ms) + rolling_resistance_power(v_ms, mass_kg)
    
    # account for accessory and auxiliary loads
    AUX_LOAD_W = 1000.0
    P_at_wheel = P_total_w + AUX_LOAD_W
    
    # account for motor efficiency and drivetrain losses
    P_batt_w = P_at_wheel / (MOTOR_EFFICIENCY * DRIVETRAIN_LOSSES)
    
    # time to travel 100 km at speed_kmh:
    hours_per_100km = 100.0 / speed_kmh
    energy_kwh_per_100km = (P_batt_w * hours_per_100km) / 1000.0
    
    return energy_kwh_per_100km


# ----------------------------
# Main exported function
# ----------------------------
def calculate_objectives(design_vars: Dict[str, float]) -> Dict[str, float]:
    """
    Given design variables, compute the five objectives.

    Input keys expected:
      - batterie_kWh
      - puissance_moteur_kW
      - masse_chassis_kg

    Returns dictionary with:
      - poids_total (kg)
      - cout_usd ($)
      - vitesse_max_kmh (km/h)
      - autonomie_km (km)
      - consommation_kwh_100km (kWh/100km)
    """
    # Extract variables
    batterie_kWh = float(design_vars["batterie_kWh"])
    puissance_moteur_kW = float(design_vars["puissance_moteur_kW"])
    masse_chassis_kg = float(design_vars["masse_chassis_kg"])

    # --- Mass estimation ---
    mass_battery_kg = batterie_kWh * RHO_BATT_KG_PER_KWH
    mass_motor_kg = puissance_moteur_kW * MOTOR_MASS_KG_PER_KW
    
    # total vehicle mass = chassis + battery + motor + fixed extras
    FIXED_ACCESSORIES_MASS = 160.0  # includes wheels, etc.
    poids_total = masse_chassis_kg + mass_battery_kg + mass_motor_kg + FIXED_ACCESSORIES_MASS

    # --- Cost estimation ---
    cost_battery = batterie_kWh * COST_BATT_USD_PER_KWH
    cost_motor = puissance_moteur_kW * COST_MOTOR_USD_PER_KW
    cost_chassis = masse_chassis_kg * CHASSIS_COST_PER_KG
    
    # integration and manufacturing costs
    integration_cost = 0.5 * batterie_kWh + 10.0 * puissance_moteur_kW
    cout_usd = cost_battery + cost_motor + cost_chassis + integration_cost + FIXED_MANUFACTURING_COST

    # --- Consumption estimation (kWh / 100 km) ---
    consommation_kwh_100km = consumption_kwh_per_100km(
        mass_kg=poids_total,
        speed_kmh=NOMINAL_SPEED_KMH
    )

    # --- Range estimation (km) ---
    # Range = usable battery energy / consumption per km
    USABLE_BATT_FRACTION = 0.9
    usable_kwh = batterie_kWh * USABLE_BATT_FRACTION
    consumption_per_km = consommation_kwh_100km / 100.0
    autonomie_km = usable_kwh / max(consumption_per_km, 1e-6)

    # --- Top speed estimation (km/h) ---
    P_motor_w = puissance_moteur_kW * 1000.0
    v_ms = top_speed_from_power(P_motor_w, poids_total)
    vitesse_max_kmh = v_ms * 3.6

    # Wrap results
    return {
        "poids_total": float(poids_total),
        "cout_usd": float(cout_usd),
        "vitesse_max_kmh": float(vitesse_max_kmh),
        "autonomie_km": float(autonomie_km),
        "consommation_kwh_100km": float(consommation_kwh_100km),
    }


# Quick local test when run as script
if __name__ == "__main__":
    test = {
        "batterie_kWh": 60.0,
        "puissance_moteur_kW": 150.0,
        "masse_chassis_kg": 700.0,
    }
    res = calculate_objectives(test)
    print("Test objectives:")
    for k, v in res.items():
        print(f"  {k}: {v:.3f}")