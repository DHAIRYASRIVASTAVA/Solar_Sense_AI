import numpy as np
import pandas as pd

FAULTS = [
    "Normal",
    "Soiling",
    "Partial Shading",
    "Hotspot",
    "Inverter Fault",
    "Degradation",
]

CLF_FEATURES = [
    "hour",
    "irradiance",
    "ambient_temp",
    "humidity",
    "wind_speed",
    "dust_index",
    "panel_age",
    "module_temp",
    "shading_pct",
    "dc_voltage",
    "dc_current",
    "inverter_temp",
    "dc_power_kw",
]

REG_FEATURES = [
    "hour",
    "irradiance",
    "ambient_temp",
    "humidity",
    "wind_speed",
    "dust_index",
    "panel_age",
    "module_temp",
    "shading_pct",
]

RATED_KW = 5.0


def add_dc_power(df):
    out = df.copy()
    out["dc_power_kw"] = out["dc_voltage"] * out["dc_current"] / 1000.0
    return out


def generate_dataset(n=12000, seed=42):
    rng = np.random.default_rng(seed)

    hour = rng.uniform(6, 18, n)
    sun = np.sin(np.pi * (hour - 6) / 12)
    cloud = rng.beta(5, 2, n)
    irradiance = np.clip(1050 * sun * cloud + rng.normal(0, 20, n), 60, 1100)
    ambient = 17 + 15 * sun + rng.normal(0, 2.5, n)
    humidity = np.clip(68 - 32 * sun + rng.normal(0, 7, n), 10, 100)
    wind = np.clip(rng.gamma(2.2, 1.3, n), 0, 20)
    age = rng.uniform(0, 20, n)

    fault_id = rng.choice(len(FAULTS), n, p=[0.40, 0.14, 0.14, 0.10, 0.10, 0.12])

    dust = np.clip(rng.beta(1.4, 9, n) * 100, 0, 35)
    shading = np.clip(rng.exponential(1.5, n), 0, 6)
    hot_rise = np.zeros(n)
    degradation_loss = np.clip(0.005 * age + rng.normal(0, 0.004, n), 0, 0.12)
    inverter_loss = np.zeros(n)
    inverter_temp = 38 + 14 * sun + rng.normal(0, 2.5, n)
    volt_noise = rng.normal(0, 3, n)

    soil = fault_id == 1
    dust[soil] = rng.uniform(55, 100, soil.sum())

    shade = fault_id == 2
    shading[shade] = rng.uniform(18, 75, shade.sum())

    hot = fault_id == 3
    hot_rise[hot] = rng.uniform(14, 32, hot.sum())

    inv = fault_id == 4
    inverter_loss[inv] = rng.uniform(0.2, 0.6, inv.sum())
    inverter_temp[inv] += rng.uniform(14, 30, inv.sum())
    volt_noise[inv] = rng.normal(0, 22, inv.sum())

    deg = fault_id == 5
    age[deg] = rng.uniform(11, 25, deg.sum())
    degradation_loss[deg] = rng.uniform(0.12, 0.28, deg.sum())

    module_temp = ambient + irradiance * 0.026 - wind * 0.7 + hot_rise + rng.normal(0, 1.2, n)

    temp_factor = 1 - 0.0040 * (module_temp - 25)
    dust_factor = 1 - 0.0036 * dust
    shade_factor = 1 - 0.0092 * shading
    hot_factor = 1 - 0.006 * hot_rise
    health_factor = (1 - degradation_loss) * (1 - inverter_loss)

    power = (
        RATED_KW
        * (irradiance / 1000)
        * temp_factor
        * dust_factor
        * shade_factor
        * hot_factor
        * health_factor
        * 0.96
    )
    power = np.clip(power + rng.normal(0, 0.05, n), 0.02, None)

    dc_voltage = 380 * (1 - 0.0034 * (module_temp - 25)) * (1 - 0.0035 * shading) + volt_noise
    dc_current = power * 1000 / (dc_voltage * 0.97)

    df = pd.DataFrame(
        {
            "hour": hour,
            "irradiance": irradiance,
            "ambient_temp": ambient,
            "humidity": humidity,
            "wind_speed": wind,
            "dust_index": dust,
            "panel_age": age,
            "module_temp": module_temp,
            "shading_pct": shading,
            "dc_voltage": dc_voltage,
            "dc_current": dc_current,
            "inverter_temp": inverter_temp,
            "power_kw": power,
            "fault_id": fault_id,
        }
    )
    df = add_dc_power(df)
    df["fault"] = df["fault_id"].map(dict(enumerate(FAULTS)))
    return df.round(3)
