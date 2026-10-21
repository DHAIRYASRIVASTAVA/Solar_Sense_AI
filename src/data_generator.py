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


DEFAULT_FAULT_PROBS = [0.40, 0.14, 0.14, 0.10, 0.10, 0.12]


def generate_dataset(n=12000, seed=42, site=None, fault_probs=None):
    rng = np.random.default_rng(seed)
    probs = DEFAULT_FAULT_PROBS if fault_probs is None else fault_probs

    hour_draw = rng.uniform(6, 18, n)
    cloud = rng.beta(5, 2, n)
    irr_noise = rng.normal(0, 20, n)
    amb_noise = rng.normal(0, 2.5, n)
    hum_noise = rng.normal(0, 7, n)
    wind_draw = rng.gamma(2.2, 1.3, n)
    age = rng.uniform(0, 20, n)
    fault_id = rng.choice(len(FAULTS), n, p=probs)
    dust = np.clip(rng.beta(1.4, 9, n) * 100, 0, 35)
    shading = np.clip(rng.exponential(1.5, n), 0, 6)
    degradation_loss = np.clip(0.005 * age + rng.normal(0, 0.004, n), 0, 0.12)
    inverter_temp_noise = rng.normal(0, 2.5, n)
    volt_noise = rng.normal(0, 3, n)
    soil_dust = rng.uniform(55, 100, n)
    shade_amount = rng.uniform(18, 75, n)
    hot_amount = rng.uniform(14, 32, n)
    inv_loss_amount = rng.uniform(0.2, 0.6, n)
    inv_temp_extra = rng.uniform(14, 30, n)
    inv_volt_noise = rng.normal(0, 22, n)
    deg_age = rng.uniform(11, 25, n)
    deg_loss_amount = rng.uniform(0.12, 0.28, n)
    module_noise = rng.normal(0, 1.2, n)
    power_noise = rng.normal(0, 0.05, n)

    if site is None:
        hour = hour_draw
        sun = np.sin(np.pi * (hour - 6) / 12)
        irradiance = np.clip(1050 * sun * cloud + irr_noise, 60, 1100)
        ambient = 17 + 15 * sun + amb_noise
        humidity = np.clip(68 - 32 * sun + hum_noise, 10, 100)
        wind = np.clip(wind_draw, 0, 20)
    else:
        hour = np.full(n, float(site["hour"]))
        sun = np.sin(np.pi * (hour - 6) / 12)
        irradiance = np.clip(site["irradiance"] + irr_noise * 0.4, 60, 1100)
        ambient = site["ambient_temp"] + amb_noise * 0.3
        humidity = np.clip(site["humidity"] + hum_noise * 0.3, 10, 100)
        wind = np.clip(site["wind_speed"] + (wind_draw - 2.9) * 0.2, 0, 20)

    hot_rise = np.zeros(n)
    inverter_loss = np.zeros(n)
    inverter_temp = 38 + 14 * sun + inverter_temp_noise

    soil = fault_id == 1
    dust[soil] = soil_dust[soil]

    shade = fault_id == 2
    shading[shade] = shade_amount[shade]

    hot = fault_id == 3
    hot_rise[hot] = hot_amount[hot]

    inv = fault_id == 4
    inverter_loss[inv] = inv_loss_amount[inv]
    inverter_temp[inv] += inv_temp_extra[inv]
    volt_noise[inv] = inv_volt_noise[inv]

    deg = fault_id == 5
    age[deg] = deg_age[deg]
    degradation_loss[deg] = deg_loss_amount[deg]

    module_temp = ambient + irradiance * 0.026 - wind * 0.7 + hot_rise + module_noise

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
    power = np.clip(power + power_noise, 0.02, None)

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
