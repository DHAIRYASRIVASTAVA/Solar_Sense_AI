import numpy as np

from src.data_generator import FAULTS, generate_dataset

PANEL_COLS = 10
PANEL_ROWS = 6

SENSORS = {
    "hour": ("Time of day", 6.0, 18.0, 0.25, "h"),
    "irradiance": ("Pyranometer (sunlight)", 60.0, 1100.0, 10.0, "W/m²"),
    "ambient_temp": ("Ambient temperature", 0.0, 50.0, 0.5, "°C"),
    "humidity": ("Humidity", 10.0, 100.0, 1.0, "%"),
    "wind_speed": ("Wind speed", 0.0, 20.0, 0.5, "m/s"),
    "t1": ("Thermal sensor T1", 10.0, 110.0, 0.5, "°C"),
    "t2": ("Thermal sensor T2", 10.0, 110.0, 0.5, "°C"),
    "t3": ("Thermal sensor T3", 10.0, 110.0, 0.5, "°C"),
    "s1": ("Light sensor S1 (shade)", 0.0, 100.0, 1.0, "%"),
    "s2": ("Light sensor S2 (shade)", 0.0, 100.0, 1.0, "%"),
    "s3": ("Light sensor S3 (shade)", 0.0, 100.0, 1.0, "%"),
    "s4": ("Light sensor S4 (shade)", 0.0, 100.0, 1.0, "%"),
    "dust_index": ("Dust sensor", 0.0, 100.0, 1.0, ""),
    "dc_voltage": ("Voltage sensor", 200.0, 450.0, 1.0, "V"),
    "dc_current": ("Current sensor", 0.0, 25.0, 0.1, "A"),
    "inverter_temp": ("Inverter temperature sensor", 25.0, 90.0, 0.5, "°C"),
    "panel_age": ("Panel age (nameplate)", 0.0, 25.0, 0.5, "years"),
}

TABS = {
    "Weather": ["hour", "irradiance", "ambient_temp", "humidity", "wind_speed"],
    "Thermal": ["t1", "t2", "t3"],
    "Light": ["s1", "s2", "s3", "s4"],
    "Dust and electrical": ["dust_index", "dc_voltage", "dc_current", "inverter_temp", "panel_age"],
}

POSITIONS = {
    "t1": (1.8, 1.3),
    "t2": (5.0, 3.0),
    "t3": (8.2, 4.7),
    "s1": (0.5, 0.5),
    "s2": (9.5, 0.5),
    "s3": (0.5, 5.5),
    "s4": (9.5, 5.5),
    "dust_index": (5.0, 0.5),
    "dc_voltage": (3.6, 6.8),
    "dc_current": (6.4, 6.8),
    "inverter_temp": (11.5, 3.0),
    "irradiance": (11.5, 0.8),
}

SHORT = {
    "t1": "T1",
    "t2": "T2",
    "t3": "T3",
    "s1": "S1",
    "s2": "S2",
    "s3": "S3",
    "s4": "S4",
    "dust_index": "Dust",
    "dc_voltage": "Volt",
    "dc_current": "Amp",
    "inverter_temp": "Inv",
    "irradiance": "Sun",
}

WATCHES = {
    "hour": "Context",
    "irradiance": "Context",
    "ambient_temp": "Context",
    "humidity": "Context",
    "wind_speed": "Context",
    "t1": "Hotspot",
    "t2": "Hotspot",
    "t3": "Hotspot",
    "s1": "Partial Shading",
    "s2": "Partial Shading",
    "s3": "Partial Shading",
    "s4": "Partial Shading",
    "dust_index": "Soiling",
    "dc_voltage": "Inverter Fault",
    "dc_current": "Unexplained power loss",
    "inverter_temp": "Inverter Fault",
    "panel_age": "Degradation",
}

EVIDENCE_GROUP = {
    "hour": "Weather station",
    "irradiance": "Weather station",
    "ambient_temp": "Weather station",
    "humidity": "Weather station",
    "wind_speed": "Weather station",
    "module_temp": "Thermal sensors T1 to T3",
    "shading_pct": "Light sensors S1 to S4",
    "dust_index": "Dust sensor",
    "dc_voltage": "Voltage and current sensors",
    "dc_current": "Voltage and current sensors",
    "dc_power_kw": "Voltage and current sensors",
    "inverter_temp": "Inverter temperature sensor",
    "panel_age": "Panel age record",
}

LEVEL_NAMES = {-1: "Info", 0: "OK", 1: "Warning", 2: "Alert"}

SCENARIO_SITE = {"hour": 12.0, "irradiance": 850.0, "ambient_temp": 30.0, "humidity": 40.0, "wind_speed": 3.0}


def snap(name, value):
    _, lo, hi, step, _ = SENSORS[name]
    return float(min(max(round(value / step) * step, lo), hi))


def expected_module_temp(r):
    return r["ambient_temp"] + r["irradiance"] * 0.026 - r["wind_speed"] * 0.7


def to_features(r):
    return {
        "hour": r["hour"],
        "irradiance": r["irradiance"],
        "ambient_temp": r["ambient_temp"],
        "humidity": r["humidity"],
        "wind_speed": r["wind_speed"],
        "dust_index": r["dust_index"],
        "panel_age": r["panel_age"],
        "module_temp": max(r["t1"], r["t2"], r["t3"]),
        "shading_pct": (r["s1"] + r["s2"] + r["s3"] + r["s4"]) / 4.0,
        "dc_voltage": r["dc_voltage"],
        "dc_current": r["dc_current"],
        "inverter_temp": r["inverter_temp"],
    }


def derive_electrical(r):
    hottest = max(r["t1"], r["t2"], r["t3"])
    shade = (r["s1"] + r["s2"] + r["s3"] + r["s4"]) / 4.0
    hot_rise = max(0.0, hottest - expected_module_temp(r))
    sun = np.sin(np.pi * (r["hour"] - 6) / 12)
    inverter_excess = r["inverter_temp"] - (38 + 14 * sun)
    inverter_loss = 0.0 if inverter_excess < 12 else min(0.6, 0.2 + (inverter_excess - 12) / 16 * 0.4)
    age_loss = min(0.28, 0.005 * r["panel_age"] + max(0.0, r["panel_age"] - 11) * 0.012)
    power = (
        5.0
        * (r["irradiance"] / 1000)
        * (1 - 0.0040 * (hottest - 25))
        * (1 - 0.0036 * r["dust_index"])
        * (1 - 0.0092 * shade)
        * (1 - 0.006 * hot_rise)
        * (1 - age_loss)
        * (1 - inverter_loss)
        * 0.96
    )
    power = max(power, 0.02)
    voltage = 380 * (1 - 0.0034 * (hottest - 25)) * (1 - 0.0035 * shade)
    current = power * 1000 / (voltage * 0.97)
    return float(voltage), float(current)


def sensor_levels(r, expected_kw, measured_kw):
    levels = {name: -1 for name in ["hour", "irradiance", "ambient_temp", "humidity", "wind_speed"]}

    base = expected_module_temp(r)
    for name in ["t1", "t2", "t3"]:
        excess = r[name] - base
        levels[name] = 2 if excess >= 18 else 1 if excess >= 9 else 0

    for name in ["s1", "s2", "s3", "s4"]:
        levels[name] = 2 if r[name] >= 35 else 1 if r[name] >= 15 else 0

    levels["dust_index"] = 2 if r["dust_index"] >= 55 else 1 if r["dust_index"] > 35 else 0

    hottest = max(r["t1"], r["t2"], r["t3"])
    expected_v = 380 * (1 - 0.0034 * (hottest - 25))
    drift = abs(r["dc_voltage"] - expected_v)
    levels["dc_voltage"] = 2 if drift >= 25 else 1 if drift >= 12 else 0

    gap = (expected_kw - measured_kw) / expected_kw if expected_kw > 0 else 0.0
    levels["dc_current"] = 2 if gap >= 0.30 else 1 if gap >= 0.15 else 0

    sun = np.sin(np.pi * (r["hour"] - 6) / 12)
    inverter_excess = r["inverter_temp"] - (38 + 14 * sun)
    levels["inverter_temp"] = 2 if inverter_excess >= 20 else 1 if inverter_excess >= 10 else 0

    levels["panel_age"] = 2 if r["panel_age"] >= 17 else 1 if r["panel_age"] >= 11 else 0
    return levels


def _idw(points):
    xs = np.arange(PANEL_COLS) + 0.5
    ys = np.arange(PANEL_ROWS) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    num = np.zeros_like(grid_x)
    den = np.zeros_like(grid_x)
    for px, py, value in points:
        weight = 1.0 / ((grid_x - px) ** 2 + (grid_y - py) ** 2 + 0.8)
        num += weight * value
        den += weight
    return num / den


def temperature_field(r):
    return _idw([(*POSITIONS[n], r[n]) for n in ["t1", "t2", "t3"]])


def shade_field(r):
    return _idw([(*POSITIONS[n], r[n]) for n in ["s1", "s2", "s3", "s4"]])


def dust_field(r):
    return np.full((PANEL_ROWS, PANEL_COLS), float(r["dust_index"]))


def build_presets():
    data = generate_dataset(600, 21, site=SCENARIO_SITE, fault_probs=[1 / 6] * 6)
    medians = data.groupby("fault")[
        ["dust_index", "panel_age", "module_temp", "shading_pct", "dc_voltage", "dc_current", "inverter_temp"]
    ].median()
    base = expected_module_temp(SCENARIO_SITE)
    presets = {}
    for fault in FAULTS:
        m = medians.loc[fault]
        row = dict(SCENARIO_SITE)
        row["dust_index"] = m["dust_index"]
        row["panel_age"] = m["panel_age"]
        row["dc_voltage"] = m["dc_voltage"]
        row["dc_current"] = m["dc_current"]
        row["inverter_temp"] = m["inverter_temp"]
        if fault == "Hotspot":
            row["t1"] = base + 0.8
            row["t2"] = m["module_temp"]
            row["t3"] = base - 0.6
        else:
            row["t1"] = m["module_temp"] + 0.6
            row["t2"] = m["module_temp"]
            row["t3"] = m["module_temp"] - 0.5
        if fault == "Partial Shading":
            edge = min(100.0, 2 * m["shading_pct"] - 1)
            row["s1"] = edge
            row["s2"] = edge
            row["s3"] = 1.0
            row["s4"] = 1.0
        else:
            for name in ["s1", "s2", "s3", "s4"]:
                row[name] = max(float(m["shading_pct"]), 0.5)
        presets[fault] = {name: snap(name, float(value)) for name, value in row.items()}
    return presets


def group_evidence(contrib):
    grouped = {}
    for _, item in contrib.iterrows():
        group = EVIDENCE_GROUP[item["feature"]]
        grouped[group] = grouped.get(group, 0.0) + float(item["shap"])
    return grouped
