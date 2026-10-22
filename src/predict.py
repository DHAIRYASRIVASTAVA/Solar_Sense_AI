import pandas as pd

from src.data_generator import add_dc_power

ACTIONS = {
    "Normal": "Array is running as expected. Keep the regular inspection schedule.",
    "Soiling": "Dust is cutting output. Schedule a panel wash, ideally early morning.",
    "Partial Shading": "Part of the array is shaded. Check for new obstructions such as trees, poles or debris.",
    "Hotspot": "A cell or string is overheating. Run a thermal inspection and look for cracked cells or bypass diode failure.",
    "Inverter Fault": "Inverter readings look abnormal. Check cooling, error logs and DC input connections.",
    "Degradation": "Panels are ageing faster than expected. Run an IV curve test and plan a replacement review.",
}

RESOLUTION = {
    "Normal": [
        "No action needed, the panel is healthy.",
        "Keep the regular inspection and cleaning schedule.",
        "Come back to this page if any sensor starts showing a warning.",
    ],
    "Soiling": [
        "Schedule a panel wash, ideally early morning or evening when the glass is cool.",
        "Use clean soft water and a soft brush, never abrasive cleaners.",
        "Check the dust sensor afterwards, it should drop below 35.",
        "If dust returns quickly, shorten the cleaning cycle.",
    ],
    "Partial Shading": [
        "Look at which light sensors (S1 to S4) read high, the shade is on that side.",
        "Remove obstructions such as branches, debris, bird droppings or new structures.",
        "If the shade is permanent, plan a layout change or add bypass diodes or optimisers.",
        "Confirm the light sensors drop back below 15 percent.",
    ],
    "Hotspot": [
        "Find the thermal sensor reading highest, the hotspot is around it.",
        "Inspect that area with a thermal camera for cracked cells or a failed bypass diode.",
        "Check for debris or a loose connection near the hot cell.",
        "If a cell is damaged, isolate the string and replace the panel.",
    ],
    "Inverter Fault": [
        "Read the inverter display or error log for fault codes.",
        "Make sure vents and fans are clear and the inverter is out of direct sun.",
        "Inspect and tighten the DC input connections, look for loose or burnt cables.",
        "Restart the inverter. If the fault returns, call a service technician.",
    ],
    "Degradation": [
        "Run an IV curve test and compare real output with the panel datasheet.",
        "Check the surface for discolouration, delamination or snail trails.",
        "Compare with neighbouring panels of the same age.",
        "If the loss is beyond the warranty limit, raise a warranty claim or plan a replacement.",
    ],
}


def build_frame(values):
    return add_dc_power(pd.DataFrame([values]))


def diagnose(bundle, values):
    frame = build_frame(values)
    clf_x = frame[bundle["clf_features"]]
    reg_x = frame[bundle["reg_features"]]
    proba = bundle["clf"].predict_proba(clf_x)[0]
    top = int(proba.argmax())
    expected = float(bundle["reg"].predict(reg_x)[0])
    measured = float(frame["dc_power_kw"].iloc[0])
    return {
        "frame": clf_x,
        "proba": proba,
        "label": bundle["classes"][top],
        "label_id": top,
        "confidence": float(proba[top]),
        "expected_kw": expected,
        "measured_kw": measured,
    }


def batch_predict(bundle, df):
    out = df.copy()
    ids = bundle["clf"].predict(out[bundle["clf_features"]])
    out["predicted_fault"] = [bundle["classes"][i] for i in ids]
    out["predicted_kw"] = bundle["reg"].predict(out[bundle["reg_features"]])
    return out
