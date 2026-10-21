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
