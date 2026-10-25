from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from zoneinfo import ZoneInfo

    LOCAL_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    LOCAL_TZ = None

from src import sensors as sn
from src import ui
from src.data_generator import FAULTS
from src.explain import local_contributions
from src.predict import ACTIONS, RESOLUTION, diagnose
from src.train import ensure_bundle

st.set_page_config(page_title="SolarSense AI", page_icon="☀️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the models for the first time. This takes around a minute.")
def get_bundle():
    return ensure_bundle()


LEVEL_COLORS = {-1: ui.BLUE, 0: ui.GREEN, 1: ui.SUN, 2: ui.RED}
THERMAL_SCALE = [
    [0.0, "#1B4F8C"],
    [0.35, "#2E86C1"],
    [0.6, "#F4C430"],
    [0.8, "#FF8A3D"],
    [1.0, "#FF3B3B"],
]
MARKER_SYMBOLS = {
    "t1": "diamond",
    "t2": "diamond",
    "t3": "diamond",
    "s1": "square",
    "s2": "square",
    "s3": "square",
    "s4": "square",
    "dust_index": "circle",
    "dc_voltage": "hexagon",
    "dc_current": "hexagon",
    "inverter_temp": "star",
    "irradiance": "triangle-up",
}


@st.cache_data(show_spinner=False)
def scenario_presets():
    return sn.build_presets()


def apply_scenario():
    chosen = st.session_state["scenario"]
    for name, value in scenario_presets()[chosen].items():
        st.session_state["sn_" + name] = value
    st.toast("Loaded the " + chosen.lower() + " scenario")


def panel_figure(r, levels):
    xs = np.arange(sn.PANEL_COLS) + 0.5
    ys = np.arange(sn.PANEL_ROWS) + 0.5
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=sn.temperature_field(r),
            colorscale=THERMAL_SCALE,
            zmin=20,
            zmax=100,
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text="Cell °C", font=dict(color=ui.MUTED)), thickness=12, len=0.6, y=0.65, tickfont=dict(color=ui.MUTED)),
            hovertemplate="Cell temperature %{z:.0f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=sn.dust_field(r),
            colorscale=[[0, "rgba(214,170,90,0)"], [1, "rgba(214,170,90,0.6)"]],
            zmin=0,
            zmax=100,
            showscale=False,
            xgap=2,
            ygap=2,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=sn.shade_field(r),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0.78)"]],
            zmin=0,
            zmax=100,
            showscale=False,
            xgap=2,
            ygap=2,
            hoverinfo="skip",
        )
    )

    names = list(sn.POSITIONS)
    hover = []
    for name in names:
        label, _, _, _, unit = sn.SENSORS[name]
        hover.append([label, f"{r[name]:.1f} {unit}".strip(), sn.LEVEL_NAMES[levels[name]], sn.WATCHES[name]])
    fig.add_trace(
        go.Scatter(
            x=[sn.POSITIONS[n][0] for n in names],
            y=[sn.POSITIONS[n][1] for n in names],
            mode="markers+text",
            text=[f"{sn.SHORT[n]} {r[n]:.0f}" for n in names],
            textposition="bottom center",
            textfont=dict(color="#FFFFFF", size=12),
            marker=dict(
                size=22,
                color=[LEVEL_COLORS[levels[n]] for n in names],
                symbol=[MARKER_SYMBOLS[n] for n in names],
                line=dict(color="#FFFFFF", width=2),
            ),
            customdata=hover,
            hovertemplate="<b>%{customdata[0]}</b><br>Reading: %{customdata[1]}<br>Status: %{customdata[2]}<br>Watches for: %{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_shape(type="rect", x0=0, y0=0, x1=sn.PANEL_COLS, y1=sn.PANEL_ROWS, line=dict(color=ui.MUTED, width=3))
    fig.add_shape(type="rect", x0=2.8, y0=6.4, x1=7.2, y1=7.2, line=dict(color=ui.EDGE, width=2), fillcolor=ui.BG)
    fig.add_shape(type="rect", x0=10.8, y0=2.1, x1=12.2, y1=3.9, line=dict(color=ui.EDGE, width=2), fillcolor=ui.BG)
    for text, x, y in [("Junction box", 5.0, 8.0), ("Inverter", 11.5, 4.45), ("Weather mast", 11.5, 0.05)]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=ui.MUTED, size=12))

    fig.update_layout(title="Panel with live sensors")
    fig.update_xaxes(visible=False, range=[-0.6, 13.0])
    fig.update_yaxes(visible=False, range=[8.4, -0.5])
    return ui.style_figure(fig, 500)


LOG_FRONT = ["Time", "Fault", "Confidence %", "Action", "Note", "Expected kW", "Measured kW", "Alerts", "Warnings"]


def sensor_column(name):
    label, _, _, _, unit = sn.SENSORS[name]
    return f"{label} ({unit})" if unit else label


def now_text():
    return datetime.now(LOCAL_TZ).strftime("%d %b %Y, %I:%M:%S %p")


def record_snapshot():
    entry = dict(st.session_state["latest"])
    entry["Action"] = "Recorded"
    entry["Note"] = st.session_state.get("log_note", "")
    st.session_state["reading_log"].append(entry)
    st.session_state["log_note"] = ""
    st.toast("Readings recorded")


def resolve_fault(fault):
    entry = dict(st.session_state["latest"])
    entry["Action"] = "Resolved: " + sn.FIXES[fault]
    entry["Note"] = st.session_state.get("log_note", "")
    st.session_state["reading_log"].append(entry)
    st.session_state["log_note"] = ""
    current = {name: st.session_state["sn_" + name] for name in sn.SENSORS if "sn_" + name in st.session_state}
    for name, value in sn.fix_values(fault, current).items():
        st.session_state["sn_" + name] = value
    st.toast(sn.FIXES[fault] + ". Sensors updated.")


def clear_log():
    st.session_state["reading_log"] = []


def page_panel(bundle):
    st.markdown(
        ui.masthead(
            "Panel sensors",
            "One panel wired with sensors. Change any reading and watch which sensors raise the alarm and which fault the model detects.",
        ),
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("reading_log", [])
    st.session_state.setdefault("log_note", "")

    if "sn_t1" not in st.session_state:
        for name, value in scenario_presets()["Normal"].items():
            st.session_state["sn_" + name] = value

    st.selectbox(
        "Start from a scenario",
        FAULTS,
        key="scenario",
        on_change=apply_scenario,
        help="Loads typical sensor readings for that condition. Every sensor can still be changed.",
    )

    left, right = st.columns([2, 3])
    reading = {}
    with left:
        auto = st.toggle(
            "Calculate voltage and current from the other sensors",
            value=True,
            key="auto_electrical",
            help="When on, the junction box readings follow the thermal, light, dust, inverter and age sensors, so changing a sensor changes the power. Turn it off to set voltage and current yourself.",
        )
        tabs = st.tabs(list(sn.TABS))
        for tab, names in zip(tabs, sn.TABS.values()):
            with tab:
                for name in names:
                    if auto and name in ("dc_voltage", "dc_current"):
                        continue
                    label, lo, hi, step, unit = sn.SENSORS[name]
                    title = f"{label} ({unit})" if unit else label
                    reading[name] = st.slider(title, lo, hi, step=step, key="sn_" + name)
        if auto:
            reading["dc_voltage"], reading["dc_current"] = sn.derive_electrical(reading)
            with tabs[3]:
                st.markdown(
                    f'<div class="panel-note">Junction box (calculated): <b>{reading["dc_voltage"]:.0f} V</b> and <b>{reading["dc_current"]:.1f} A</b></div>',
                    unsafe_allow_html=True,
                )

    result = diagnose(bundle, sn.to_features(reading))
    levels = sn.sensor_levels(reading, result["expected_kw"], result["measured_kw"])
    label = result["label"]

    with right:
        st.plotly_chart(panel_figure(reading, levels))

    contrib = local_contributions(bundle["clf"], result["frame"], result["label_id"])
    grouped = sn.group_evidence(contrib)
    top_group = max(grouped, key=grouped.get)

    alerts = sum(1 for n, v in levels.items() if v == 2)
    warnings = sum(1 for n, v in levels.items() if v == 1)

    snapshot = {
        "Time": now_text(),
        "Fault": label,
        "Confidence %": round(result["confidence"] * 100, 1),
        "Expected kW": round(result["expected_kw"], 2),
        "Measured kW": round(result["measured_kw"], 2),
        "Alerts": alerts,
        "Warnings": warnings,
    }
    for name in sn.SENSORS:
        snapshot[sensor_column(name)] = round(reading[name], 1)
    st.session_state["latest"] = snapshot

    raised = [sn.SHORT[n] for n, v in levels.items() if v == 2 and n in sn.SHORT]
    if label == "Normal":
        detail = "All sensors read within their normal range."
    else:
        detail = f"<b>Detected mainly by:</b> {top_group}"
        if raised:
            detail += f"<br><b>Sensors in alert:</b> {', '.join(raised)}"

    verdict_col, resolve_col = st.columns(2)
    with verdict_col:
        st.markdown(
            f'<div class="verdict" style="border-top-color:{ui.FAULT_COLORS[label]}">'
            f'<h2>{label}</h2><p>{ACTIONS[label]}</p><p style="margin-top:8px">{detail}</p>'
            f'<p style="margin-top:8px">Confidence {result["confidence"] * 100:.0f}%</p></div>',
            unsafe_allow_html=True,
        )
    with resolve_col:
        steps = "".join(f"<li>{step}</li>" for step in RESOLUTION[label])
        st.markdown(
            f'<div class="panel-note"><b>How to resolve it</b>'
            f'<ol style="margin:6px 0 0 0;padding-left:22px;line-height:1.7">{steps}</ol></div>',
            unsafe_allow_html=True,
        )
        if label != "Normal":
            st.write("")
            st.button(f"Mark as resolved: {sn.FIXES[label]}", on_click=resolve_fault, args=(label,), key="btn_resolve")

    st.write("")
    note_col, record_col = st.columns([4, 1])
    note_col.text_input("Note for the log (optional)", key="log_note", placeholder="For example: checked on site, wash booked for Friday")
    record_col.write("")
    record_col.write("")
    record_col.button("Record readings", on_click=record_snapshot, key="btn_record")

    log = st.session_state["reading_log"]
    with st.expander(f"Recorded readings ({len(log)})"):
        if not log:
            st.write("Nothing recorded yet.")
        else:
            frame = pd.DataFrame(log[::-1])
            ordered = LOG_FRONT + [c for c in frame.columns if c not in LOG_FRONT]
            st.dataframe(frame[ordered], hide_index=True)
            download_col, clear_col = st.columns(2)
            download_col.download_button(
                "Download log as CSV",
                pd.DataFrame(log).to_csv(index=False),
                file_name="solarsense_reading_log.csv",
                mime="text/csv",
            )
            clear_col.button("Clear log", on_click=clear_log)


def main():
    bundle = get_bundle()
    page_panel(bundle)
    st.caption(f'Fault detector: {bundle["clf_name"]}. Power model: {bundle["reg_name"]}. All sensor data is simulated.')


main()
