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
    [0.4, "#2E86C1"],
    [0.65, "#F4C430"],
    [0.85, "#FF8A3D"],
    [1.0, "#FF3B3B"],
]
LOG_FRONT = ["Time", "Fault", "Confidence %", "Action", "Note", "Expected kW", "Measured kW", "Problems", "Checks"]


@st.cache_data(show_spinner=False)
def healthy_readings():
    return sn.build_presets()["Normal"]


def load_values(values):
    for name, value in values.items():
        st.session_state["sn_" + name] = value


def load_random():
    values, _ = sn.random_scenario(np.random.default_rng())
    load_values(values)
    st.toast("Random readings loaded. Can you spot the fault?")


def load_healthy():
    load_values(healthy_readings())
    st.toast("All sensors reset to healthy readings")


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
    load_values(sn.fix_values(fault, current))
    st.toast(sn.FIXES[fault] + ". Sensors updated.")


def clear_log():
    st.session_state["reading_log"] = []


def chip_text(name, value):
    return f"{sn.SHORT[name]} {value:.0f}{sn.CHIP_UNITS[name]}"


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
            showscale=False,
            xgap=2,
            ygap=2,
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
            mode="markers",
            marker=dict(size=16, color=[LEVEL_COLORS[levels[n]] for n in names], line=dict(color="#FFFFFF", width=2)),
            customdata=hover,
            hovertemplate="<b>%{customdata[0]}</b><br>Reading: %{customdata[1]}<br>Status: %{customdata[2]}<br>Watches for: %{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )
    for name in names:
        x, y = sn.POSITIONS[name]
        fig.add_annotation(
            x=x,
            y=y,
            text=chip_text(name, r[name]),
            showarrow=False,
            yshift=-24,
            bgcolor="rgba(11,22,38,0.85)",
            bordercolor=LEVEL_COLORS[levels[name]],
            borderwidth=1.5,
            borderpad=3,
            font=dict(color="#FFFFFF", size=11),
        )

    fig.add_shape(type="rect", x0=0, y0=0, x1=sn.PANEL_COLS, y1=sn.PANEL_ROWS, line=dict(color=ui.MUTED, width=3))
    fig.add_shape(type="rect", x0=2.6, y0=6.4, x1=7.4, y1=7.2, line=dict(color=ui.EDGE, width=2), fillcolor=ui.BG)
    fig.add_shape(type="rect", x0=10.8, y0=2.1, x1=12.2, y1=3.9, line=dict(color=ui.EDGE, width=2), fillcolor=ui.BG)
    for text, x, y in [("Junction box", 5.0, 8.15), ("Inverter box", 11.5, 4.9), ("Weather mast", 11.5, 0.05)]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(color=ui.MUTED, size=12))

    fig.update_layout(title="Live panel sensors")
    fig.update_xaxes(visible=False, range=[-1.4, 13.4])
    fig.update_yaxes(visible=False, range=[8.5, -0.5])
    return ui.style_figure(fig, 520)


LEGEND = (
    f'<div style="display:flex;flex-wrap:wrap;gap:18px;margin:6px 0 4px 0;color:{ui.TEXT}">'
    f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{ui.GREEN};margin-right:6px"></span>Normal</span>'
    f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{ui.SUN};margin-right:6px"></span>Check</span>'
    f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{ui.RED};margin-right:6px"></span>Problem</span>'
    f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{ui.BLUE};margin-right:6px"></span>Weather info</span>'
    f"</div>"
    f'<div style="color:{ui.MUTED};font-size:0.95rem">Each dot is a sensor. The panel turns orange and red where it is hot, and dark where it is shaded.</div>'
)


def contribution_card(grouped, label):
    shares = [(name, pct) for name, pct in sn.evidence_share(grouped) if pct >= 1]
    rows = ""
    for index, (name, pct) in enumerate(shares[:5]):
        color = ui.SUN if index == 0 else ui.BLUE
        rows += (
            f'<div style="margin-top:12px"><div style="display:flex;justify-content:space-between">'
            f"<span>{name}</span><b>{pct:.0f}%</b></div>"
            f'<div style="background:{ui.EDGE};border-radius:6px;height:10px;margin-top:4px">'
            f'<div style="width:{pct:.0f}%;background:{color};height:10px;border-radius:6px"></div></div></div>'
        )
    return (
        f'<div class="panel-note"><b>What contributed to this verdict</b>'
        f'<div style="color:{ui.MUTED};font-size:0.95rem">Share of the evidence pointing to {label}</div>{rows}</div>'
    )


def page_panel(bundle):
    st.markdown(
        ui.masthead(
            "Panel sensors",
            "One panel with its sensors. Load random readings and SolarSense finds the fault, shows which sensors caused it and how to fix it.",
        ),
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("reading_log", [])
    st.session_state.setdefault("log_note", "")

    if "sn_t1" not in st.session_state:
        load_values(healthy_readings())

    first, second, _ = st.columns([1, 1, 3])
    first.button("Load random readings", on_click=load_random, key="btn_random")
    second.button("Reset to healthy", on_click=load_healthy, key="btn_healthy")

    left, right = st.columns([2, 3])
    reading = {}
    with left:
        auto = st.toggle(
            "Calculate voltage and current from the other sensors",
            value=True,
            key="auto_electrical",
            help="When on, the junction box readings follow the heat, shade, dust, inverter and age sensors, so changing a sensor changes the power. Turn it off to set voltage and current yourself.",
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
        st.markdown(LEGEND, unsafe_allow_html=True)

    contrib = local_contributions(bundle["clf"], result["frame"], result["label_id"])
    grouped = sn.group_evidence(contrib)
    shares = sn.evidence_share(grouped)

    problems = sum(1 for v in levels.values() if v == 2)
    checks = sum(1 for v in levels.values() if v == 1)

    snapshot = {
        "Time": now_text(),
        "Fault": label,
        "Confidence %": round(result["confidence"] * 100, 1),
        "Expected kW": round(result["expected_kw"], 2),
        "Measured kW": round(result["measured_kw"], 2),
        "Problems": problems,
        "Checks": checks,
    }
    for name in sn.SENSORS:
        snapshot[sensor_column(name)] = round(reading[name], 1)
    st.session_state["latest"] = snapshot

    raised = [sn.SHORT[n] for n, v in levels.items() if v == 2 and n in sn.SHORT]
    if label == "Normal":
        detail = "All sensors read within their normal range."
    else:
        detail = f"<b>Biggest clue:</b> {shares[0][0]}"
        if raised:
            detail += f"<br><b>Sensors showing a problem:</b> {', '.join(raised)}"

    st.write("")
    verdict_col, contribution_col = st.columns(2)
    with verdict_col:
        st.markdown(
            f'<div class="verdict" style="border-top-color:{ui.FAULT_COLORS[label]}">'
            f'<h2>{label}</h2><p>{ACTIONS[label]}</p><p style="margin-top:8px">{detail}</p>'
            f'<p style="margin-top:8px">Confidence {result["confidence"] * 100:.0f}%</p></div>',
            unsafe_allow_html=True,
        )
    with contribution_col:
        st.markdown(contribution_card(grouped, label), unsafe_allow_html=True)

    st.write("")
    steps_col, record_col = st.columns([3, 2])
    with steps_col:
        steps = "".join(f"<li>{step}</li>" for step in RESOLUTION[label])
        st.markdown(
            f'<div class="panel-note"><b>How to resolve it</b>'
            f'<ol style="margin:6px 0 0 0;padding-left:22px;line-height:1.7">{steps}</ol></div>',
            unsafe_allow_html=True,
        )
        if label != "Normal":
            st.write("")
            st.button(f"Mark as resolved: {sn.FIXES[label]}", on_click=resolve_fault, args=(label,), key="btn_resolve")
    with record_col:
        st.text_input("Note for the log (optional)", key="log_note", placeholder="For example: wash booked for Friday")
        st.button("Record readings", on_click=record_snapshot, key="btn_record")

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
