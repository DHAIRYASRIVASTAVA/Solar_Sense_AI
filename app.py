import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import sensors as sn
from src import ui
from src.data_generator import FAULTS
from src.explain import local_contributions, make_explainer
from src.predict import ACTIONS, RESOLUTION, diagnose
from src.train import ensure_bundle

st.set_page_config(page_title="SolarSense AI", page_icon="☀️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the models for the first time. This takes around a minute.")
def get_bundle():
    return ensure_bundle()


@st.cache_resource(show_spinner=False)
def get_explainer(_model, token):
    return make_explainer(_model)


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


def level_color(value):
    colors = {"OK": ui.GREEN, "Warning": ui.SUN, "Alert": ui.RED, "Info": ui.BLUE}
    return f"background-color: {colors[value]}; color: #06121F; font-weight: 700"


def page_panel(bundle, token):
    st.markdown(
        ui.masthead(
            "Panel sensors",
            "One panel wired with sensors. Change any reading and watch which sensors raise the alarm and which fault the model detects.",
        ),
        unsafe_allow_html=True,
    )

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

    explainer = get_explainer(bundle["clf"], token)
    contrib = local_contributions(explainer, result["frame"], result["label_id"])
    grouped = sn.group_evidence(contrib)
    top_group = max(grouped, key=grouped.get)

    alerts = sum(1 for n, v in levels.items() if v == 2)
    warnings = sum(1 for n, v in levels.items() if v == 1)

    if label == "Normal":
        detail = "All sensors read within their normal range."
    else:
        detail = f"<b>Detected mainly by:</b> {top_group}"

    verdict_col, evidence_col = st.columns([2, 3])
    with verdict_col:
        st.markdown(
            f'<div class="verdict" style="border-top-color:{ui.FAULT_COLORS[label]}">'
            f"<h2>{label}</h2><p>{ACTIONS[label]}</p><p style=\"margin-top:8px\">{detail}</p></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        a, b = st.columns(2)
        a.markdown(ui.stat("Confidence", f'{result["confidence"] * 100:.0f}%'), unsafe_allow_html=True)
        b.markdown(ui.stat("Sensors alarming", f"{alerts} alert, {warnings} warning"), unsafe_allow_html=True)
    with evidence_col:
        ordered = sorted(grouped.items(), key=lambda item: item[1])
        fig = go.Figure(
            go.Bar(
                x=[v for _, v in ordered],
                y=[k for k, _ in ordered],
                orientation="h",
                marker_color=[ui.SUN if k == top_group and label != "Normal" else ui.BLUE for k, _ in ordered],
                hovertemplate="%{y}<br>SHAP %{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(title=f"Which sensors pushed the model toward {label}", xaxis_title="Combined SHAP value")
        st.plotly_chart(ui.style_figure(fig, 330))

    st.markdown("#### How to resolve it")
    steps = "".join(f"<li>{step}</li>" for step in RESOLUTION[label])
    st.markdown(
        f'<div class="panel-note"><ol style="margin:0;padding-left:22px;line-height:1.7">{steps}</ol></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown("#### Every sensor, its status and the fault it watches")
    rows = []
    for name, (sensor_label, _, _, _, unit) in sn.SENSORS.items():
        rows.append(
            {
                "Sensor": sensor_label,
                "Reading": f"{reading[name]:.1f} {unit}".strip(),
                "Status": sn.LEVEL_NAMES[levels[name]],
                "Watches for": sn.WATCHES[name],
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table.style.map(level_color, subset=["Status"]), hide_index=True)
    st.caption("Status compares each reading with what the weather and time of day would normally produce. The fault verdict comes from the trained model.")


def main():
    bundle = get_bundle()
    token = f'{bundle["clf_name"]}-{bundle["n_samples"]}'
    page_panel(bundle, token)
    st.caption(f'Fault detector: {bundle["clf_name"]}. Power model: {bundle["reg_name"]}. All sensor data is simulated.')


main()
