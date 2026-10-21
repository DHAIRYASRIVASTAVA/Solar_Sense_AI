import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import sensors as sn
from src import ui
from src.data_generator import CLF_FEATURES, FAULTS, add_dc_power, generate_dataset
from src.explain import global_importance, local_contributions, make_explainer
from src.predict import ACTIONS, batch_predict, diagnose
from src.train import ensure_bundle, train_all

st.set_page_config(page_title="SolarSense AI", page_icon="☀️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

FIELDS = {
    "hour": ("Time of day (hour)", 6.0, 18.0, 0.25),
    "irradiance": ("Irradiance (W/m²)", 60.0, 1100.0, 10.0),
    "ambient_temp": ("Ambient temperature (°C)", 0.0, 50.0, 0.5),
    "humidity": ("Humidity (%)", 10.0, 100.0, 1.0),
    "wind_speed": ("Wind speed (m/s)", 0.0, 20.0, 0.5),
    "dust_index": ("Dust on panels (0 to 100)", 0.0, 100.0, 1.0),
    "panel_age": ("Panel age (years)", 0.0, 25.0, 0.5),
    "module_temp": ("Module temperature (°C)", 10.0, 95.0, 0.5),
    "shading_pct": ("Shaded area (%)", 0.0, 80.0, 1.0),
    "dc_voltage": ("DC voltage (V)", 200.0, 450.0, 1.0),
    "dc_current": ("DC current (A)", 0.0, 25.0, 0.1),
    "inverter_temp": ("Inverter temperature (°C)", 25.0, 90.0, 0.5),
}

GROUPS = [
    ("Weather", ["hour", "irradiance", "ambient_temp", "humidity", "wind_speed"]),
    ("Panel condition", ["dust_index", "panel_age", "module_temp", "shading_pct"]),
    ("Electrical readings", ["dc_voltage", "dc_current", "inverter_temp"]),
]


GRID_ROWS = 5
GRID_COLS = 8
SITE_FIELDS = ["hour", "irradiance", "ambient_temp", "humidity", "wind_speed"]
SITE_DEFAULTS = {"hour": 12.0, "irradiance": 850.0, "ambient_temp": 30.0, "humidity": 40.0, "wind_speed": 3.0}
EDIT_FIELDS = ["dust_index", "shading_pct", "module_temp", "panel_age", "dc_voltage", "dc_current", "inverter_temp"]
SHORT_NAMES = {
    "Normal": "OK",
    "Soiling": "Dusty",
    "Partial Shading": "Shaded",
    "Hotspot": "Hot",
    "Inverter Fault": "Inverter",
    "Degradation": "Aged",
}
DENSITY = {
    "Few faults": [0.80, 0.04, 0.05, 0.04, 0.04, 0.03],
    "Some faults": [0.55, 0.09, 0.09, 0.09, 0.09, 0.09],
    "Many faults": [0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
}


@st.cache_resource(show_spinner="Training the models for the first time. This takes around a minute.")
def get_bundle():
    return ensure_bundle()


@st.cache_resource(show_spinner=False)
def get_explainer(_model, token):
    return make_explainer(_model)


@st.cache_data(show_spinner=False)
def preset_table():
    data = generate_dataset(4000, 11)
    medians = data.groupby("fault")[list(FIELDS)].median()
    presets = {}
    for fault in FAULTS:
        row = {}
        for name, (_, lo, hi, step) in FIELDS.items():
            value = round(float(medians.loc[fault, name]) / step) * step
            row[name] = float(min(max(value, lo), hi))
        presets[fault] = row
    return presets


@st.cache_data(show_spinner=False)
def fleet_data(_bundle, token):
    data = generate_dataset(2500, 2024)
    scored = batch_predict(_bundle, data)
    scored.insert(0, "panel_id", ["PV-" + str(i + 1).zfill(4) for i in range(len(scored))])
    scored["status"] = scored["predicted_fault"].map(lambda f: "Healthy" if f == "Normal" else "Fault flagged")
    return scored


def apply_preset():
    chosen = st.session_state["preset"]
    for name, value in preset_table()[chosen].items():
        st.session_state["in_" + name] = value
    st.toast("Loaded a typical " + chosen.lower() + " reading")


def prepare_inputs():
    if "in_hour" not in st.session_state:
        for name, value in preset_table()["Normal"].items():
            st.session_state["in_" + name] = value


def read_inputs():
    return {name: float(st.session_state["in_" + name]) for name in FIELDS}


@st.cache_data(show_spinner=False)
def simulate_array(site_values, seed, density):
    site = dict(zip(SITE_FIELDS, site_values))
    count = GRID_ROWS * GRID_COLS
    data = generate_dataset(count, seed, site=site, fault_probs=DENSITY[density])
    data = data.drop(columns=["power_kw", "fault_id", "fault"])
    data.insert(0, "panel_id", [chr(65 + i // GRID_COLS) + str(i % GRID_COLS + 1) for i in range(count)])
    return data


def snap(name, value):
    _, lo, hi, step = FIELDS[name]
    return float(min(max(round(value / step) * step, lo), hi))


def bump_seed():
    st.session_state["array_seed"] = st.session_state.get("array_seed", 7) + 1
    st.session_state["overrides"] = {}


def clear_edits():
    st.session_state["overrides"] = {}
    current = st.session_state.get("edit_pid")
    if current:
        st.session_state["use_" + current] = False


def page_array(bundle, token):
    st.markdown(
        ui.masthead(
            "Array monitor",
            "Change the sensor readings. SolarSense scans every panel and shows which ones are faulty and what is wrong with each.",
        ),
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("array_seed", 7)
    st.session_state.setdefault("overrides", {})

    columns = st.columns(5)
    site = []
    for column, name in zip(columns, SITE_FIELDS):
        label, lo, hi, step = FIELDS[name]
        with column:
            site.append(st.slider(label, lo, hi, SITE_DEFAULTS[name], step=step, key="site_" + name))

    control_a, control_b, control_c = st.columns([2, 1, 3])
    density = control_a.selectbox("How faulty is this array?", list(DENSITY), index=1)
    control_b.write("")
    control_b.button("New sensor feed", on_click=bump_seed)

    data = simulate_array(tuple(site), st.session_state["array_seed"], density).copy()
    ids = data["panel_id"].tolist()

    with st.expander("Edit one panel's sensors by hand"):
        pick = st.selectbox("Panel", ids, key="edit_pid")
        row = data[data["panel_id"] == pick].iloc[0]
        saved = st.session_state["overrides"].get(pick, {})
        edit_columns = st.columns(4)
        edits = {}
        for i, name in enumerate(EDIT_FIELDS):
            label, lo, hi, step = FIELDS[name]
            start = snap(name, saved.get(name, float(row[name])))
            with edit_columns[i % 4]:
                edits[name] = st.slider(label, lo, hi, start, step=step, key=f"edit_{pick}_{name}")
        use = st.checkbox("Use these readings for this panel", value=pick in st.session_state["overrides"], key="use_" + pick)
        if use:
            st.session_state["overrides"][pick] = edits
        else:
            st.session_state["overrides"].pop(pick, None)
        st.button("Clear all manual edits", on_click=clear_edits)

    for pid, values in st.session_state["overrides"].items():
        for name, value in values.items():
            data.loc[data["panel_id"] == pid, name] = value

    data = add_dc_power(data)
    proba = bundle["clf"].predict_proba(data[bundle["clf_features"]])
    label_ids = proba.argmax(axis=1)
    data["detected"] = [FAULTS[i] for i in label_ids]
    data["confidence"] = proba.max(axis=1)
    data["expected_kw"] = bundle["reg"].predict(data[bundle["reg_features"]])

    healthy = int((data["detected"] == "Normal").sum())
    faulty = len(data) - healthy
    a, b, c, d = st.columns(4)
    a.markdown(ui.stat("Healthy panels", f"{healthy} of {len(data)}"), unsafe_allow_html=True)
    b.markdown(ui.stat("Faults detected", str(faulty)), unsafe_allow_html=True)
    c.markdown(ui.stat("Array output", f'{data["dc_power_kw"].sum():.1f} kW'), unsafe_allow_html=True)
    top = data[data["detected"] != "Normal"]["detected"].value_counts()
    d.markdown(ui.stat("Most common fault", top.idxmax() if len(top) else "None"), unsafe_allow_html=True)

    z, text, hover = [], [], []
    for r in range(GRID_ROWS):
        z_row, t_row, h_row = [], [], []
        for col in range(GRID_COLS):
            item = data.iloc[r * GRID_COLS + col]
            fault_index = FAULTS.index(item["detected"])
            z_row.append(fault_index + 0.5)
            t_row.append(f'{item["panel_id"]}<br>{SHORT_NAMES[item["detected"]]}')
            h_row.append(
                [
                    item["panel_id"],
                    item["detected"],
                    f'{item["confidence"] * 100:.0f}%',
                    f'{item["dc_power_kw"]:.2f} kW',
                    f'{item["module_temp"]:.1f} °C',
                ]
            )
        z.append(z_row)
        text.append(t_row)
        hover.append(h_row)

    scale = []
    for i, fault in enumerate(FAULTS):
        scale.append([i / len(FAULTS), ui.FAULT_COLORS[fault]])
        scale.append([(i + 1) / len(FAULTS), ui.FAULT_COLORS[fault]])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            text=text,
            customdata=hover,
            colorscale=scale,
            zmin=0,
            zmax=len(FAULTS),
            showscale=False,
            xgap=6,
            ygap=6,
            texttemplate="%{text}",
            textfont=dict(color="#06121F", size=13),
            hovertemplate="<b>%{customdata[0]}</b><br>Detected: %{customdata[1]}<br>Confidence: %{customdata[2]}<br>Output: %{customdata[3]}<br>Module temp: %{customdata[4]}<extra></extra>",
        )
    )
    fig.update_layout(title="Live array map")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    st.plotly_chart(ui.style_figure(fig, 400))

    counts = data["detected"].value_counts()
    chips = "".join(
        f'<span style="display:inline-block;margin:0 14px 6px 0;color:{ui.TEXT}">'
        f'<span style="display:inline-block;width:14px;height:14px;border-radius:4px;background:{ui.FAULT_COLORS[f]};margin-right:6px;vertical-align:-2px"></span>'
        f"{f}: <b>{int(counts.get(f, 0))}</b></span>"
        for f in FAULTS
    )
    st.markdown(chips, unsafe_allow_html=True)

    flagged = data[data["detected"] != "Normal"].sort_values("confidence", ascending=False)
    if flagged.empty:
        st.success("No faults detected. Every panel is reading as healthy.")
        return

    st.markdown("#### Faulty panels and what to do")
    table = flagged.assign(action=flagged["detected"].map(ACTIONS), confidence=(flagged["confidence"] * 100).round(0))[
        ["panel_id", "detected", "confidence", "dc_power_kw", "action"]
    ].rename(
        columns={
            "panel_id": "Panel",
            "detected": "Fault",
            "confidence": "Confidence %",
            "dc_power_kw": "Output kW",
            "action": "Recommended action",
        }
    )
    st.dataframe(table.round(2), hide_index=True)

    st.markdown("#### Why was this panel flagged?")
    target = st.selectbox("Faulty panel", flagged["panel_id"].tolist(), key="why_pid")
    chosen = data[data["panel_id"] == target].iloc[[0]]
    class_id = FAULTS.index(chosen["detected"].iloc[0])
    explainer = get_explainer(bundle["clf"], token)
    contrib = local_contributions(explainer, chosen[bundle["clf_features"]], class_id).tail(7)
    fig = go.Figure(
        go.Bar(
            x=contrib["shap"],
            y=contrib["feature"],
            orientation="h",
            marker_color=[ui.RED if v > 0 else ui.BLUE for v in contrib["shap"]],
            customdata=contrib["value"],
            hovertemplate="%{y} = %{customdata:.2f}<br>SHAP %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f'Readings behind the {chosen["detected"].iloc[0]} verdict on {target}',
        xaxis_title="SHAP value (red raises the odds, blue lowers them)",
    )
    st.plotly_chart(ui.style_figure(fig, 340))


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


def page_diagnosis(bundle, token):
    st.markdown(
        ui.masthead(
            "Panel diagnosis",
            "Enter what your sensors are reporting and SolarSense tells you what is most likely wrong, how sure it is, and why.",
        ),
        unsafe_allow_html=True,
    )

    prepare_inputs()
    st.selectbox(
        "Start from a sample reading",
        FAULTS,
        key="preset",
        on_change=apply_preset,
        help="Loads typical sensor values for that condition. You can still adjust every slider.",
    )

    columns = st.columns(3)
    for column, (title, names) in zip(columns, GROUPS):
        with column:
            st.markdown(f"**{title}**")
            for name in names:
                label, lo, hi, step = FIELDS[name]
                st.slider(label, lo, hi, step=step, key="in_" + name)

    result = diagnose(bundle, read_inputs())
    label = result["label"]
    color = ui.FAULT_COLORS[label]

    tab_a, tab_b = st.tabs(["Diagnosis", "What-if explorer"])

    with tab_a:
        st.markdown("---")
        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                f'<div class="verdict" style="border-top-color:{color}">'
                f"<h2>{label}</h2><p>{ACTIONS[label]}</p></div>",
                unsafe_allow_html=True,
            )
            gap = result["expected_kw"] - result["measured_kw"]
            gap_pct = gap / result["expected_kw"] * 100 if result["expected_kw"] > 0 else 0.0
            st.write("")
            a, b, c = st.columns(3)
            a.markdown(
                ui.stat("Expected from conditions", f'{result["expected_kw"]:.2f} kW', "regression model"),
                unsafe_allow_html=True,
            )
            b.markdown(
                ui.stat("Measured DC power", f'{result["measured_kw"]:.2f} kW', "voltage × current"),
                unsafe_allow_html=True,
            )
            c.markdown(
                ui.stat("Gap", f"{gap_pct:+.1f}%", "expected minus measured"),
                unsafe_allow_html=True,
            )
        with right:
            st.plotly_chart(ui.gauge(result["confidence"], "Model confidence"))

        left, right = st.columns(2)
        with left:
            order = sorted(range(len(FAULTS)), key=lambda i: result["proba"][i])
            fig = go.Figure(
                go.Bar(
                    x=[result["proba"][i] * 100 for i in order],
                    y=[FAULTS[i] for i in order],
                    orientation="h",
                    marker_color=[ui.FAULT_COLORS[FAULTS[i]] for i in order],
                    text=[f"{result['proba'][i] * 100:.1f}%" for i in order],
                    textposition="outside",
                )
            )
            fig.update_layout(title="Probability of each condition", xaxis_title="Probability (%)")
            fig.update_xaxes(range=[0, 115])
            st.plotly_chart(ui.style_figure(fig, 380))
        with right:
            explainer = get_explainer(bundle["clf"], token)
            contrib = local_contributions(explainer, result["frame"], result["label_id"]).tail(8)
            fig = go.Figure(
                go.Bar(
                    x=contrib["shap"],
                    y=contrib["feature"],
                    orientation="h",
                    marker_color=[ui.RED if v > 0 else ui.BLUE for v in contrib["shap"]],
                    customdata=contrib["value"],
                    hovertemplate="%{y} = %{customdata:.2f}<br>SHAP %{x:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"What pushed the model toward {label}",
                xaxis_title="SHAP value (red raises the odds, blue lowers them)",
            )
            st.plotly_chart(ui.style_figure(fig, 380))

    with tab_b:
        st.markdown("Pick one reading and slide it across its full range. Everything else stays at the values you set above.")
        sweep_name = st.selectbox(
            "Reading to vary",
            list(FIELDS),
            format_func=lambda k: FIELDS[k][0],
            key="sweep_name",
        )
        label_s, lo, hi, _ = FIELDS[sweep_name]
        grid = np.linspace(lo, hi, 60)
        base = read_inputs()
        sweep = add_dc_power(pd.DataFrame([{**base, sweep_name: float(v)} for v in grid]))
        power_curve = bundle["reg"].predict(sweep[bundle["reg_features"]])
        probs = bundle["clf"].predict_proba(sweep[bundle["clf_features"]])
        current = base[sweep_name]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=power_curve, mode="lines", line=dict(color=ui.BLUE, width=3), name="Expected power"))
        fig.add_vline(x=current, line_dash="dash", line_color=ui.SUN, annotation_text="your value")
        fig.update_layout(title=f"Expected power as {label_s.lower()} changes", xaxis_title=label_s, yaxis_title="Power (kW)")
        st.plotly_chart(ui.style_figure(fig, 360))

        fig = go.Figure()
        for i, fault in enumerate(FAULTS):
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=probs[:, i] * 100,
                    mode="lines",
                    stackgroup="one",
                    name=fault,
                    line=dict(width=0.5, color=ui.FAULT_COLORS[fault]),
                )
            )
        fig.add_vline(x=current, line_dash="dash", line_color=ui.TEXT)
        fig.update_layout(title=f"How the diagnosis shifts as {label_s.lower()} changes", xaxis_title=label_s, yaxis_title="Probability (%)", hovermode="x unified")
        st.plotly_chart(ui.style_figure(fig, 380))


def page_fleet(bundle, token):
    st.markdown(
        ui.masthead(
            "Fleet dashboard",
            "A simulated plant of 2,500 panels, scored by the trained models. Filter by condition to see where output is being lost.",
        ),
        unsafe_allow_html=True,
    )

    data = fleet_data(bundle, token)
    chosen = st.multiselect("Conditions to show", FAULTS, default=FAULTS)
    view = data[data["predicted_fault"].isin(chosen)] if chosen else data.iloc[0:0]

    if view.empty:
        st.info("Pick at least one condition to see the dashboard.")
        return

    flagged = (view["status"] == "Fault flagged").mean() * 100
    top_fault = view["predicted_fault"].value_counts().idxmax()
    a, b, c, d = st.columns(4)
    a.markdown(ui.stat("Panels in view", f"{len(view):,}"), unsafe_allow_html=True)
    b.markdown(ui.stat("Flagged with a fault", f"{flagged:.1f}%"), unsafe_allow_html=True)
    c.markdown(ui.stat("Average output", f'{view["dc_power_kw"].mean():.2f} kW'), unsafe_allow_html=True)
    d.markdown(ui.stat("Most common condition", top_fault), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        counts = view["predicted_fault"].value_counts().reset_index()
        counts.columns = ["fault", "panels"]
        fig = px.pie(
            counts,
            names="fault",
            values="panels",
            hole=0.55,
            color="fault",
            color_discrete_map=ui.FAULT_COLORS,
            title="Condition mix",
        )
        st.plotly_chart(ui.style_figure(fig, 380))
    with right:
        axis_options = {
            "irradiance": "Irradiance (W/m²)",
            "dc_power_kw": "DC power (kW)",
            "module_temp": "Module temperature (°C)",
            "ambient_temp": "Ambient temperature (°C)",
            "dust_index": "Dust index",
            "panel_age": "Panel age (years)",
            "inverter_temp": "Inverter temperature (°C)",
        }
        pick_x, pick_y = st.columns(2)
        x_axis = pick_x.selectbox("X axis", list(axis_options), format_func=lambda k: axis_options[k], index=0)
        y_axis = pick_y.selectbox("Y axis", list(axis_options), format_func=lambda k: axis_options[k], index=1)
        sample = view.sample(min(len(view), 900), random_state=1)
        fig = px.scatter(
            sample,
            x=x_axis,
            y=y_axis,
            color="predicted_fault",
            color_discrete_map=ui.FAULT_COLORS,
            opacity=0.7,
            hover_data=["panel_id"],
            title=f"{axis_options[y_axis]} against {axis_options[x_axis]}",
            labels={x_axis: axis_options[x_axis], y_axis: axis_options[y_axis], "predicted_fault": "Condition"},
        )
        st.plotly_chart(ui.style_figure(fig, 330))

    left, right = st.columns(2)
    with left:
        hourly = view.assign(hour_bin=view["hour"].round().astype(int))
        hourly = hourly.groupby(["hour_bin", "status"], as_index=False)["dc_power_kw"].mean()
        fig = px.line(
            hourly,
            x="hour_bin",
            y="dc_power_kw",
            color="status",
            markers=True,
            color_discrete_map={"Healthy": ui.GREEN, "Fault flagged": ui.RED},
            title="Average output through the day",
            labels={"hour_bin": "Hour of day", "dc_power_kw": "DC power (kW)", "status": ""},
        )
        st.plotly_chart(ui.style_figure(fig, 360))
    with right:
        fig = px.box(
            view,
            x="predicted_fault",
            y="module_temp",
            color="predicted_fault",
            color_discrete_map=ui.FAULT_COLORS,
            title="Module temperature by condition",
            labels={"predicted_fault": "", "module_temp": "Module temperature (°C)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(ui.style_figure(fig, 360))

    st.markdown("#### Panels that need attention first")
    view = view.assign(shortfall_kw=view["predicted_kw"] - view["dc_power_kw"])
    attention_all = view[view["status"] == "Fault flagged"].sort_values("shortfall_kw", ascending=False)
    attention = attention_all.head(12)
    table = attention[
        ["panel_id", "predicted_fault", "irradiance", "module_temp", "dc_power_kw", "predicted_kw", "shortfall_kw"]
    ].rename(
        columns={
            "panel_id": "Panel",
            "predicted_fault": "Condition",
            "irradiance": "Irradiance",
            "module_temp": "Module temp",
            "dc_power_kw": "Measured kW",
            "predicted_kw": "Expected kW",
            "shortfall_kw": "Shortfall kW",
        }
    )
    st.dataframe(table.round(2), hide_index=True)
    st.download_button(
        "Download flagged panels as CSV",
        attention_all.to_csv(index=False),
        file_name="solarsense_flagged_panels.csv",
        mime="text/csv",
    )


def page_lab(bundle, token):
    st.markdown(
        ui.masthead(
            "Model lab",
            "How the models were built, how well they score on unseen data, and which readings they lean on.",
        ),
        unsafe_allow_html=True,
    )

    best = bundle["clf_scores"][bundle["clf_name"]]
    reg = bundle["reg_scores"][bundle["reg_name"]]
    a, b, c, d = st.columns(4)
    a.markdown(ui.stat("Fault detector", bundle["clf_name"], "picked on macro F1"), unsafe_allow_html=True)
    b.markdown(ui.stat("Accuracy", f'{best["accuracy"] * 100:.1f}%', f'macro F1 {best["f1_macro"]:.3f}'), unsafe_allow_html=True)
    c.markdown(ui.stat("Power model", bundle["reg_name"].replace(" Regressor", "")), unsafe_allow_html=True)
    d.markdown(ui.stat("R² score", f'{reg["r2"]:.3f}', f'MAE {reg["mae"]:.3f} kW'), unsafe_allow_html=True)

    st.markdown("#### Candidates compared")
    left, right = st.columns(2)
    clf_table = pd.DataFrame(bundle["clf_scores"]).T.rename(columns={"accuracy": "Accuracy", "f1_macro": "Macro F1"})
    reg_table = pd.DataFrame(bundle["reg_scores"]).T.rename(columns={"r2": "R²", "mae": "MAE (kW)", "rmse": "RMSE (kW)"})
    left.dataframe(clf_table.round(4))
    right.dataframe(reg_table.round(4))

    left, right = st.columns(2)
    with left:
        fig = px.imshow(
            bundle["confusion"],
            x=bundle["classes"],
            y=bundle["classes"],
            text_auto=True,
            color_continuous_scale=[[0, ui.CARD], [1, ui.CYAN]],
            labels=dict(x="Predicted", y="Actual", color="Panels"),
            title="Confusion matrix on the test set",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(ui.style_figure(fig, 420))
    with right:
        explainer = get_explainer(bundle["clf"], token)
        importance = global_importance(explainer, bundle["background"])
        fig = go.Figure(go.Bar(x=importance["importance"], y=importance["feature"], orientation="h", marker_color=ui.SUN))
        fig.update_layout(title="Global SHAP importance", xaxis_title="Mean absolute SHAP value")
        st.plotly_chart(ui.style_figure(fig, 420))

    actual = bundle["reg_test"]["actual"]
    predicted = bundle["reg_test"]["predicted"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", marker=dict(color=ui.BLUE, opacity=0.5), name="Test panels"))
    top = float(max(actual.max(), predicted.max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=ui.SUN, dash="dash"), name="Perfect fit"))
    fig.update_layout(title="Power prediction on the test set", xaxis_title="Actual power (kW)", yaxis_title="Predicted power (kW)")
    st.plotly_chart(ui.style_figure(fig, 400))

    st.markdown("#### Retrain from scratch")
    st.write(
        "This generates a fresh synthetic dataset, retrains all four candidates, keeps the best of each task and replaces the saved Joblib file."
    )
    samples = st.slider("Training rows", 4000, 30000, int(bundle["n_samples"]), step=2000)
    if st.button("Retrain models"):
        with st.spinner("Training. Please keep this tab open."):
            train_all(samples)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


def page_about():
    st.markdown(
        ui.masthead(
            "How SolarSense works",
            "A small, honest pipeline from simulated sensor data to a deployed diagnosis tool.",
        ),
        unsafe_allow_html=True,
    )
    steps = [
        ("Generate data", "NumPy and Pandas simulate a day of readings for thousands of panels, including six conditions: normal, soiling, partial shading, hotspot, inverter fault and degradation."),
        ("Train", "Random Forest and XGBoost compete to classify faults. Random Forest and Gradient Boosting regressors compete to predict power. The best of each is kept. Training runs in Google Colab or on first launch of the app."),
        ("Explain", "SHAP breaks every prediction into the sensor readings that pushed it, so a verdict is never a black box."),
        ("Save and serve", "Joblib stores the winning models in one file. Streamlit loads it and draws the dashboard with Plotly."),
        ("Ship", "The code lives on GitHub and Streamlit Community Cloud redeploys it on every push."),
    ]
    for title, body in steps:
        st.markdown(f"**{title}**")
        st.markdown(f'<div class="panel-note">{body}</div>', unsafe_allow_html=True)
        st.write("")
    st.caption("All data in this project is synthetic. Replace it with your plant's real telemetry before relying on any result.")


def main():
    bundle = get_bundle()
    token = f'{bundle["clf_name"]}-{bundle["n_samples"]}'

    with st.sidebar:
        st.markdown("## ☀️ SolarSense AI")
        st.caption("Fault detection and power forecasting for solar arrays")
        page = st.radio("Go to", ["Panel sensors", "Array monitor", "Panel diagnosis", "Fleet dashboard", "Model lab", "How it works"])
        st.markdown("---")
        st.caption(f'Detector: {bundle["clf_name"]}')
        st.caption(f'Power model: {bundle["reg_name"]}')

    if page == "Panel sensors":
        page_panel(bundle, token)
    elif page == "Array monitor":
        page_array(bundle, token)
    elif page == "Panel diagnosis":
        page_diagnosis(bundle, token)
    elif page == "Fleet dashboard":
        page_fleet(bundle, token)
    elif page == "Model lab":
        page_lab(bundle, token)
    else:
        page_about()


main()
