import plotly.graph_objects as go

INK = "#0F2438"
BLUE = "#1F5FA8"
SUN = "#F2A900"
RED = "#C8372D"
GREEN = "#2E8B57"
SLATE = "#5B6B7C"
SKY = "#EEF3F8"

FAULT_COLORS = {
    "Normal": GREEN,
    "Soiling": "#B8860B",
    "Partial Shading": "#6C5CE7",
    "Hotspot": RED,
    "Inverter Fault": "#E67E22",
    "Degradation": "#7F8C8D",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Source Sans 3', sans-serif;
    color: {INK};
}}

.stApp {{
    background: {SKY};
}}

h1, h2, h3, h4 {{
    font-family: 'Outfit', sans-serif;
    color: {INK};
    letter-spacing: -0.01em;
}}

[data-testid="stSidebar"] {{
    background: {INK};
}}

[data-testid="stSidebar"] * {{
    color: #E8EEF5;
}}

[data-testid="stSidebar"] .stRadio label {{
    padding: 6px 4px;
    font-size: 1.02rem;
}}

.block-container {{
    padding-top: 1.6rem;
    max-width: 1180px;
}}

.masthead {{
    background: {INK};
    border-radius: 18px 18px 18px 4px;
    padding: 30px 34px 26px 34px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}}

.masthead h1 {{
    color: #FFFFFF;
    font-size: 2.3rem;
    margin: 0 0 6px 0;
}}

.masthead p {{
    color: #B9C7D6;
    font-size: 1.05rem;
    margin: 0;
    max-width: 560px;
}}

.masthead svg {{
    position: absolute;
    right: -10px;
    top: -6px;
    opacity: 0.95;
}}

.stat {{
    background: #FFFFFF;
    border-left: 5px solid {BLUE};
    border-radius: 4px 14px 14px 4px;
    padding: 14px 18px;
    margin-bottom: 10px;
}}

.stat .label {{
    font-size: 0.88rem;
    color: {SLATE};
    margin-bottom: 2px;
}}

.stat .figure {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: {INK};
    line-height: 1.15;
}}

.stat .note {{
    font-size: 0.85rem;
    color: {SLATE};
    margin-top: 2px;
}}

.verdict {{
    border-radius: 14px;
    padding: 20px 22px;
    background: #FFFFFF;
    border-top: 6px solid {BLUE};
}}

.verdict h2 {{
    margin: 0 0 4px 0;
    font-size: 1.7rem;
}}

.verdict p {{
    margin: 0;
    color: {SLATE};
}}

.panel-note {{
    background: #FFFFFF;
    border-radius: 12px;
    padding: 14px 18px;
    color: {INK};
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 6px;
}}

.stButton > button {{
    background: {SUN};
    color: {INK};
    border: none;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
}}

.stButton > button:hover {{
    background: #FFC233;
    color: {INK};
}}

.stButton > button:focus-visible {{
    outline: 3px solid {BLUE};
}}
</style>
"""

SUN_ART = f"""
<svg width="260" height="150" viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="190" cy="40" r="46" fill="{SUN}"/>
  <circle cx="190" cy="40" r="62" fill="none" stroke="{SUN}" stroke-opacity="0.35" stroke-width="2"/>
  <circle cx="190" cy="40" r="82" fill="none" stroke="{SUN}" stroke-opacity="0.18" stroke-width="2"/>
  <g transform="translate(110 96) skewX(-18)">
    <rect x="0" y="0" width="120" height="48" rx="3" fill="{BLUE}"/>
    <g stroke="{INK}" stroke-width="1.5" opacity="0.7">
      <line x1="30" y1="0" x2="30" y2="48"/>
      <line x1="60" y1="0" x2="60" y2="48"/>
      <line x1="90" y1="0" x2="90" y2="48"/>
      <line x1="0" y1="16" x2="120" y2="16"/>
      <line x1="0" y1="32" x2="120" y2="32"/>
    </g>
  </g>
</svg>
"""


def masthead(title, subtitle):
    return f'<div class="masthead">{SUN_ART}<h1>{title}</h1><p>{subtitle}</p></div>'


def stat(label, figure, note=""):
    note_html = f'<div class="note">{note}</div>' if note else ""
    return (
        f'<div class="stat"><div class="label">{label}</div>'
        f'<div class="figure">{figure}</div>{note_html}</div>'
    )


def style_figure(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Source Sans 3, sans-serif", color=INK, size=13),
        title=dict(font=dict(family="Outfit, sans-serif", size=17)),
        legend=dict(bgcolor="rgba(255,255,255,0)"),
    )
    fig.update_xaxes(gridcolor="#E6ECF2", zeroline=False)
    fig.update_yaxes(gridcolor="#E6ECF2", zeroline=False)
    return fig


def gauge(value, title):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number=dict(suffix="%", font=dict(family="Outfit", size=38)),
            title=dict(text=title, font=dict(size=15)),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=BLUE),
                bgcolor="#E6ECF2",
                borderwidth=0,
            ),
        )
    )
    return style_figure(fig, 250)
