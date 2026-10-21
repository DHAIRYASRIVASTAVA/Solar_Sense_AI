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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

.stApp {{
    background: {SKY};
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.05rem;
}}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
    font-family: 'Outfit', sans-serif;
    color: {INK};
    letter-spacing: -0.01em;
}}

.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCaptionContainer"] {{
    color: {INK};
}}

.stApp [data-testid="stWidgetLabel"] p {{
    font-weight: 600;
    font-size: 1rem;
}}

.stApp [data-testid="stSliderThumbValue"],
.stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"] {{
    color: {INK};
    font-weight: 600;
}}

.stApp [data-baseweb="select"] > div {{
    background: #FFFFFF;
    border: 1.5px solid #B8C6D6;
}}

.stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] input {{
    color: {INK};
}}

.stApp button[data-baseweb="tab"] {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    color: {SLATE};
}}

.stApp button[data-baseweb="tab"][aria-selected="true"] {{
    color: {INK};
}}

.stApp [data-baseweb="tab-highlight"] {{
    background-color: {SUN};
    height: 4px;
}}

.stApp section[data-testid="stSidebar"] {{
    background: {INK};
}}

.stApp section[data-testid="stSidebar"] * {{
    color: #F1F5F9;
}}

.stApp section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
    color: #B9C7D6;
}}

.stApp section[data-testid="stSidebar"] .stRadio label {{
    padding: 8px 6px;
    font-size: 1.08rem;
    font-weight: 500;
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

.stApp .masthead h1 {{
    color: #FFFFFF;
    font-size: 2.3rem;
    margin: 0 0 6px 0;
}}

.stApp .masthead p {{
    color: #DCE6F0;
    font-size: 1.1rem;
    margin: 0;
    max-width: 560px;
}}

.masthead svg {{
    position: absolute;
    right: -10px;
    top: -6px;
}}

@media (max-width: 760px) {{
    .masthead svg {{
        display: none;
    }}
}}

.stat {{
    background: #FFFFFF;
    border-left: 6px solid {BLUE};
    border-radius: 4px 14px 14px 4px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}

.stat:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(15, 36, 58, 0.14);
}}

.stat .label {{
    font-size: 0.95rem;
    font-weight: 600;
    color: #3D4E60;
    margin-bottom: 2px;
}}

.stat .figure {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: {INK};
    line-height: 1.15;
}}

.stat .note {{
    font-size: 0.9rem;
    color: #3D4E60;
    margin-top: 2px;
}}

.verdict {{
    border-radius: 14px;
    padding: 20px 22px;
    background: #FFFFFF;
    border-top: 8px solid {BLUE};
}}

.stApp .verdict h2 {{
    margin: 0 0 6px 0;
    font-size: 1.9rem;
    color: {INK};
}}

.stApp .verdict p {{
    margin: 0;
    color: #2A3B4D;
    font-size: 1.08rem;
}}

.stApp .panel-note {{
    background: #FFFFFF;
    border-radius: 12px;
    padding: 14px 18px;
    color: {INK};
    border-left: 6px solid {SUN};
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 6px;
}}

.stApp .stButton > button, .stApp .stDownloadButton > button {{
    background: {SUN};
    color: {INK};
    border: none;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.55rem 1.3rem;
    transition: transform 0.12s ease, background 0.12s ease;
}}

.stApp .stButton > button p, .stApp .stDownloadButton > button p {{
    color: {INK};
    font-weight: 700;
}}

.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {{
    background: #FFC233;
    color: {INK};
    transform: translateY(-2px);
}}

.stApp .stButton > button:focus-visible, .stApp .stDownloadButton > button:focus-visible {{
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
        font=dict(family="Source Sans 3, sans-serif", color=INK, size=14),
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
