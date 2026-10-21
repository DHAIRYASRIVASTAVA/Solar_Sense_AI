import plotly.graph_objects as go

BG = "#0B1626"
CARD = "#13243A"
EDGE = "#25405E"
TEXT = "#EAF1F8"
MUTED = "#A9BCD0"
BLUE = "#4C9AFF"
CYAN = "#22D3EE"
SUN = "#FFB703"
RED = "#FF6B6B"
GREEN = "#2ED8A3"
INK = TEXT
SLATE = MUTED
SKY = BG

FAULT_COLORS = {
    "Normal": GREEN,
    "Soiling": "#F4C430",
    "Partial Shading": "#A78BFA",
    "Hotspot": RED,
    "Inverter Fault": "#FF9F43",
    "Degradation": "#94A3B8",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

.stApp {{
    background: radial-gradient(1200px 500px at 85% -10%, #16365C 0%, {BG} 60%);
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.05rem;
    color: {TEXT};
}}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
    font-family: 'Outfit', sans-serif;
    color: {TEXT};
    letter-spacing: -0.01em;
}}

.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCaptionContainer"] {{
    color: {TEXT};
}}

.stApp [data-testid="stCaptionContainer"] {{
    color: {MUTED};
}}

.stApp [data-testid="stWidgetLabel"] p {{
    font-weight: 600;
    font-size: 1rem;
}}

.stApp [data-testid="stSliderThumbValue"] {{
    color: {SUN};
    font-weight: 700;
}}

.stApp [data-testid="stSliderTickBarMin"], .stApp [data-testid="stSliderTickBarMax"] {{
    color: {MUTED};
}}

.stApp [data-baseweb="select"] > div {{
    background: {CARD};
    border: 1.5px solid {EDGE};
    transition: border-color 0.15s ease;
}}

.stApp [data-baseweb="select"] > div:hover {{
    border-color: {CYAN};
}}

.stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] input {{
    color: {TEXT};
}}

.stApp [data-baseweb="tag"] {{
    background: {BLUE};
}}

.stApp [data-baseweb="tag"] span {{
    color: #06121F;
    font-weight: 600;
}}

.stApp button[data-baseweb="tab"] {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.08rem;
    color: {MUTED};
}}

.stApp button[data-baseweb="tab"]:hover {{
    color: {CYAN};
}}

.stApp button[data-baseweb="tab"][aria-selected="true"] {{
    color: {SUN};
}}

.stApp [data-baseweb="tab-highlight"] {{
    background-color: {SUN};
    height: 4px;
}}

.stApp section[data-testid="stSidebar"] {{
    background: #07101D;
    border-right: 1px solid {EDGE};
}}

.stApp section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}

.stApp section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
    color: {MUTED};
}}

.stApp section[data-testid="stSidebar"] .stRadio label {{
    padding: 9px 10px;
    border-radius: 10px;
    font-size: 1.08rem;
    font-weight: 500;
    transition: background 0.15s ease;
}}

.stApp section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {CARD};
}}

.stApp hr {{
    border-color: {EDGE};
}}

.block-container {{
    padding-top: 1.6rem;
    max-width: 1180px;
}}

.masthead {{
    background: linear-gradient(120deg, #123259 0%, #0F2A4D 50%, #1C5191 100%);
    border: 1px solid {EDGE};
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
    color: #D3E2F2;
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
    background: {CARD};
    border: 1px solid {EDGE};
    border-left: 6px solid {CYAN};
    border-radius: 4px 14px 14px 4px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}

.stat:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(34, 211, 238, 0.18);
}}

div[data-testid="stColumn"]:nth-child(2) .stat {{
    border-left-color: {SUN};
}}

div[data-testid="stColumn"]:nth-child(3) .stat {{
    border-left-color: {GREEN};
}}

div[data-testid="stColumn"]:nth-child(4) .stat {{
    border-left-color: {BLUE};
}}

.stat .label {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {MUTED};
    margin-bottom: 2px;
}}

.stat .figure {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.15;
}}

.stat .note {{
    font-size: 0.9rem;
    color: {MUTED};
    margin-top: 2px;
}}

.verdict {{
    border-radius: 14px;
    padding: 20px 22px;
    background: {CARD};
    border: 1px solid {EDGE};
    border-top: 8px solid {BLUE};
}}

.stApp .verdict h2 {{
    margin: 0 0 6px 0;
    font-size: 1.9rem;
    color: #FFFFFF;
}}

.stApp .verdict p {{
    margin: 0;
    color: #D3E2F2;
    font-size: 1.08rem;
}}

.stApp .panel-note {{
    background: {CARD};
    border: 1px solid {EDGE};
    border-left: 6px solid {SUN};
    border-radius: 12px;
    padding: 14px 18px;
    color: {TEXT};
}}

div[data-testid="stPlotlyChart"] {{
    background: {CARD};
    border: 1px solid {EDGE};
    border-radius: 14px;
    padding: 6px;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}}

div[data-testid="stPlotlyChart"]:hover {{
    border-color: {BLUE};
    box-shadow: 0 10px 26px rgba(76, 154, 255, 0.18);
}}

.stApp .stButton > button, .stApp .stDownloadButton > button {{
    background: {SUN};
    color: #1A1200;
    border: none;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.55rem 1.3rem;
    transition: transform 0.12s ease, background 0.12s ease, box-shadow 0.12s ease;
}}

.stApp .stButton > button p, .stApp .stDownloadButton > button p {{
    color: #1A1200;
    font-weight: 700;
}}

.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {{
    background: #FFC933;
    color: #1A1200;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(255, 183, 3, 0.35);
}}

.stApp .stButton > button:focus-visible, .stApp .stDownloadButton > button:focus-visible {{
    outline: 3px solid {CYAN};
}}
</style>
"""

SUN_ART = f"""
<svg width="260" height="150" viewBox="0 0 260 150" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="190" cy="40" r="46" fill="{SUN}"/>
  <circle cx="190" cy="40" r="62" fill="none" stroke="{SUN}" stroke-opacity="0.45" stroke-width="2"/>
  <circle cx="190" cy="40" r="82" fill="none" stroke="{SUN}" stroke-opacity="0.22" stroke-width="2"/>
  <g transform="translate(110 96) skewX(-18)">
    <rect x="0" y="0" width="120" height="48" rx="3" fill="{BLUE}"/>
    <g stroke="{BG}" stroke-width="1.5" opacity="0.8">
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
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family="Source Sans 3, sans-serif", color=TEXT, size=14),
        title=dict(font=dict(family="Outfit, sans-serif", size=18, color="#FFFFFF")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=BG, bordercolor=CYAN, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=EDGE, zerolinecolor=EDGE, linecolor=EDGE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=EDGE, zerolinecolor=EDGE, linecolor=EDGE, tickfont=dict(color=MUTED))
    return fig


def gauge(value, title):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number=dict(suffix="%", font=dict(family="Outfit", size=40, color="#FFFFFF")),
            title=dict(text=title, font=dict(size=16, color=TEXT)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=MUTED),
                bar=dict(color=CYAN),
                bgcolor=EDGE,
                borderwidth=0,
            ),
        )
    )
    return style_figure(fig, 250)
