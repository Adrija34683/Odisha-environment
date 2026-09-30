import streamlit as st
import streamlit.components.v1 as components
import ee
import joblib
import numpy as np
import pandas as pd
import math
from geopy.geocoders import Nominatim
from google.oauth2 import service_account
import plotly.graph_objects as go

# ===========================================
# PAGE CONFIG
# ===========================================

st.set_page_config(
    page_title="EcoOdisha | Environmental Impact Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# ECOODISHA CSS  (matches the reference design)
# ===========================================

st.markdown("""
<style>
html,body,.stApp{
    background:linear-gradient(135deg,#0a0f2c 0%,#15154a 55%,#0c1638 100%);
    background-attachment:fixed;
    color:#EAF0FF;
    font-family:'Segoe UI','Inter',sans-serif;
}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:0 !important;max-width:100% !important;padding-left:2rem;padding-right:2rem;}

/* ---------- top bar ---------- */
.topbar{
    display:flex;justify-content:space-between;align-items:center;
    padding:18px 8px;margin:0 -1rem 10px -1rem;
    border-bottom:1px solid rgba(255,255,255,.07);
    background:linear-gradient(90deg,rgba(15,18,60,.9),rgba(60,45,140,.35));
}
.brand{display:flex;align-items:center;gap:14px;padding-left:16px;}
.brand .name{font-size:32px;font-weight:700;line-height:1;color:white;}
.brand .name span{color:#a78bfa;}
.brand .tag{font-size:14px;color:#b9c3e6;margin-top:4px;}
.pill{
    display:flex;align-items:center;gap:8px;margin-right:16px;
    padding:10px 18px;border-radius:30px;font-size:14px;
    background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
}
.pill .dot{width:9px;height:9px;border-radius:50%;background:#34d399;box-shadow:0 0 10px #34d399;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{
    background:rgba(10,14,46,.92);border-right:1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] .stRadio>div{gap:6px;}
section[data-testid="stSidebar"] .stRadio label{
    background:transparent;padding:12px 16px;border-radius:10px;width:100%;
    transition:.25s;cursor:pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(139,92,246,.15);}
section[data-testid="stSidebar"] .stRadio label:has(input:checked){
    background:linear-gradient(90deg,rgba(99,102,241,.45),rgba(59,130,246,.25));
}
section[data-testid="stSidebar"] .stRadio label>div:first-child{display:none;}
.side-foot{margin-top:180px;font-style:italic;color:#9fb0dd;font-size:15px;line-height:1.5;}

/* ---------- panels (bordered containers) ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
    background:rgba(18,24,66,.62);
    border:1px solid rgba(255,255,255,.09) !important;
    border-radius:18px !important;
    box-shadow:0 10px 40px rgba(0,0,0,.25);
    padding:10px 8px;
}
.panel-head{display:flex;align-items:center;gap:16px;padding-bottom:14px;
    border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:14px;}
.panel-head .ico{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;
    justify-content:center;font-size:24px;background:linear-gradient(135deg,#8b5cf6,#6366f1);}
.panel-head .ico.blue{background:linear-gradient(135deg,#0ea5e9,#2563eb);border-radius:50%;}
.panel-head h3{margin:0;font-size:26px;color:white;}
.panel-head p{margin:2px 0 0 0;font-size:14px;color:#b9c3e6 !important;}
.live-badge{margin-left:auto;padding:8px 14px;border-radius:20px;font-size:13px;
    background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);}
.live-badge:before{content:"●";color:#34d399;margin-right:8px;}

p,label{color:#EAF0FF !important;}
.stTextInput label,.stNumberInput label,.stSelectbox label,.stSlider label{font-weight:600;font-size:15px;}

.stTextInput input,.stNumberInput input{
    background:#f1f3f8 !important;color:#000 !important;-webkit-text-fill-color:#000 !important;
    caret-color:#000;
    border:1px solid rgba(255,255,255,.12) !important;border-radius:12px !important;
    height:48px;font-size:16px;
}
.stTextInput input::placeholder,.stNumberInput input::placeholder{color:#555 !important;-webkit-text-fill-color:#555 !important;}
.stNumberInput button{color:#000 !important;}
div[data-baseweb="select"] *{color:#000 !important;}
.stTextInput input:focus,.stNumberInput input:focus{
    border-color:#8b5cf6 !important;box-shadow:0 0 0 2px rgba(139,92,246,.35);
}
.stNumberInput button{background:#e3e7f1 !important;color:#000 !important;}
div[data-baseweb="select"]>div{
    background:#f1f3f8 !important;border:1px solid rgba(255,255,255,.12) !important;
    border-radius:12px !important;color:#000 !important;min-height:48px;
}

div.stButton>button{
    width:100%;height:58px;font-size:19px;font-weight:600;border-radius:14px;border:none;
    background:linear-gradient(90deg,#8b5cf6,#3b82f6);color:white;transition:.3s;
    box-shadow:0 8px 30px rgba(99,102,241,.35);
}
div.stButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(99,102,241,.6);color:white;}

/* ---------- features + banner ---------- */
.features{
    display:flex;gap:24px;align-items:center;padding:24px;border-radius:16px;
    background:linear-gradient(135deg,rgba(40,45,130,.55),rgba(25,35,100,.55));
    border:1px solid rgba(255,255,255,.08);
}
.features .orb{font-size:92px;filter:drop-shadow(0 0 25px rgba(56,189,248,.5));}
.features h4{margin:0 0 14px 0;font-size:28px;color:white;}
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px 40px;}
.fgrid div{font-size:16px;}
.fgrid div:before{content:"✔";color:#34d399;margin-right:12px;font-weight:bold;}
.banner{
    margin-top:18px;border-radius:16px;height:300px;padding:40px;position:relative;overflow:hidden;
    background:
      linear-gradient(90deg,rgba(6,12,30,.75),rgba(6,12,30,.1)),
      url('https://images.unsplash.com/photo-1448375240586-882707db888b?w=1400') center/cover,
      linear-gradient(135deg,#0f3d2e,#1d6b4a 50%,#0b2a3a);
}
.banner .q{color:#34d399;font-size:48px;line-height:1;}
.banner .t{font-style:italic;font-size:26px;line-height:1.35;margin-top:6px;}
.banner .bar{width:44px;height:3px;background:#34d399;margin-top:18px;}

/* ---------- metric cards (kept) ---------- */
.metric-card{
    background:rgba(25,32,85,.65);padding:18px;border-radius:16px;text-align:center;
    border:1px solid rgba(255,255,255,.09);transition:.3s;
}
.metric-card:hover{transform:translateY(-3px);box-shadow:0 0 28px rgba(139,92,246,.45);}
.metric-card .label{font-size:13px;color:#c4cdf0;letter-spacing:1px;}
.metric-card .value{font-size:28px;font-weight:bold;color:white;margin-top:6px;}

[data-testid="stDataFrame"]{background:rgba(17,27,48,.6);border-radius:14px;}
::-webkit-scrollbar{width:10px;} ::-webkit-scrollbar-thumb{background:#8b5cf6;border-radius:30px;}
</style>
""", unsafe_allow_html=True)

# ===========================================
# CURSOR GLITTER (UNCHANGED)
# ===========================================

components.html("""
<style>
.spark{
    position:fixed;
    width:7px;
    height:7px;
    border-radius:50%;
    background:#8efcff;
    pointer-events:none;
    box-shadow:0 0 15px cyan,0 0 25px cyan;
    animation:spark .8s linear forwards;
    z-index:9999;
}
@keyframes spark{
    0%{opacity:1;transform:scale(1);}
    100%{opacity:0;transform:translateY(-18px) scale(3);}
}
</style>
<script>
document.addEventListener("mousemove",(e)=>{
    let s=document.createElement("div");
    s.className="spark";
    s.style.left=e.clientX+"px";
    s.style.top=e.clientY+"px";
    document.body.appendChild(s);
    setTimeout(()=>{s.remove();},700);
});
</script>
""", height=0)

# ===========================================
# TOP BAR
# ===========================================

st.markdown("""
<div class="topbar">
  <div class="brand">
    <svg width="54" height="46" viewBox="0 0 54 46" fill="none">
      <path d="M27 42C27 26 14 12 2 10c0 18 10 28 25 32Z" fill="#6d5df6"/>
      <path d="M27 42C27 24 38 8 52 4c0 20-10 34-25 38Z" fill="#38bdf8"/>
    </svg>
    <div>
      <div class="name">Eco<span>Odisha</span></div>
      <div class="tag">Smarter Decisions for a Greener Tomorrow</div>
    </div>
  </div>
  <div class="pill"><span class="dot"></span> Environmental Impact Prediction</div>
</div>
""", unsafe_allow_html=True)

# ===========================================
# HERO  (video-style text: blur-in reveal + glass lens with RGB split)
# ===========================================

components.html("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600&family=Instrument+Serif:ital@1&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;background:transparent;overflow:hidden;font-family:'Manrope','Segoe UI',sans-serif;}
  #stage{position:relative;width:100%;height:380px;cursor:none;}
  .layer{position:absolute;left:0;top:0;width:100%;height:100%;color:#fff;}
  .hero{position:absolute;left:4%;top:30px;font-size:clamp(44px,6vw,90px);font-weight:600;
        letter-spacing:-.035em;line-height:1.02;white-space:nowrap;}
  .hero .l2{display:block;margin-left:10%;white-space:nowrap;}
  .acc{position:absolute;right:6%;top:30px;font-family:'Instrument Serif',Georgia,serif;font-style:italic;
       font-weight:400;font-size:clamp(34px,4.6vw,66px);color:#fbbf24;letter-spacing:0;}
  .w{display:inline-block;opacity:0;filter:blur(22px);transform:translateY(34px);
     animation:in 1.3s cubic-bezier(.2,.8,.2,1) forwards;}
  @keyframes in{to{opacity:1;filter:blur(0);transform:none;}}
  .sub{position:absolute;left:4%;bottom:14px;font-size:12px;line-height:1.5;color:#c9d3f3;
       letter-spacing:.04em;text-transform:uppercase;opacity:0;animation:in 1.2s 1.4s forwards;}
  #lens{position:absolute;width:190px;height:190px;border-radius:50%;overflow:hidden;pointer-events:none;
        border:1.5px solid rgba(255,255,255,.45);box-shadow:0 0 40px rgba(139,92,246,.35),inset 0 0 30px rgba(255,255,255,.12);
        background:rgba(255,255,255,.05);backdrop-filter:blur(1.5px) saturate(1.4);
        opacity:0;transition:opacity .3s;}
  #lens .inner{position:absolute;}
  #lens .layer{text-shadow:-4px 0 rgba(255,40,110,.85),4px 0 rgba(0,140,255,.85);}
  #lens .w,#lens .sub{animation:none;opacity:1;filter:none;transform:none;}
</style>

<div id="stage">
  <div class="layer" id="base">
    <div class="hero">
      <span class="w" style="animation-delay:.1s">Protecting</span><br>
      <span class="l2"><span class="w" style="animation-delay:.45s">Odisha's</span>
      <span class="w" style="animation-delay:.75s">forests</span></span>
    </div>
    <div class="acc"><span class="w" style="animation-delay:1.0s">satellite-led</span></div>
    <div class="sub">Sentinel-2 &amp; Machine Learning<br>Real-time land cover analysis</div>
  </div>
  <div id="lens"><div class="inner" id="inner"></div></div>
</div>

<script>
const stage=document.getElementById('stage'), lens=document.getElementById('lens'),
      inner=document.getElementById('inner'), base=document.getElementById('base');
const R=95, Z=1.35;
function build(){
  inner.innerHTML=''; const c=base.cloneNode(true); c.removeAttribute('id');
  c.style.width=stage.clientWidth+'px'; c.style.height=stage.clientHeight+'px';
  inner.style.width=stage.clientWidth+'px'; inner.style.height=stage.clientHeight+'px';
  inner.appendChild(c);
}
build(); window.addEventListener('resize',build);
let tx=0,ty=0,cx=0,cy=0,on=false;
stage.addEventListener('mousemove',e=>{
  const r=stage.getBoundingClientRect(); tx=e.clientX-r.left; ty=e.clientY-r.top; on=true; lens.style.opacity=1;
});
stage.addEventListener('mouseleave',()=>{on=false;lens.style.opacity=0;});
function loop(){
  cx+=(tx-cx)*.14; cy+=(ty-cy)*.14;
  lens.style.left=(cx-R)+'px'; lens.style.top=(cy-R)+'px';
  inner.style.left=(-(cx-R))+'px'; inner.style.top=(-(cy-R))+'px';
  inner.style.transformOrigin=cx+'px '+cy+'px';
  inner.style.transform='scale('+Z+')';
  requestAnimationFrame(loop);
}
loop();
</script>
""", height=390)

# ===========================================
# 3D DIGITAL EARTH (UNCHANGED)
# ===========================================

components.html("""
<div id="globeViz"></div>
<script src="https://unpkg.com/three"></script>
<script src="https://unpkg.com/globe.gl"></script>
<script>
const world=Globe()
(document.getElementById("globeViz"))
.globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
.bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
.backgroundColor('rgba(0,0,0,0)');
world.controls().autoRotate=true;
world.controls().autoRotateSpeed=0.8;
</script>
<style>
#globeViz{height:420px;width:100%;}
</style>
""", height=430)

# ===========================================
# EARTH ENGINE INITIALIZATION (UNCHANGED)
# ===========================================

@st.cache_resource
def init_ee():
    service_account_info = dict(st.secrets["gee_service_account"])
    scoped_credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/earthengine']
    )
    ee.Initialize(scoped_credentials)

init_ee()

# ===========================================
# LOAD MODEL (UNCHANGED)
# ===========================================

@st.cache_resource
def load_model():
    model = joblib.load("svm_model_rbf_c10.pkl")
    scaler = joblib.load("svm_scaler.pkl")
    return model, scaler

svm_model, scaler = load_model()

classifier_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDBI', 'BSI']

class_names = {
    0: 'Mining',
    1: 'Builtup',
    2: 'Water',
    3: 'Open Forest',
    4: 'Dense Forest'
}

land_use_impact_weights = {
    'road': 1.4,
    'mining': 1.8,
    'factory': 1.6,
    'hotel_resort': 1.1,
    'residential': 1.0,
    'agriculture_expansion': 0.85,
    'other': 1.0
}

forest_loss_rate = 1.13

# ===========================================
# CORE FUNCTIONS (UNCHANGED)
# ===========================================

def get_location_breakdown_svm(lat, lon, area_sqm):

    radius = math.sqrt(area_sqm / math.pi)
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(radius)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2025-10-01", "2026-03-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(aoi)
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")

    bsi = s2.expression(
        "((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))",
        {
            "SWIR": s2.select("B11"),
            "RED": s2.select("B4"),
            "NIR": s2.select("B8"),
            "BLUE": s2.select("B2"),
        },
    ).rename("BSI")

    stack = (
        s2.select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .addBands(ndvi)
        .addBands(ndbi)
        .addBands(bsi)
    )

    pixel_samples = stack.select(classifier_bands).sample(
        region=aoi,
        scale=10,
        geometries=False,
        numPixels=2000,
    )

    pixel_data = pixel_samples.getInfo()["features"]

    if len(pixel_data) == 0:
        return None, aoi, s2

    rows = [f["properties"] for f in pixel_data]

    pixel_df = pd.DataFrame(rows)[classifier_bands].dropna()

    pixel_scaled = scaler.transform(pixel_df)
    predictions = svm_model.predict(pixel_scaled)

    hectares_per_pixel = (10 * 10) / 10000

    breakdown = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    unique, counts = np.unique(predictions, return_counts=True)

    for cls, cnt in zip(unique, counts):
        breakdown[int(cls)] = cnt * hectares_per_pixel

    return breakdown, aoi, s2


def predict_environmental_impact(lat, lon, area_sqm, percent_for_construction, land_use_type):

    breakdown, aoi, s2 = get_location_breakdown_svm(lat, lon, area_sqm)

    if breakdown is None:
        return None

    forest_ha = breakdown[4] + breakdown[3]

    land_converted_ha = (area_sqm / 10000) * (percent_for_construction / 100)

    weight = land_use_impact_weights.get(land_use_type, 1.0)

    estimated_forest_loss = land_converted_ha * forest_loss_rate * weight

    if forest_ha > 0:
        percent_impact = min(100, (estimated_forest_loss / forest_ha) * 100)
    else:
        percent_impact = 0

    return {
        "breakdown": {class_names[k]: round(v, 2) for k, v in breakdown.items()},
        "current_forest_ha": round(forest_ha, 2),
        "land_converted_ha": round(land_converted_ha, 2),
        "estimated_forest_loss_ha": round(estimated_forest_loss, 2),
        "environmental_impact_percent": round(percent_impact, 2),
    }


# ===========================================
# PLOTLY GAUGE (colour accents only)
# ===========================================

def create_gauge(value):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 42, "color": "white"}},
            title={"text": "Environmental Impact", "font": {"size": 22, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#8b5cf6"},
                "steps": [
                    {"range": [0, 25], "color": "#2ecc71"},
                    {"range": [25, 50], "color": "#f1c40f"},
                    {"range": [50, 75], "color": "#e67e22"},
                    {"range": [75, 100], "color": "#e74c3c"},
                ],
            },
        )
    )

    fig.update_layout(
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# ===========================================
# SIDEBAR NAVIGATION
# ===========================================

with st.sidebar:
    st.write("")
    page = st.radio("Navigate", ["🏠  Home", "ℹ️  About", "⚙️  How It Works", "📊  Insights"],
                    label_visibility="collapsed")
    st.markdown('<div class="side-foot">🌿<br>Healthy forests.<br>A sustainable<br>Odisha.</div>',
                unsafe_allow_html=True)

if not page.endswith("Home"):
    content = {
        "About": "EcoOdisha estimates how a proposed project could affect forest cover in Odisha, "
                 "using Sentinel-2 imagery from Google Earth Engine and an SVM land-cover classifier.",
        "How It Works": "1) Your location and area define an area of interest. "
                        "2) Sentinel-2 pixels are sampled and classified (Mining, Builtup, Water, Open/Dense Forest). "
                        "3) Converted land × loss rate × land-use weight gives the estimated forest loss and impact %.",
        "Insights": "Higher-impact land uses (mining, factory, road) carry larger weights than residential or "
                    "agriculture expansion. Reducing the construction percentage lowers the impact linearly.",
    }[page.split("  ")[1]]
    with st.container(border=True):
        st.subheader(page)
        st.write(content)
    st.stop()

# =====================================================
# DASHBOARD
# =====================================================

left_col, right_col = st.columns([1, 1.25], gap="large")

# ---------------- INPUT PANEL ----------------

with left_col:
    with st.container(border=True):

        st.markdown("""
        <div class="panel-head"><div class="ico">📍</div>
        <div><h3>Project Details</h3>
        <p>Enter the location and project parameters to predict the environmental impact.</p></div></div>
        """, unsafe_allow_html=True)

        place_name = st.text_input("📍 Place Name", value="Keonjhar",
                                   help="Enter a place name in Odisha (e.g., Keonjhar, Rourkela, Bhubaneswar)")
        c_lat, c_lon = st.columns(2)
        with c_lat:
            lat_input = st.number_input("Latitude", value=0.0, format="%.4f")
        with c_lon:
            lon_input = st.number_input("Longitude", value=0.0, format="%.4f")
        area_sqm = st.number_input("Area (Square meters)", value=50000, step=1000)
        pct = st.slider("Construction Percentage", 0, 100, 10)
        land_use = st.selectbox("Land Use Type", list(land_use_impact_weights.keys()))

        st.write("")
        submit = st.button("✨ Analyze Impact  →", use_container_width=True)

# ---------------- RESULTS PANEL ----------------

with right_col:
    with st.container(border=True):

        st.markdown("""
        <div class="panel-head"><div class="ico blue">🌎</div>
        <div><h3>Live Analysis</h3>
        <p>Fill in the project information and click on “Analyze Impact” to see the results.</p></div>
        <div class="live-badge">Real-time | GEE + ML</div></div>
        """, unsafe_allow_html=True)

        if not submit:

            st.markdown("""
            <div class="features">
              <div class="orb">🌍</div>
              <div style="flex:1">
                <h4>⭐ Features</h4>
                <div class="fgrid">
                  <div>Satellite-based Analysis</div><div>Land Cover Breakdown</div>
                  <div>Machine Learning Classification</div><div>AI Assisted Decision Support</div>
                  <div>Forest Loss Estimation</div><div>Accurate &amp; Reliable Results</div>
                  <div>Environmental Impact Prediction</div>
                </div>
              </div>
            </div>
            <div class="banner">
              <div class="q">“</div>
              <div class="t">Better data.<br>Healthier forests.<br>A sustainable future.</div>
              <div class="bar"></div>
            </div>
            """, unsafe_allow_html=True)

        else:

            with st.spinner("🛰 Downloading Sentinel-2 imagery..."):

                if lat_input == 0 and lon_input == 0:

                    geolocator = Nominatim(user_agent="odisha_environment_dashboard")
                    location = geolocator.geocode(place_name + ", Odisha, India")

                    if location is None:
                        st.error("Location not found.")
                        st.stop()

                    lat = location.latitude
                    lon = location.longitude
                    label = place_name

                else:

                    lat = lat_input
                    lon = lon_input

                    if place_name.strip() == "":
                        label = f"{lat:.4f}, {lon:.4f}"
                    else:
                        label = place_name

                result = predict_environmental_impact(lat, lon, area_sqm, pct, land_use)

            if result is None:
                st.error("❌ No satellite data available for this location.")
                st.stop()

            impact = result["environmental_impact_percent"]

            if impact < 15:
                severity = "🟢 LOW IMPACT"
                color = "#2ecc71"
                recommendation = """
✅ The proposed project has a relatively LOW environmental impact.

• Existing forest cover remains largely unaffected.
• Development can proceed with routine environmental safeguards.
• Native tree plantation is still recommended.
"""
            elif impact < 40:
                severity = "🟡 MODERATE IMPACT"
                color = "#f1c40f"
                recommendation = """
⚠ MODERATE environmental impact detected.

• Partial forest disturbance expected.
• Consider reducing construction area.
• Maintain ecological corridors.
• Compensatory plantation is recommended.
"""
            elif impact < 70:
                severity = "🟠 HIGH IMPACT"
                color = "#e67e22"
                recommendation = """
⚠ HIGH environmental impact predicted.

• Significant vegetation loss possible.
• Reconsider project design.
• Reduce construction footprint.
• Detailed Environmental Impact Assessment (EIA) is recommended.
"""
            else:
                severity = "🔴 VERY HIGH IMPACT"
                color = "#e74c3c"
                recommendation = """
🚨 VERY HIGH ENVIRONMENTAL RISK

• Severe forest degradation expected.
• Major biodiversity loss possible.
• Relocate or redesign the project.
• Comprehensive EIA and Government approval strongly recommended.
"""

            st.success("✅ Prediction Completed Successfully")

            st.subheader("🌍 Environmental Impact Prediction")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("🌳 Current Forest Area", f"{result['current_forest_ha']} ha")
                st.metric("🏗 Land Converted", f"{result['land_converted_ha']} ha")

            with c2:
                st.metric("🌲 Estimated Forest Loss", f"{result['estimated_forest_loss_ha']} ha")
                st.metric("⚠ Environmental Impact", f"{impact} %")

            st.markdown("---")

            st.subheader("📊 Land Cover Classification")

            bd_df = pd.DataFrame(result["breakdown"].items(), columns=["Land Cover", "Area (ha)"])

            st.dataframe(bd_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            st.subheader("📈 Environmental Impact Gauge")

            gauge = create_gauge(impact)
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown("---")

            info1, info2 = st.columns(2)

            with info1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">📍 LOCATION</div>
                    <div class="value" style="font-size:20px;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            with info2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">🌎 COORDINATES</div>
                    <div class="value" style="font-size:18px;">
                        {lat:.4f}<br>{lon:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            st.markdown(f"""
            <div style="
                background:{color};
                padding:18px;
                border-radius:20px;
                text-align:center;
                font-size:28px;
                font-weight:bold;
                color:white;
                box-shadow:0 0 35px {color};
            ">
                Environmental Impact
                <br><br>
                {impact}%
                <br><br>
                {severity}
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">🌲 Forest Cover</div>
                    <div class="value">{result['current_forest_ha']} ha</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">🏗 Land Converted</div>
                    <div class="value">{result['land_converted_ha']} ha</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">🌳 Forest Loss</div>
                    <div class="value">{result['estimated_forest_loss_ha']} ha</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("### 🧭 Recommendation")
            st.markdown(recommendation)