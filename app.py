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
    page_title="Odisha Environmental Impact Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================================
# FUTURISTIC CSS
# ===========================================

st.markdown("""
<style>
html,body,.stApp{
    background:
    radial-gradient(circle at top,#123b69 0%,#08111d 40%,#050812 100%);
    background-attachment:fixed;
    color:white;
    overflow-x:hidden;
    font-family:'Segoe UI',sans-serif;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{background:transparent;}

.stApp:before{
    content:"";
    position:fixed;
    top:0;
    left:0;
    width:100%;
    height:100%;
    background-image:radial-gradient(white 1px,transparent 1px);
    background-size:45px 45px;
    opacity:.08;
    animation:starsMove 120s linear infinite;
    z-index:-5;
}

@keyframes starsMove{
    from{transform:translateY(0);}
    to{transform:translateY(-2000px);}
}

h1{
    color:#9dfcff;
    font-size:52px;
    font-weight:700;
    text-shadow:0 0 10px cyan,0 0 30px cyan;
}

h2,h3{color:white;}
p,label{color:#EDF8FF !important;font-size:16px;}

.panel{
    background:rgba(17,27,48,.55);
    backdrop-filter:blur(20px);
    border-radius:22px;
    padding:24px;
    border:1px solid rgba(110,220,255,.18);
    box-shadow:0 0 35px rgba(0,255,255,.08);
    transition:.35s;
}

.panel:hover{
    transform:translateY(-4px);
    box-shadow:0 0 45px rgba(0,255,255,.30);
}

.metric-card{
    background:rgba(20,35,55,.60);
    padding:18px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,.08);
    text-align:center;
    transition:.35s;
    box-shadow:0 0 20px rgba(0,255,255,.08);
}

.metric-card:hover{
    transform:scale(1.05);
    box-shadow:0 0 35px cyan;
}

.metric-card .label{
    font-size:13px;
    color:#BFEFFF;
    letter-spacing:1px;
}

.metric-card .value{
    font-size:28px;
    font-weight:bold;
    color:white;
    margin-top:6px;
}

.stTextInput input,
.stNumberInput input{
    background:#07131f !important;
    color:white !important;
    font-size:17px;
    border:2px solid #00ddff !important;
    border-radius:14px;
}

.stTextInput input:focus,
.stNumberInput input:focus{
    box-shadow:0 0 20px cyan;
}

div[data-baseweb="select"]>div{
    background:#07131f !important;
    border:2px solid #00ddff !important;
    color:white !important;
}

.stSlider{color:cyan;}

div.stButton>button{
    width:100%;
    height:58px;
    font-size:20px;
    font-weight:bold;
    border-radius:40px;
    border:none;
    background:linear-gradient(90deg,#00cfff,#0088ff);
    color:white;
    transition:.4s;
    box-shadow:0 0 25px rgba(0,255,255,.25);
}

div.stButton>button:hover{
    transform:scale(1.03);
    box-shadow:0 0 45px cyan;
}

[data-testid="stDataFrame"]{
    background:rgba(17,27,48,.6);
    border-radius:18px;
}

::-webkit-scrollbar{width:10px;}
::-webkit-scrollbar-thumb{background:#00ccff;border-radius:30px;}
</style>
""", unsafe_allow_html=True)

# ===========================================
# CURSOR GLITTER
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
# 3D DIGITAL EARTH
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
# TITLE
# ===========================================

st.markdown("""
<h1 style="text-align:center">🌍 Odisha Environmental Impact Predictor</h1>
<p style="text-align:center;font-size:20px;color:#CDEEFF;margin-bottom:30px;">
AI Powered Satellite-based Environmental Monitoring System
</p>
""", unsafe_allow_html=True)

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
# PLOTLY GAUGE
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
                "bar": {"color": "#00E5FF"},
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


# =====================================================
# FUTURISTIC UI
# =====================================================

st.markdown("""
<h2 style="text-align:center;color:#7DF9FF;margin-bottom:25px;">
🌍 Environmental Analysis Dashboard
</h2>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.6], gap="large")

# =====================================================
# INPUT PANEL
# =====================================================

with left_col:

    st.markdown("""
    <div class="panel">
    <h3 style="text-align:center;color:#7DF9FF;">📍 Project Details</h3>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:rgba(0,180,255,.08);
        padding:15px;
        border-radius:15px;
        border:1px solid rgba(0,255,255,.15);
        margin-bottom:20px;
        color:white;
        line-height:1.7;
    ">
    <b>Instructions</b><br><br>
    🌍 Enter a place name in Odisha
    <br><br>
    OR
    <br><br>
    📌 Enter Latitude & Longitude manually.
    <br><br>
    Google Maps → Right Click → Copy Coordinates
    </div>
    """, unsafe_allow_html=True)

    place_name = st.text_input("📍 Place Name", value="Keonjhar")
    lat_input = st.number_input("Latitude", value=0.0, format="%.4f")
    lon_input = st.number_input("Longitude", value=0.0, format="%.4f")
    area_sqm = st.number_input("Area (Square meters)", value=50000, step=1000)
    pct = st.slider("Construction Percentage", 0, 100, 10)
    land_use = st.selectbox("Land Use Type", list(land_use_impact_weights.keys()))

    st.write("")

    submit = st.button("🚀 Predict Environmental Impact", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIGHT PANEL — RESULTS
# =====================================================

with right_col:

    st.markdown("""
    <div class="panel">
    <h3 style="color:#7DF9FF;text-align:center;">🌎 Live Analysis</h3>
    """, unsafe_allow_html=True)

    if not submit:

        st.info("Fill the project information and press **Predict Environmental Impact**.")

        st.markdown("""
        <br>

        ### Features
        ✔ Satellite-based Analysis
        ✔ Machine Learning Classification
        ✔ Forest Loss Estimation
        ✔ Environmental Impact Prediction
        ✔ Land Cover Breakdown
        ✔ AI Assisted Decision Support
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

        # ============================================
        # LOCATION INFORMATION
        # ============================================

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

        # ============================================
        # IMPACT BADGE
        # ============================================

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

        # ============================================
        # METRIC CARDS
        # ============================================

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

       