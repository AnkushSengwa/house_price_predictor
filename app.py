import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import sklearn

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Custom CSS
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1420 0%, #141a2b 100%);
    }
    h1 {
        background: linear-gradient(90deg, #00c6ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .price-card {
        background: linear-gradient(135deg, #1e2a4a, #131a2f);
        border: 1px solid #2d3a5f;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .price-value {
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(90deg, #00e5a0, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .price-label {
        color: #9aa5c0;
        font-size: 14px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00c6ff, #7b61ff);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 12px 0;
        width: 100%;
        font-size: 16px;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Load model & scaler
# =========================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_scaler()

FEATURE_NAMES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

FEATURE_INFO = {
    "CRIM":    dict(label="Crime Rate",            min=0.0,   max=90.0,  default=3.6,   step=0.1),
    "ZN":      dict(label="Residential Zoning %",  min=0.0,   max=100.0, default=11.0,  step=1.0),
    "INDUS":   dict(label="Non-Retail Business %", min=0.0,   max=28.0,  default=11.0,  step=0.1),
    "CHAS":    dict(label="Bounds Charles River",  min=0,     max=1,     default=0,     step=1),
    "NOX":     dict(label="Nitric Oxide Level",    min=0.35,  max=0.90,  default=0.55,  step=0.01),
    "RM":      dict(label="Avg Rooms per Home",    min=3.5,   max=9.0,   default=6.3,   step=0.1),
    "AGE":     dict(label="% Built Before 1940",   min=0.0,   max=100.0, default=68.0,  step=1.0),
    "DIS":     dict(label="Distance to Jobs Hub",  min=1.0,   max=13.0,  default=3.8,   step=0.1),
    "RAD":     dict(label="Highway Accessibility", min=1.0,   max=24.0,  default=9.5,   step=1.0),
    "TAX":     dict(label="Property Tax Rate",     min=180.0, max=720.0, default=408.0, step=1.0),
    "PTRATIO": dict(label="Pupil-Teacher Ratio",   min=12.0,  max=22.0,  default=18.5,  step=0.1),
    "B":       dict(label="B (Demographic Index)", min=0.0,   max=400.0, default=357.0, step=1.0),
    "LSTAT":   dict(label="% Lower Status Pop.",   min=1.0,   max=38.0,  default=12.7,  step=0.1),
}

# =========================================================
# Header
# =========================================================
st.title("🏠 Boston House Price Predictor")
st.caption("Adjust the neighborhood & property features to estimate the median home value.")
st.markdown("---")

# =========================================================
# Sidebar inputs
# =========================================================
st.sidebar.header("🔧 Input Features")
st.sidebar.write("Tune each feature below:")

values = {}
for feat in FEATURE_NAMES:
    info = FEATURE_INFO[feat]
    if feat == "CHAS":
        values[feat] = st.sidebar.selectbox(f"{info['label']} (CHAS)", options=[0, 1], index=0)
    else:
        values[feat] = st.sidebar.slider(
            f"{info['label']}  ({feat})",
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["default"]),
            step=float(info["step"]),
        )

predict_clicked = st.sidebar.button("🚀 Predict Price")

# =========================================================
# Main layout
# =========================================================
col_left, col_right = st.columns([1, 1.3])

input_array = np.array([[values[f] for f in FEATURE_NAMES]])

if predict_clicked:
    scaled = scaler.transform(input_array)
    prediction = model.predict(scaled)[0]
    price_usd = prediction * 1000

    with col_left:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Median House Price</div>
            <div class="price-value">${price_usd:,.0f}</div>
            <div class="price-label" style="margin-top:8px;">Raw MEDV: {prediction:.2f} (in $1000s)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###")
        st.subheader("📋 Your Inputs")
        input_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Value": [values[f] for f in FEATURE_NAMES]
        })
        st.dataframe(input_df, hide_index=True, use_container_width=True)

    with col_right:
        # ---- Feature contribution chart (coef * scaled value) ----
        st.subheader("📊 What's Driving This Price?")
        if hasattr(model, "coef_"):
            contributions = model.coef_ * scaled[0]
            contrib_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Contribution": contributions
            }).sort_values("Contribution")

            colors = ["#ff5c7c" if v < 0 else "#00e5a0" for v in contrib_df["Contribution"]]

            fig = go.Figure(go.Bar(
                x=contrib_df["Contribution"],
                y=contrib_df["Feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=430,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Impact on Predicted Price ($1000s)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Green = pushes price up · 🔴 Red = pushes price down")

        # ---- Gauge chart for the price ----
        st.subheader("🎯 Price Gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"prefix": "$", "suffix": "k"},
            gauge={
                "axis": {"range": [0, 55]},
                "bar": {"color": "#00c6ff"},
                "steps": [
                    {"range": [0, 17], "color": "#2a1f3d"},
                    {"range": [17, 34], "color": "#1f3d3a"},
                    {"range": [34, 55], "color": "#1f3d2a"},
                ],
            },
        ))
        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(gauge, use_container_width=True)

else:
    with col_left:
        st.info("👈 Set the feature values in the sidebar and click **Predict Price** to see the result.")
    with col_right:
        st.empty()

st.markdown("---")
st.caption("Model: Linear Regression trained on the Boston Housing dataset · Scaler: StandardScaler")