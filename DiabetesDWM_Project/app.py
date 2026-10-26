"""
app.py
======
Premium interactive Streamlit dashboard for diabetes prediction.

Features:
  • Real‑time prediction with animated gauge & risk breakdown
  • Patient radar‑chart profile vs. population averages
  • Full dataset explorer with interactive Plotly charts
  • Batch CSV prediction upload
  • BMI / Glucose / BP health indicator badges
  • Personalised health tips
  • Prediction history across session
  • Dark glassmorphism UI with micro‑animations

Usage:
    streamlit run app.py
"""

import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
DATA_PATH = BASE_DIR / "diabetes_dataset.csv"
PLOTS_DIR = BASE_DIR / "plots"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor · XGBoost AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS — glassmorphism dark theme with micro‑animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ========== FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ========== GLOBAL ========== */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4,
    section[data-testid="stSidebar"] label {
        color: #d0d0f0 !important;
    }

    /* ========== HERO ========== */
    .hero-card {
        background: linear-gradient(135deg, #6C63FF 0%, #a855f7 40%, #ec4899 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 400px;
        height: 400px;
        border-radius: 50%;
        background: rgba(255,255,255, 0.06);
        animation: floatBlob 8s ease-in-out infinite;
    }
    .hero-card::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -20%;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: rgba(255,255,255, 0.04);
        animation: floatBlob 12s ease-in-out infinite reverse;
    }
    @keyframes floatBlob {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, -20px) scale(1.1); }
    }
    .hero-card h1 { font-size: 2.2rem; margin-bottom: 0.2rem; position: relative; z-index: 1; }
    .hero-card p  { opacity: 0.9; font-size: 1rem; position: relative; z-index: 1; }
    .hero-subtitle { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.5rem; position: relative; z-index: 1; }
    .hero-badge {
        background: rgba(255,255,255,0.18);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* ========== GLASS CARDS ========== */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.2);
        border-color: rgba(108, 99, 255, 0.3);
    }
    .glass-card h3 { margin: 0 0 0.3rem 0; font-size: 0.82rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; color: #a0a0c0; }
    .glass-card .value { font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; }
    .glass-card .sub { font-size: 0.78rem; opacity: 0.5; margin-top: 0.2rem; }

    .card-safe .value { color: #00e6a0; }
    .card-risk .value { color: #ff6b6b; }
    .card-warn .value { color: #feca57; }
    .card-info .value { color: #74b9ff; }
    .card-purple .value { color: #a29bfe; }

    /* ========== RESULT BANNERS ========== */
    .result-banner {
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        font-weight: 600;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 1.05rem;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .banner-danger {
        background: linear-gradient(135deg, rgba(255,107,107,0.2), rgba(214,48,49,0.15));
        border: 1px solid rgba(255,107,107,0.3);
        color: #ff6b6b;
    }
    .banner-safe {
        background: linear-gradient(135deg, rgba(0,230,160,0.15), rgba(0,184,148,0.1));
        border: 1px solid rgba(0,230,160,0.25);
        color: #00e6a0;
    }

    /* ========== HEALTH BADGES ========== */
    .health-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0; }
    .badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-green { background: rgba(0,230,160,0.15); color: #00e6a0; border: 1px solid rgba(0,230,160,0.3); }
    .badge-yellow { background: rgba(254,202,87,0.15); color: #feca57; border: 1px solid rgba(254,202,87,0.3); }
    .badge-red { background: rgba(255,107,107,0.15); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.3); }
    .badge-blue { background: rgba(116,185,255,0.15); color: #74b9ff; border: 1px solid rgba(116,185,255,0.3); }

    /* ========== TIPS CARD ========== */
    .tips-card {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .tips-card h4 { color: #a29bfe; margin-bottom: 0.6rem; }
    .tips-card ul { margin: 0; padding-left: 1.2rem; }
    .tips-card li { margin: 0.3rem 0; font-size: 0.9rem; opacity: 0.85; }

    /* ========== SECTION HEADERS ========== */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 1.8rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .section-header .accent {
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #6C63FF, #a855f7);
        border-radius: 4px;
    }

    /* ========== STAT ROW (dataset overview) ========== */
    .stat-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .stat-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        flex: 1;
        min-width: 140px;
        text-align: center;
    }
    .stat-pill .stat-val { font-size: 1.5rem; font-weight: 700; color: #a29bfe; }
    .stat-pill .stat-label { font-size: 0.75rem; opacity: 0.5; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ========== HISTORY TABLE ========== */
    .history-row {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }

    /* ========== MISC ========== */
    footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# LOAD RESOURCES
# ===========================================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


model = load_model()
dataset = load_dataset()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# ===========================================================================
# FEATURE METADATA
# ===========================================================================
FEATURES = {
    "Gender":                   {"min": 0,    "max": 1,    "default": 1,    "step": 1,    "icon": "🧑", "unit": "(0=Male,1=Female)", "type": int},
    "Glucose":                  {"min": 44,   "max": 199,  "default": 110,  "step": 1,    "icon": "🍬", "unit": "mg/dL",  "type": int},
    "BloodPressure":            {"min": 30,   "max": 122,  "default": 72,   "step": 1,    "icon": "💓", "unit": "mm Hg",  "type": int},
    "SkinThickness":            {"min": 7,    "max": 63,   "default": 26,   "step": 1,    "icon": "📏", "unit": "mm",     "type": int},
    "Insulin":                  {"min": 14,   "max": 846,  "default": 120,  "step": 1,    "icon": "💉", "unit": "µU/mL",  "type": int},
    "BMI":                      {"min": 18.0, "max": 60.0, "default": 30.0, "step": 0.1,  "icon": "⚖️", "unit": "kg/m²",  "type": float},
    "DiabetesPedigreeFunction": {"min": 0.08, "max": 2.42, "default": 0.47, "step": 0.01, "icon": "🧬", "unit": "",       "type": float},
    "Age":                      {"min": 21,   "max": 81,   "default": 33,   "step": 1,    "icon": "🎂", "unit": "years",  "type": int},
}

FEATURE_NAMES = list(FEATURES.keys())


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
def bmi_category(bmi: float) -> tuple:
    """Return (category, css_class)."""
    if bmi < 18.5:   return "Underweight", "badge-blue"
    if bmi < 25.0:   return "Normal",      "badge-green"
    if bmi < 30.0:   return "Overweight",  "badge-yellow"
    return "Obese", "badge-red"


def glucose_category(g: int) -> tuple:
    if g < 100:  return "Normal",       "badge-green"
    if g < 126:  return "Pre‑diabetic", "badge-yellow"
    return "Diabetic range", "badge-red"


def bp_category(bp: int) -> tuple:
    if bp < 80:   return "Normal",   "badge-green"
    if bp < 90:   return "Elevated", "badge-yellow"
    return "High", "badge-red"


def risk_level(prob: float) -> tuple:
    """Return (label, color_token)."""
    if prob < 0.25: return "Low",      "card-safe"
    if prob < 0.50: return "Moderate", "card-warn"
    if prob < 0.75: return "High",     "card-risk"
    return "Very High", "card-risk"


def render_gauge(probability: float):
    """Plotly gauge — diabetes probability."""
    if probability < 0.35:
        bar_color, zone = "#00e6a0", "Low Risk"
    elif probability < 0.60:
        bar_color, zone = "#feca57", "Moderate Risk"
    else:
        bar_color, zone = "#ff6b6b", "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 52, "color": "white", "family": "Inter"}},
        title={"text": f"Diabetes Probability  ·  {zone}", "font": {"size": 15, "color": "#aaa"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#444", "tickfont": {"color": "#999", "size": 11}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "rgba(0,230,160,0.08)"},
                {"range": [35, 60], "color": "rgba(254,202,87,0.08)"},
                {"range": [60, 100],"color": "rgba(255,107,107,0.08)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.85, "value": probability * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=30, r=30, t=70, b=10), font={"color": "white"},
    )
    return fig


def render_radar(inputs: dict, dataset: pd.DataFrame):
    """Radar chart comparing patient values to population means (normalised)."""
    means = dataset[FEATURE_NAMES].mean()
    maxs  = dataset[FEATURE_NAMES].max()

    # Normalise to 0‑1
    patient_norm = [(inputs[f] / maxs[f]) * 100 if maxs[f] else 0 for f in FEATURE_NAMES]
    pop_norm     = [(means[f] / maxs[f]) * 100 if maxs[f] else 0 for f in FEATURE_NAMES]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=patient_norm + [patient_norm[0]],
        theta=FEATURE_NAMES + [FEATURE_NAMES[0]],
        fill='toself',
        fillcolor='rgba(108,99,255,0.15)',
        line=dict(color='#6C63FF', width=2),
        name='Patient',
    ))
    fig.add_trace(go.Scatterpolar(
        r=pop_norm + [pop_norm[0]],
        theta=FEATURE_NAMES + [FEATURE_NAMES[0]],
        fill='toself',
        fillcolor='rgba(0,230,160,0.08)',
        line=dict(color='#00e6a0', width=1.5, dash='dot'),
        name='Population Avg',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#bbb")),
        ),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(font=dict(color="#ccc", size=11), bgcolor="rgba(0,0,0,0)"),
        height=380, margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


def render_feature_contribution(inputs: dict, model_pipeline, feature_names: list):
    """Approximate per‑feature contribution using simple perturbation."""
    base_input = pd.DataFrame([inputs])
    base_prob = model_pipeline.predict_proba(base_input)[0][1]

    # Use dataset medians as neutral baseline instead of zero
    if dataset is not None:
        medians = dataset[feature_names].median().to_dict()
    else:
        medians = {f: FEATURES[f]["default"] for f in feature_names}

    contributions = {}
    for feat in feature_names:
        perturbed = base_input.copy()
        perturbed[feat] = medians[feat]  # set to population median
        pert_prob = model_pipeline.predict_proba(perturbed)[0][1]
        contributions[feat] = base_prob - pert_prob  # positive = pushes toward diabetes

    sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [f[0] for f in sorted_feats]
    vals  = [f[1] for f in sorted_feats]
    colors = ['#ff6b6b' if v > 0 else '#00e6a0' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in vals],
        textposition='outside',
        textfont=dict(size=11, color="#ccc"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Contribution to Diabetes Probability", gridcolor="rgba(255,255,255,0.05)",
                   zerolinecolor="rgba(255,255,255,0.15)", tickfont=dict(color="#aaa"),
                   title_font=dict(color="#aaa")),
        yaxis=dict(tickfont=dict(color="#ccc", size=11)),
        height=350, margin=dict(l=150, r=80, t=20, b=50),
    )
    return fig


def get_health_tips(prediction: int, inputs: dict) -> list:
    """Generate personalised health tips based on inputs."""
    tips = []
    if inputs["Glucose"] >= 126:
        tips.append("🍬 Your glucose level is in the diabetic range. Consider consulting an endocrinologist.")
    elif inputs["Glucose"] >= 100:
        tips.append("🍬 Your glucose is slightly elevated. Reduce refined sugars and get regular HbA1c tests.")

    bmi_cat, _ = bmi_category(inputs["BMI"])
    if bmi_cat == "Obese":
        tips.append("⚖️ Your BMI indicates obesity. A 5‑10% weight reduction can significantly lower diabetes risk.")
    elif bmi_cat == "Overweight":
        tips.append("⚖️ You're slightly overweight. Regular exercise (150 min/week) can help manage weight.")

    if inputs["BloodPressure"] >= 90:
        tips.append("💓 Your blood pressure is high. Monitor regularly and consider reducing sodium intake.")

    if inputs["Age"] >= 45:
        tips.append("🎂 Age 45+ carries higher diabetes risk. Annual screenings are recommended.")

    if inputs["DiabetesPedigreeFunction"] >= 0.8:
        tips.append("🧬 Your genetic predisposition score is high. Early prevention is especially important for you.")

    if inputs["Insulin"] >= 300:
        tips.append("💉 Your insulin level is significantly elevated, which may indicate insulin resistance.")

    if prediction == 1:
        tips.append("🏥 This is a screening tool only. Please schedule a comprehensive evaluation with your healthcare provider.")
    else:
        tips.append("✅ Keep up a healthy lifestyle with balanced nutrition, regular exercise, and routine checkups.")

    return tips


def preprocess_batch_upload(batch_df: pd.DataFrame, reference_df: pd.DataFrame | None) -> tuple[pd.DataFrame, int, list[str]]:
    """Clean uploaded batch data before inference."""
    if reference_df is not None:
        fill_values = reference_df[FEATURE_NAMES].mean().to_dict()
    else:
        fill_values = {feature: FEATURES[feature]["default"] for feature in FEATURE_NAMES}

    missing_columns = [column for column in FEATURE_NAMES if column not in batch_df.columns]
    features_df = batch_df.reindex(columns=FEATURE_NAMES).copy()

    for column in FEATURE_NAMES:
        features_df[column] = pd.to_numeric(features_df[column], errors="coerce")

    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    features_df[zero_as_missing] = features_df[zero_as_missing].replace(0, np.nan)
    missing_before_imputation = int(features_df.isna().sum().sum())

    features_df = features_df.fillna(fill_values)
    return features_df, missing_before_imputation, missing_columns


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("## 🩺 Patient Data Input")
    st.markdown("<p style='opacity:0.6; font-size:0.85rem;'>Adjust sliders to match patient vitals</p>", unsafe_allow_html=True)
    st.markdown("---")

    inputs = {}
    for feat, meta in FEATURES.items():
        label = f"{meta['icon']}  {feat}"
        if meta["unit"]:
            label += f"  ({meta['unit']})"
        if meta["type"] == int:
            inputs[feat] = st.slider(label, min_value=meta["min"], max_value=meta["max"],
                                     value=meta["default"], step=meta["step"])
        else:
            inputs[feat] = st.slider(label, min_value=meta["min"], max_value=meta["max"],
                                     value=meta["default"], step=meta["step"], format="%.2f")

    st.markdown("---")

    # Quick presets
    st.markdown("#### ⚡ Quick Presets")
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("🟢 Healthy", width="stretch"):
            st.session_state.update({"preset": "healthy"})
            st.rerun()
    with preset_col2:
        if st.button("🔴 At-Risk", width="stretch"):
            st.session_state.update({"preset": "risk"})
            st.rerun()

    st.markdown("---")
    predict_btn = st.button("🔬  Analyse Patient", width="stretch", type="primary")

# Handle presets
if st.session_state.get("preset") == "healthy":
    inputs = {"Gender": 1, "Glucose": 88, "BloodPressure": 68, "SkinThickness": 22,
              "Insulin": 80, "BMI": 23.5, "DiabetesPedigreeFunction": 0.25, "Age": 28}
    st.session_state.pop("preset")
    predict_btn = True
elif st.session_state.get("preset") == "risk":
    inputs = {"Gender": 0, "Glucose": 165, "BloodPressure": 92, "SkinThickness": 38,
              "Insulin": 310, "BMI": 38.2, "DiabetesPedigreeFunction": 1.15, "Age": 55}
    st.session_state.pop("preset")
    predict_btn = True


# ===========================================================================
# HERO
# ===========================================================================
st.markdown("""
<div class="hero-card">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>Advanced AI‑powered screening tool using XGBoost classification</p>
    <div class="hero-subtitle">
        <span class="hero-badge">⚡ XGBoost</span>
        <span class="hero-badge">📊 1,200 Patient Dataset</span>
        <span class="hero-badge">🎯 Tuned Hyperparameters</span>
        <span class="hero-badge">🔬 Real‑time Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# MODEL CHECK
# ===========================================================================
if model is None:
    st.error("⚠️ **Model not found!** Run `python data_generation.py` then `python model_training.py` first.")
    st.stop()
if dataset is None:
    st.warning("⚠️ Dataset not loaded. Some features will be unavailable.")


# ===========================================================================
# MAIN TABS
# ===========================================================================
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    "🔬 Prediction", "📊 Dataset Explorer", "📈 Model Insights", "📤 Batch Prediction", "📜 History"
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ───────────────────────────────────────────────────────────────────────────
with main_tab1:
    if predict_btn:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        prob_diabetes = proba[1]
        prob_no_diabetes = proba[0]
        confidence = max(prob_diabetes, prob_no_diabetes)
        risk_label, risk_cls = risk_level(prob_diabetes)

        # Save to history
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "prediction": "Diabetic" if prediction == 1 else "Healthy",
            "probability": f"{prob_diabetes*100:.1f}%",
            "confidence": f"{confidence*100:.1f}%",
            "glucose": inputs["Glucose"],
            "bmi": inputs["BMI"],
        })

        # ── Result Banner ──
        if prediction == 1:
            st.markdown(
                '<div class="result-banner banner-danger">'
                '⚠️ &nbsp; <strong>HIGH RISK</strong> — The model predicts diabetes. '
                'Immediate clinical evaluation is recommended.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-banner banner-safe">'
                '✅ &nbsp; <strong>LOW RISK</strong> — The model predicts no diabetes. '
                'Continue maintaining a healthy lifestyle.</div>',
                unsafe_allow_html=True)

        # ── Health Badges ──
        bmi_cat, bmi_cls = bmi_category(inputs["BMI"])
        glu_cat, glu_cls = glucose_category(inputs["Glucose"])
        bp_cat, bp_cls   = bp_category(inputs["BloodPressure"])

        st.markdown(f"""
        <div class="health-badges">
            <span class="badge {bmi_cls}">BMI: {bmi_cat}</span>
            <span class="badge {glu_cls}">Glucose: {glu_cat}</span>
            <span class="badge {bp_cls}">BP: {bp_cat}</span>
            <span class="badge badge-blue">Age: {inputs['Age']}y</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ── Metric Cards ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card_cls = "card-risk" if prediction == 1 else "card-safe"
            label = "Diabetic" if prediction == 1 else "Healthy"
            st.markdown(f'<div class="glass-card {card_cls}"><h3>Prediction</h3><div class="value">{label}</div><div class="sub">Binary classification</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="glass-card card-info"><h3>Diabetes Probability</h3><div class="value">{prob_diabetes*100:.1f}%</div><div class="sub">Positive class score</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="glass-card card-purple"><h3>Model Confidence</h3><div class="value">{confidence*100:.1f}%</div><div class="sub">Prediction certainty</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="glass-card {"card-risk" if risk_label in ["High","Very High"] else "card-warn" if risk_label == "Moderate" else "card-safe"}"><h3>Risk Level</h3><div class="value">{risk_label}</div><div class="sub">Based on probability</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # ── Gauge + Radar side by side ──
        gauge_col, radar_col = st.columns([1, 1])
        with gauge_col:
            st.markdown('<div class="section-header"><span class="accent"></span> Probability Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(render_gauge(prob_diabetes), width="stretch")
        with radar_col:
            st.markdown('<div class="section-header"><span class="accent"></span> Patient Profile vs Population</div>', unsafe_allow_html=True)
            if dataset is not None:
                st.plotly_chart(render_radar(inputs, dataset), width="stretch")
            else:
                st.info("Dataset not available for radar comparison.")

        # ── Feature Contribution ──
        st.markdown('<div class="section-header"><span class="accent"></span> Feature Contribution Analysis</div>', unsafe_allow_html=True)
        st.caption("Shows how each feature pushes the prediction toward (red) or away from (green) diabetes.")
        st.plotly_chart(render_feature_contribution(inputs, model, FEATURE_NAMES), width="stretch")

        # ── Health Tips ──
        tips = get_health_tips(prediction, inputs)
        if tips:
            tips_html = "".join(f"<li>{t}</li>" for t in tips)
            st.markdown(f'<div class="tips-card"><h4>💡 Personalised Health Insights</h4><ul>{tips_html}</ul></div>', unsafe_allow_html=True)

        # ── Patient Input Summary ──
        st.markdown('<div class="section-header"><span class="accent"></span> Patient Input Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Value": [inputs[f] for f in FEATURE_NAMES],
            "Unit": [FEATURES[f]["unit"] for f in FEATURE_NAMES],
        })
        st.dataframe(summary_df, width="stretch", hide_index=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <p style="font-size: 3rem; margin-bottom: 0.5rem;">🔬</p>
            <h3 style="opacity: 0.7;">Adjust the patient parameters in the sidebar</h3>
            <p style="opacity: 0.4;">Then click <strong>Analyse Patient</strong> to generate a comprehensive risk report.</p>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — DATASET EXPLORER
# ───────────────────────────────────────────────────────────────────────────
with main_tab2:
    if dataset is not None:
        st.markdown('<div class="section-header"><span class="accent"></span> Dataset Overview</div>', unsafe_allow_html=True)

        # Stats pills
        total = len(dataset)
        n_diabetic = int(dataset["Outcome"].sum())
        n_healthy = total - n_diabetic
        n_features = len(FEATURE_NAMES)
        missing = int(dataset.isnull().sum().sum())

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill"><div class="stat-val">{total:,}</div><div class="stat-label">Total Patients</div></div>
            <div class="stat-pill"><div class="stat-val">{n_healthy:,}</div><div class="stat-label">Healthy (0)</div></div>
            <div class="stat-pill"><div class="stat-val">{n_diabetic:,}</div><div class="stat-label">Diabetic (1)</div></div>
            <div class="stat-pill"><div class="stat-val">{n_features}</div><div class="stat-label">Features</div></div>
            <div class="stat-pill"><div class="stat-val">{missing}</div><div class="stat-label">Missing Values</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Sub‑tabs
        dt1, dt2, dt3, dt4 = st.tabs(["📋 Data Table", "📊 Distributions", "🔗 Correlations", "📈 Pair Plots"])

        with dt1:
            # Filterable table
            col_filter, col_search = st.columns([1, 2])
            with col_filter:
                outcome_filter = st.selectbox("Filter by Outcome", ["All", "Healthy (0)", "Diabetic (1)"])
            with col_search:
                age_range = st.slider("Age Range", 21, 81, (21, 81))

            filtered = dataset.copy()
            if outcome_filter == "Healthy (0)":
                filtered = filtered[filtered["Outcome"] == 0]
            elif outcome_filter == "Diabetic (1)":
                filtered = filtered[filtered["Outcome"] == 1]
            filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

            st.dataframe(filtered, width="stretch", hide_index=True, height=400)
            st.caption(f"Showing {len(filtered)} of {total} records")

        with dt2:
            feat_select = st.selectbox("Select feature", FEATURE_NAMES, key="dist_feat")
            fig = px.histogram(
                dataset, x=feat_select, color="Outcome",
                color_discrete_map={0: "#6C63FF", 1: "#ff6b6b"},
                barmode="overlay", opacity=0.7,
                labels={"Outcome": "Class"},
                category_orders={"Outcome": [0, 1]},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(font=dict(color="#ccc")),
                font=dict(color="#ccc"),
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

        with dt3:
            corr = dataset[FEATURE_NAMES + ["Outcome"]].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), height=500,
            )
            st.plotly_chart(fig, width="stretch")

        with dt4:
            pair_x = st.selectbox("X-axis", FEATURE_NAMES, index=1, key="pair_x")
            pair_y = st.selectbox("Y-axis", FEATURE_NAMES, index=5, key="pair_y")
            fig = px.scatter(
                dataset, x=pair_x, y=pair_y, color="Outcome",
                color_discrete_map={0: "#6C63FF", 1: "#ff6b6b"},
                opacity=0.6, hover_data=["Age", "Glucose", "BMI"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(font=dict(color="#ccc")),
                font=dict(color="#ccc"), height=450,
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Dataset file `diabetes_dataset.csv` not found.")


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ───────────────────────────────────────────────────────────────────────────
with main_tab3:
    st.markdown('<div class="section-header"><span class="accent"></span> Model Performance & EDA</div>', unsafe_allow_html=True)

    mi1, mi2, mi3, mi4 = st.tabs(["🔥 Feature Importance", "🗺️ Correlation Heatmap", "📊 Feature Distributions", "📉 Confusion Matrix & ROC"])

    with mi1:
        img = PLOTS_DIR / "feature_importance.png"
        if img.exists():
            st.image(str(img), width="stretch")
        else:
            st.info("Run `python model_training.py` to generate this plot.")

        # Also show interactive version from model
        if model is not None:
            st.markdown("**Interactive Feature Importance**")
            clf = model.named_steps["clf"]
            imp = clf.feature_importances_
            imp_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": imp}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                font=dict(color="#ccc"), height=380, showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

    with mi2:
        img = PLOTS_DIR / "correlation_heatmap.png"
        if img.exists():
            st.image(str(img), width="stretch")
        else:
            st.info("Run `python model_training.py` to generate this plot.")

    with mi3:
        img = PLOTS_DIR / "feature_distributions.png"
        if img.exists():
            st.image(str(img), width="stretch")
        else:
            st.info("Run `python model_training.py` to generate this plot.")

    with mi4:
        c1, c2 = st.columns(2)
        with c1:
            img = PLOTS_DIR / "confusion_matrix.png"
            if img.exists():
                st.image(str(img), width="stretch")
            else:
                st.info("Run `python model_training.py` to generate.")
        with c2:
            img = PLOTS_DIR / "roc_curve.png"
            if img.exists():
                st.image(str(img), width="stretch")
            else:
                st.info("Run `python model_training.py` to generate.")


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — BATCH PREDICTION
# ───────────────────────────────────────────────────────────────────────────
with main_tab4:
    st.markdown('<div class="section-header"><span class="accent"></span> Batch CSV Prediction</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with the same feature columns to predict outcomes for multiple patients at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            batch_df.columns = [column.strip() for column in batch_df.columns]
            features_df, imputed_cells, auto_added_cols = preprocess_batch_upload(batch_df, dataset)
            preds = model.predict(features_df)
            probas = model.predict_proba(features_df)[:, 1]

            batch_df["Predicted_Outcome"] = preds
            batch_df["Diabetes_Probability"] = (probas * 100).round(2)
            batch_df["Risk_Level"] = batch_df["Diabetes_Probability"].apply(
                lambda p: risk_level(p / 100)[0]
            )

            # Summary
            n_risk = int((preds == 1).sum())
            n_safe = int((preds == 0).sum())

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(f'<div class="glass-card card-info"><h3>Total Patients</h3><div class="value">{len(batch_df)}</div></div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<div class="glass-card card-safe"><h3>Healthy</h3><div class="value">{n_safe}</div></div>', unsafe_allow_html=True)
            with rc3:
                st.markdown(f'<div class="glass-card card-risk"><h3>Diabetic</h3><div class="value">{n_risk}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            if auto_added_cols:
                st.warning(f"Missing columns were auto-added using dataset averages: {auto_added_cols}")
            st.caption(f"Uploaded values were cleaned with numeric coercion, zero-as-missing handling, and mean imputation before prediction. Imputed cells: {imputed_cells}.")
            st.dataframe(batch_df, width="stretch", hide_index=True, height=400)

            # Download results
            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", csv_out, "batch_predictions.csv", "text/csv",
                               width="stretch")

            # Distribution chart
            fig = px.histogram(batch_df, x="Diabetes_Probability", nbins=20,
                               color_discrete_sequence=["#6C63FF"], opacity=0.8)
            fig.update_layout(
                title="Prediction Probability Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Diabetes Probability (%)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Patient Count"),
                font=dict(color="#ccc"), height=350,
            )
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; opacity:0.5;">
            <p style="font-size: 2.5rem;">📤</p>
            <p>Drop a CSV file above or click to browse.</p>
            <p style="font-size:0.8rem;">Required columns: Gender, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age</p>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 5 — HISTORY
# ───────────────────────────────────────────────────────────────────────────
with main_tab5:
    st.markdown('<div class="section-header"><span class="accent"></span> Prediction History (Session)</div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, width="stretch", hide_index=True, height=350)

        # Quick stats
        n_total = len(hist_df)
        n_diab = len(hist_df[hist_df["prediction"] == "Diabetic"])
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill"><div class="stat-val">{n_total}</div><div class="stat-label">Total Predictions</div></div>
            <div class="stat-pill"><div class="stat-val">{n_total - n_diab}</div><div class="stat-label">Healthy</div></div>
            <div class="stat-pill"><div class="stat-val">{n_diab}</div><div class="stat-label">Diabetic</div></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️  Clear History", width="stretch"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; opacity:0.5;">
            <p style="font-size: 2.5rem;">📜</p>
            <p>No predictions yet. Run an analysis to start building your history.</p>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; opacity:0.35; font-size:0.8rem; padding-bottom:1rem;'>"
    "Built with ❤️ using XGBoost + Streamlit  ·  For educational & screening purposes only  ·  Not a substitute for medical advice"
    "</div>",
    unsafe_allow_html=True,
)
