import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- 1. SET UP THE WEB PAGE PANEL ---
st.set_page_config(page_title="GelNB-DES Bone Regeneration Predictor", layout="centered")

st.title("🦷 GelNB-DES Hydrogel Platform")
st.subheader("Predictive Modeling for Dental Bone Regeneration Maturation")
st.write("Adjust the formulation parameters below to predict mechanical strength and degradation kinetics.")

# --- 2. TRAIN THE BASE MODEL (Using your lab dataset) ---
@st.cache_data # This keeps the AI model loaded in the background so it responds instantly
def train_biomaterial_model():
    data = {
        'GelNB_percent': [5.0, 10.0, 7.5, 12.0, 5.0],
        'DES_percent':   [10.0, 20.0, 15.0, 5.0,  25.0],
        'LAP_mM':        [1.0,  2.0,  1.5,  2.0,  1.0],
        'Light_secs':    [30.0, 60.0, 45.0, 60.0, 30.0],
        'Compressive_MPa': [0.8, 3.4, 2.1, 4.2, 0.5],
        'Degradation_pct': [65.0, 20.0, 35.0, 12.0, 80.0]
    }
    df = pd.DataFrame(data)
    X = df[['GelNB_percent', 'DES_percent', 'LAP_mM', 'Light_secs']]
    Y = df[['Compressive_MPa', 'Degradation_pct']]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)
    return model

model = train_biomaterial_model()

# --- 3. CREATE INTERACTIVE SLIDERS FOR THE USER ---
st.sidebar.header("🧪 Adjust Hydrogel Formulation Recipe")

user_gelnb = st.sidebar.slider("GelNB Concentration (%)", min_value=3.0, max_value=15.0, value=9.0, step=0.5)
user_des = st.sidebar.slider("Deep Eutectic Solvent (%)", min_value=0.0, max_value=30.0, value=18.0, step=1.0)
user_lap = st.sidebar.slider("LAP Concentration (mM)", min_value=0.5, max_value=3.0, value=1.8, step=0.1)
user_light = st.sidebar.slider("Blue Light Exposure (seconds)", min_value=10, max_value=90, value=50, step=5)

# --- 4. EXECUTE AI PREDICTION LIVE ---
new_recipe = pd.DataFrame([{
    'GelNB_percent': user_gelnb,
    'DES_percent': user_des,
    'LAP_mM': user_lap,
    'Light_secs': user_light
}])

prediction = model.predict(new_recipe)
predicted_mpa = prediction[0][0]
predicted_deg = prediction[0][1]

# --- 5. DISPLAY PRESENTABLE RESULTS TO THE USER ---
st.markdown("---")
st.markdown("### 📊 AI Virtual Lab Predictions")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted Compressive Strength", value=f"{predicted_mpa:.2f} MPa")
    if predicted_mpa < 1.0:
        st.warning("⚠️ May be too weak for load-bearing alveolar bone defects.")
    elif 1.0 <= predicted_mpa <= 4.0:
        st.success("✅ Favorable range for early-stage dental bone scaffold integration.")
    else:
        st.info("💡 High structural rigidity; ensure porosity remains adequate for cellular infiltration.")

with col2:
    st.metric(label="Predicted 14-Day Degradation Rate", value=f"{predicted_deg:.1f}%")
    if predicted_deg > 50.0:
        st.danger("❌ Fast degradation! Gel may collapse before osteoblasts deposit native bone matrix.")
    else:
        st.success("✅ Gradual degradation supports structural stability during bone maturation.")
