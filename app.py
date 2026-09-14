import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import requests
import json

# --- 1. SET UP THE APPLICATION INTERFACE ---
st.set_page_config(page_title="GelNB-DES Bone Regeneration Predictor", layout="wide")

st.title("🦷 GelNB-DES Hydrogel Platform")
st.subheader("Predictive Modeling & AI Assistant for Dental Bone Regeneration")

# --- 2. CONFIGURE THE NATIVE GEMINI API CONNECTION ---
try:
    # Pull the raw text string from Secrets
    raw_secret = str(st.secrets["GOOGLE_API_KEY"])
    
    # SELF-CLEANING MECHANIC: Automatically strip out accidental text wrappers
    clean_key = raw_secret.replace("GOOGLE_API_KEY", "").replace("=", "").replace('"', "").replace("'", "").strip()
    
    # Base endpoint address targeting the modern architecture
    API_URL = "https://googleapis.com"
    api_ready = True
except Exception as e:
    st.error(f"Failed to parse secret credentials. Error: {e}")
    api_ready = False

# --- 3. TRAIN THE PREDICTIVE MODELLING AI ---
@st.cache_data
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

# --- 4. CREATE SIDEBAR RECIPE SLIDERS ---
st.sidebar.header("🧪 Adjust Hydrogel Recipe")
user_gelnb = st.sidebar.slider("GelNB Concentration (%)", 3.0, 15.0, 9.0, 0.5)
user_des = st.sidebar.slider("Deep Eutectic Solvent (%)", 0.0, 30.0, 18.0, 1.0)
user_lap = st.sidebar.slider("LAP Concentration (mM)", 0.5, 3.0, 1.8, 0.1)
user_light = st.sidebar.slider("Blue Light Exposure (seconds)", 10, 90, 50, 5)

# Calculate live mechanical predictions
new_recipe = pd.DataFrame([{'GelNB_percent': user_gelnb, 'DES_percent': user_des, 'LAP_mM': user_lap, 'Light_secs': user_light}])
prediction = model.predict(new_recipe)

# INDEXING FIX: Extract the values as single numbers out of the array matrix
predicted_mpa = float(prediction[0][0])
predicted_deg = float(prediction[0][1])

# --- 5. LAYOUT: SPLIT SCREEN INTO PREDICTIONS VS CHAT PANEL ---
left_column, right_column = st.columns(2)

with left_column:
    st.markdown("### 📊 Mechanical & Kinetic Outputs")
    
    st.metric(label="Predicted Compressive Strength", value=f"{predicted_mpa:.2f} MPa")
    if predicted_mpa < 1.0:
        st.warning("⚠️ May be structurally too weak for load-bearing alveolar bone defects.")
    elif 1.0 <= predicted_mpa <= 4.0:
        st.success("✅ Favorable modulus for early mechanical integration in jaw sockets.")
    else:
        st.info("💡 High rigidity; ensure crosslinking density doesn't completely block cell infiltration.")
        
    st.metric(label="Predicted 14-Day Degradation Rate", value=f"{predicted_deg:.1f}%")
    if predicted_deg > 50.0:
        st.danger("❌ Collapses too fast! Matrix will vanish before osteoblasts finish depositing mineralized bone.")
    else:
        st.success("✅ Gradual maturation matching natural bone remodeling timelines.")

with right_column:
    st.markdown("### 💬 Ask the Platform Assistant")
    st.write("Type a custom scientific question about GelNB, LAP crosslinking, or Deep Eutectic Solvents below:")
    
    user_question = st.text_input("Your Question:", placeholder="e.g., Why is LAP better than Irgacure for dental use?")
    
    if user_question:
        if api_ready:
            with st.spinner("Gemini is analyzing biomaterial properties..."):
                expert_prompt = (
                    "You are an elite expert AI in dental bone regeneration biomaterials. "
                    "Context: We are developing a platform using Gelatin-Norbornene (GelNB), Deep Eutectic Solvents (DES), "
                    "and LAP photoinitiator with 405nm blue light. "
                    f"Answer this question concisely and scientifically: {user_question}"
                )
                
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": clean_key
                }
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": expert_prompt
                        }]
                    }]
                }
                
                try:
                    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
                    
                    if response.status_code == 200:
                        response_json = response.json()
                        answer_text = response_json['candidates']['content']['parts']['text']
                        st.info(answer_text)
                    else:
                        st.error(f"Google API Error ({response.status_code}): {response.text}")
                except Exception as api_err:
                    st.error(f"Network Processing Fault: {api_err}")
        else:
            st.error("⚠️ AI Chat engine is offline. Check secret credentials structure configuration.")
